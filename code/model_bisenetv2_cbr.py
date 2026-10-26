"""
model_bisenetv2_cbr.py
BiSeNetV2-CBR：上下文与边界精炼完整方法（BiSeNetV2 版本）。

核心创新（三模块协同）：
  1. L-ASPP：轻量级空洞空间金字塔池化，捕获多尺度上下文信息
  2. CoordAtt（CA）：坐标注意力，增强方向感知（SBNet 上已验证 +2.85%，最优轮口径）
  3. BGFR：边界引导特征精炼，推理时边界注意力仍生效

架构：
  输入 (B, 3, H, W)
    ├── Detail Branch → (B, 128, H/4, W/4)
    └── Semantic Branch → (B, 256, H/16, W/16)
            ↓
        L-ASPP(256, rates=[1,3,6])    ← 多尺度上下文
            ↓
        CoordAtt(256)                  ← 方向注意力
            ↓
        BGFR(256)                      ← 边界引导精炼（推理时仍生效！）
            ↓ (B, 256, H/16, W/16)
        Upsample → cat(detail) → Head → 分割输出

  总损失: L = 0.4×CE + 0.6×Dice + λ_bdy×BCE(boundary)

参数增量（2026-09-13 实测，单位 M）：
  精简双分支基线 0.9446 → 单独加 L-ASPP 1.2812（+0.3366）→ 单独加 CA 0.9513（+0.0067）
  完整方法（L-ASPP + CA + BGFR）= 1.3211，由此反推 BGFR ≈ +0.0332

消融矩阵（SBNet 体系，最优轮 mIoU 口径）：
  | 模型              | L-ASPP | CA  | BGFR | mIoU 与相对基线      |
  | SBNet             | ✗      | ✗   | ✗    | 70.35（基线）        |
  | SBNet+CA          | ✗      | ✓   | ✗    | 73.20（+2.85）       |
  | SBNet+L-ASPP      | ✓      | ✗   | ✗    | 76.84（+6.49）       |
  | SBNet-CBR         | ✓      | ✓   | ✓    | 77.70（+7.35）       |
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from components.laspp import LightweightASPP
from components.ca_attention import CoordinateAttention
from components.bgfr import BGFR


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class BiSeNetV2CBR(nn.Module):
    """
    BiSeNetV2-CBR：上下文与边界精炼完整方法。

    参数:
        num_classes:    分类数，默认 2（背景 + 可通行区域）
        boundary_weight: 边界损失权重 λ_bdy，默认 0.5
    """
    def __init__(self, num_classes=2, boundary_weight=0.5):
        super().__init__()
        self.boundary_weight = boundary_weight

        # ---- 浅层细节分支（与精简双分支基线一致） ----
        self.detail_branch = nn.Sequential(
            ConvBNReLU(3, 64, 3, 2, 1),
            ConvBNReLU(64, 64, 3, 1, 1),
            ConvBNReLU(64, 128, 3, 2, 1),
        )
        # ---- 深层语义分支（与精简双分支基线一致） ----
        self.semantic_branch = nn.Sequential(
            ConvBNReLU(3, 32, 3, 2, 1),
            ConvBNReLU(32, 64, 3, 2, 1),
            ConvBNReLU(64, 128, 3, 2, 1),
            ConvBNReLU(128, 256, 3, 2, 1),
        )

        # ★ CBR 三模块
        self.laspp = LightweightASPP(in_channels=256, rates=[1, 3, 6])          # 多尺度上下文
        self.ca = CoordinateAttention(inp=256, oup=256, reduction=32)            # 方向注意力
        self.bgfr = BGFR(in_channels=256, mid_channels=128, boundary_weight=boundary_weight)  # 边界精炼

        # ---- 融合头 + 分类输出（与精简双分支基线一致） ----
        self.head = nn.Sequential(
            ConvBNReLU(384, 128, 3, 1, 1),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )

    def forward(self, x, mask=None):
        """
        前向传播。

        参数:
            x:    输入图像 (B, 3, H, W)
            mask: 分割真值 (B, H, W)，训练时提供用于 BGFR 边界损失
        返回:
            训练模式: (seg_out, bdy_loss)
            推理模式: (seg_out,)
        """
        _, _, H, W = x.shape
        detail = self.detail_branch(x)
        semantic = self.semantic_branch(x)

        # ★ CBR 三模块
        semantic = self.laspp(semantic)       # 1) 多尺度上下文
        semantic = self.ca(semantic)          # 2) 方向注意力
        # 3) 边界引导特征精炼
        if self.training and mask is not None:
            semantic, bdy_loss = self.bgfr(semantic, mask)
        else:
            result = self.bgfr(semantic)
            semantic = result[0]
            bdy_loss = None

        # 上采样 + 融合
        semantic_up = F.interpolate(semantic, size=detail.shape[2:], mode='bilinear', align_corners=False)
        fuse = torch.cat([detail, semantic_up], dim=1)
        out = self.head(fuse)
        seg_out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        if self.training and mask is not None:
            return seg_out, bdy_loss
        else:
            return (seg_out,)


def get_bisenetv2_cbr_2class(device):
    """返回 BiSeNetV2-CBR 二分类模型"""
    model = BiSeNetV2CBR(num_classes=2)
    model = model.to(device)
    return model
