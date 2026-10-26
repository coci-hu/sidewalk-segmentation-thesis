"""
model_enet_cbr.py
ENet-CBR：上下文与边界精炼（Context & Boundary Refinement）完整方法。

核心创新（三模块协同）：
  1. L-ASPP：轻量级空洞空间金字塔池化，捕获多尺度上下文信息
     → 解决 ENet 缺乏多尺度感受野的根本问题
  2. CoordAtt（CA）：坐标注意力，增强水平/垂直方向感知
     → L-ASPP 提供的丰富上下文为 CA 提供更好的特征基础
  3. BGFR：边界引导特征精炼，边界注意力精炼主特征（推理时仍生效）
     → 不同于 DBA 的"仅训练时提供损失"，BGFR 的边界注意力在推理时仍然参与

架构：
  输入 (B, 3, H, W)
    ↓
  ENet Encoder (Stage1→Stage2→Stage3)
    ↓ (B, 128, H/8, W/8)
  ┌──────────────────────────────────────┐
  │  L-ASPP(128, rates=[1,3,6])         │  ← 多尺度上下文
  │  → (B, 128, H/8, W/8)              │
  │  ↓                                   │
  │  CoordAtt(128)                       │  ← 方向注意力
  │  → (B, 128, H/8, W/8)              │
  │  ↓                                   │
  │  BGFR(128)                           │  ← 边界引导精炼（推理时仍生效！）
  │  → refined (B, 128, H/8, W/8)      │
  └──────────────────────────────────────┘
    ↓
  ENet Decoder → 分类输出

  总损失: L = 0.4×CE + 0.6×Dice + λ_bdy×BCE(boundary)

参数增量：ENet 0.26M + L-ASPP ~69K + CA ~3.4K + BGFR ~8.3K ≈ 0.36M（实测 0.3618M）

消融矩阵：
  | 模型           | L-ASPP | CA  | BGFR | 目标              |
  | ENet           | ✗      | ✗   | ✗    | 基线              |
  | ENet+CA        | ✗      | ✓   | ✗    | CA 单独（已验证-0.36%）|
  | ENet+BA        | ✗      | ✗   | ✗*   | BA 仅loss（已验证0%） |
  | ENet-DBA       | ✗      | ✓   | ✗*   | CA+BA仅loss（+0.07%）|
  | ENet+L-ASPP    | ✓      | ✗   | ✗    | 多尺度单独效果      |
  | ENet-CBR(本文件)| ✓      | ✓   | ✓    | 完整方法           |

  * BA 仅提供训练损失，推理时丢弃；BGFR 推理时仍生效
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_enet import InitialBlock, Bottleneck
from components.laspp import LightweightASPP
from components.ca_attention import CoordAtt
from components.bgfr import BGFR


class ENetCBR(nn.Module):
    """
    ENet-CBR：上下文与边界精炼完整方法。

    参数:
        num_classes:    分类数，默认 2（背景 + 可通行区域）
        boundary_weight: 边界损失权重 λ_bdy，默认 0.5
    """
    def __init__(self, num_classes=2, boundary_weight=0.5):
        super().__init__()
        self.boundary_weight = boundary_weight

        # ---- 初始模块 ----
        self.initial = InitialBlock()
        # ---- 编码器（与 ENet 基线一致） ----
        self.stage1 = nn.Sequential(
            Bottleneck(16, 64, stride=2, dropout_prob=0.01),
            Bottleneck(64, 64, dropout_prob=0.01),
            Bottleneck(64, 64, dropout_prob=0.01),
            Bottleneck(64, 64, dropout_prob=0.01),
            Bottleneck(64, 64, dropout_prob=0.01),
        )
        self.stage2 = nn.Sequential(
            Bottleneck(64, 128, stride=2, dropout_prob=0.1),
            Bottleneck(128, 128, dilation=2, dropout_prob=0.1),
            Bottleneck(128, 128, asymmetric=True, dropout_prob=0.1),
            Bottleneck(128, 128, dilation=4, dropout_prob=0.1),
        )
        self.stage3 = nn.Sequential(
            Bottleneck(128, 128, dropout_prob=0.1),
            Bottleneck(128, 128, dilation=2, dropout_prob=0.1),
            Bottleneck(128, 128, asymmetric=True, dropout_prob=0.1),
            Bottleneck(128, 128, dilation=4, dropout_prob=0.1),
            Bottleneck(128, 128, dropout_prob=0.1),
        )

        # ★ CBR 三模块
        self.laspp = LightweightASPP(in_channels=128, rates=[1, 3, 6])  # 多尺度上下文
        self.ca = CoordAtt(channels=128, reduction=32)                   # 方向注意力
        self.bgfr = BGFR(in_channels=128, mid_channels=64, boundary_weight=boundary_weight)  # 边界精炼

        # ---- 解码器（与 ENet 基线一致） ----
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=False)
        self.bn_up4 = nn.BatchNorm2d(64)
        self.prelu_up4 = nn.PReLU(64)
        self.stage4 = nn.Sequential(
            Bottleneck(64, 64, dropout_prob=0.1),
            Bottleneck(64, 64, dropout_prob=0.1),
        )
        self.up5 = nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2, bias=False)
        self.bn_up5 = nn.BatchNorm2d(16)
        self.prelu_up5 = nn.PReLU(16)
        self.stage5 = nn.Sequential(
            Bottleneck(16, 16, dropout_prob=0.01),
        )
        self.classifier = nn.Conv2d(16, num_classes, kernel_size=1)

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
        # 编码器
        x = self.initial(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        # ★ CBR 三模块
        x = self.laspp(x)   # 1) 多尺度上下文
        x = self.ca(x)      # 2) 方向注意力
        # 3) 边界引导特征精炼
        if self.training and mask is not None:
            x, bdy_loss = self.bgfr(x, mask)  # (refined, bdy_loss)
        else:
            result = self.bgfr(x)
            x = result[0]
            bdy_loss = None

        # 解码器
        x = self.prelu_up4(self.bn_up4(self.up4(x)))
        x = self.stage4(x)
        x = self.prelu_up5(self.bn_up5(self.up5(x)))
        x = self.stage5(x)
        out = self.classifier(x)
        seg_out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        if self.training and mask is not None:
            return seg_out, bdy_loss
        else:
            return (seg_out,)


def get_enet_cbr_2class(device):
    """返回 ENet-CBR 二分类模型"""
    model = ENetCBR(num_classes=2)
    model = model.to(device)
    return model
