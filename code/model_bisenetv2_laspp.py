"""
model_bisenetv2_laspp.py
BiSeNetV2 + L-ASPP 消融模型。

在 BiSeNetV2 语义分支输出后插入轻量级 ASPP 模块，
验证多尺度上下文信息对精简双分支网络可通行区域分割的单独效果。

架构：
  输入 (B, 3, H, W)
    ├── Detail Branch → (B, 128, H/4, W/4)
    └── Semantic Branch → (B, 256, H/16, W/16)
            ↓
        L-ASPP(256, rates=[1,3,6])   ← 多尺度上下文
            ↓ (B, 256, H/16, W/16)
        Upsample → cat(detail) → Head → 分割输出

参数增量：精简双分支基线 0.94M + L-ASPP ~138K ≈ 1.08M（实测 1.2812M）

消融定位：验证 L-ASPP 单独效果（无 CA，无 BGFR）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from components.laspp import LightweightASPP


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class BiSeNetV2LASSP(nn.Module):
    """BiSeNetV2 + Lightweight ASPP 消融模型。"""
    def __init__(self, num_classes=2):
        super().__init__()
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
        # ★ L-ASPP：多尺度上下文增强
        self.laspp = LightweightASPP(in_channels=256, rates=[1, 3, 6])
        # ---- 融合头 + 分类输出（与精简双分支基线一致） ----
        self.head = nn.Sequential(
            ConvBNReLU(384, 128, 3, 1, 1),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )

    def forward(self, x):
        _, _, H, W = x.shape
        detail = self.detail_branch(x)
        semantic = self.semantic_branch(x)
        semantic = self.laspp(semantic)  # ★ 多尺度上下文增强
        semantic_up = F.interpolate(semantic, size=detail.shape[2:], mode='bilinear', align_corners=False)
        fuse = torch.cat([detail, semantic_up], dim=1)
        out = self.head(fuse)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out


def get_bisenetv2_laspp_2class(device):
    """返回 BiSeNetV2+L-ASPP 二分类模型"""
    model = BiSeNetV2LASSP(num_classes=2)
    model = model.to(device)
    return model
