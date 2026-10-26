"""
model_enet_laspp.py
ENet + L-ASPP 消融模型。

在 ENet 编码器 Stage3 输出后插入轻量级 ASPP 模块，
验证多尺度上下文信息对可通行区域分割的单独效果。

架构：
  输入 (B, 3, H, W)
    ↓
  ENet Encoder (Stage1→Stage2→Stage3)
    ↓ (B, 128, H/8, W/8)
  ┌──────────────────────────────────────┐
  │  LightweightASPP(in_channels=128)    │  ← 多尺度上下文
  │  rates=[1,3,6] + global_pool         │
  │  → (B, 128, H/8, W/8)              │
  └──────────────────────────────────────┘
    ↓
  ENet Decoder (Stage4→Stage5) → 分类输出

参数增量：ENet 0.26M + L-ASPP ~69K ≈ 0.35M（实测 0.3500M）

消融定位：验证 L-ASPP 单独效果（无 CA，无 BGFR）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_enet import InitialBlock, Bottleneck
from components.laspp import LightweightASPP


class ENetLASSP(nn.Module):
    """ENet + Lightweight ASPP 消融模型。"""
    def __init__(self, num_classes=2):
        super().__init__()
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
        # ★ L-ASPP：多尺度上下文增强
        self.laspp = LightweightASPP(in_channels=128, rates=[1, 3, 6])
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

    def forward(self, x):
        _, _, H, W = x.shape
        x = self.initial(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.laspp(x)       # ★ 多尺度上下文增强
        x = self.prelu_up4(self.bn_up4(self.up4(x)))
        x = self.stage4(x)
        x = self.prelu_up5(self.bn_up5(self.up5(x)))
        x = self.stage5(x)
        out = self.classifier(x)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out


def get_enet_laspp_2class(device):
    """返回 ENet+L-ASPP 二分类模型"""
    model = ENetLASSP(num_classes=2)
    model = model.to(device)
    return model
