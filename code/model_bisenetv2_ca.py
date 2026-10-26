"""
model_bisenetv2_ca.py
基于本文自实现的精简双分支网络（论文显示名 SBNet，非官方 BiSeNetV2），
在上下文分支（语义分支）输出后、两路融合前嵌入 Coordinate Attention
（CA 坐标注意力）模块的改进版本。

改动说明：
  - CA 模块已抽离为通用组件，从 components.ca_attention 导入
  - BiSeNetV2_CA 在语义分支输出后接一个 CA(channels=256)
  - CA 输出通道与空间尺寸均不变，融合头输入仍为 384 通道
  - 模型输入输出与基线 100% 兼容
  - state_dict 参数名完全不变，旧权重可直接加载

对外接口：
  get_bisenetv2_ca_2class(device)  →  返回已移至 device 的 BiSeNetV2_CA 模型
"""

import sys
import os
import torch
import torch.nn as nn

# 确保能从上级目录导入 components 包
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from components.ca_attention import CoordinateAttention


# ======================== 基础卷积单元（与精简双分支基线一致） ========================
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# ======================== 改进版 BiSeNetV2（嵌入 CA） ========================
class BiSeNetV2_CA(nn.Module):
    """
    与精简双分支基线的唯一区别：
      语义分支输出 (B, 256, H/16, W/16) 后，先经过 CoordinateAttention(256, 256)
      再上采样与细节分支融合。CA 不改变通道数和空间尺寸，融合头输入仍为 384。
    """

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

        # ★ 坐标注意力模块（从 components.ca_attention 导入，参数名与旧版完全一致）
        self.ca = CoordinateAttention(inp=256, oup=256, reduction=32)

        # ---- 融合头 + 分类输出（与精简双分支基线一致） ----
        self.head = nn.Sequential(
            ConvBNReLU(384, 128, 3, 1, 1),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )

    def forward(self, x):
        detail = self.detail_branch(x)          # (B, 128, H/4,  W/4)
        semantic = self.semantic_branch(x)       # (B, 256, H/16, W/16)
        semantic = self.ca(semantic)             # ★ CA 注意力加权，通道/尺寸不变
        semantic = nn.functional.interpolate(
            semantic, size=detail.shape[2:], mode='bilinear', align_corners=False
        )                                        # (B, 256, H/4,  W/4)
        fuse = torch.cat([detail, semantic], dim=1)  # (B, 384, H/4, W/4)
        out = self.head(fuse)                    # (B, 2,   H/4, W/4)
        out = nn.functional.interpolate(
            out, size=x.shape[2:], mode='bilinear', align_corners=False
        )                                        # (B, 2,   H,   W)
        return out


# ======================== 对外调用函数 ========================
def get_bisenetv2_ca_2class(device):
    """
    返回嵌入 Coordinate Attention 的 BiSeNetV2 二分类模型。
    入参与返回值和基线 get_bisenetv2_2class(device) 完全一致。
    """
    model = BiSeNetV2_CA(num_classes=2)
    model = model.to(device)
    return model
