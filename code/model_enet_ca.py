"""
model_enet_ca.py
ENet + Coordinate Attention（坐标注意力）改进版。

在 ENet 编码器输出后（Stage3 → 128通道）插入 CoordAtt 模块，
增强对可通行区域条状特征的方向感知能力，不改变输出形状和通道数。

与 ENet 基线完全相同的超参数和训练条件，唯一差异是增加了 CA 模块。

对外接口：
  get_enet_ca_2class(device)  →  返回已移至 device 的 ENet+CA 模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_enet import ENet, InitialBlock, Bottleneck
from components.ca_attention import CoordAtt


class ENetCA(nn.Module):
    """
    ENet + Coordinate Attention 改进版。

    改进位置：编码器 Stage3 输出后插入 CoordAtt(channels=128, reduction=32)，
    使模型在编码器最高语义层捕获可通行区域的水平/垂直方向信息。

    参数量：ENet 基线 ~0.26M + CA ~0.01M ≈ 0.27M
    """
    def __init__(self, num_classes=2):
        super().__init__()
        # ---- 初始模块 ----
        self.initial = InitialBlock()  # (B, 3, H, W) → (B, 16, H/2, W/2)

        # ---- 编码器 ----
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

        # ★ CA 注意力：插入编码器最高语义层（128通道）
        self.ca = CoordAtt(channels=128, reduction=32)

        # ---- 解码器 ----
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

        # ---- 分类头 ----
        self.classifier = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        _, _, H, W = x.shape

        # 编码器
        x = self.initial(x)    # (B, 16, H/2, W/2)
        x = self.stage1(x)     # (B, 64, H/4, W/4)
        x = self.stage2(x)     # (B, 128, H/8, W/8)
        x = self.stage3(x)     # (B, 128, H/8, W/8)

        # ★ CA 注意力增强
        x = self.ca(x)         # (B, 128, H/8, W/8) → 形状不变

        # 解码器
        x = self.prelu_up4(self.bn_up4(self.up4(x)))    # (B, 64, H/4, W/4)
        x = self.stage4(x)
        x = self.prelu_up5(self.bn_up5(self.up5(x)))    # (B, 16, H/2, W/2)
        x = self.stage5(x)

        # 分类
        out = self.classifier(x)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out


def get_enet_ca_2class(device):
    """返回 ENet+CA 二分类模型"""
    model = ENetCA(num_classes=2)
    model = model.to(device)
    return model
