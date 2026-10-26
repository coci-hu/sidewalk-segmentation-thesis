"""
model_unet.py
标准 U-Net 二分类语义分割网络（无预训练权重，随机初始化）。

架构：5 级编码器-解码器 + 跳跃连接
  编码器通道：64 → 128 → 256 → 512 → 1024
  解码器通道：512 → 256 → 128 → 64
  输入：(B, 3, H, W)   输出：(B, 2, H, W)

对外接口：
  get_unet_2class(device)  →  返回已移至 device 的 U-Net 模型
"""

import torch
import torch.nn as nn


# ======================== 基础卷积块 ========================
class DoubleConv(nn.Module):
    """U-Net 经典双卷积块：Conv-BN-ReLU × 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ======================== U-Net 网络 ========================
class UNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # ---- 编码器（下采样） ----
        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2, 2)

        # ---- 瓶颈层 ----
        self.bottleneck = DoubleConv(512, 1024)

        # ---- 解码器（上采样 + 跳跃连接） ----
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # ---- 分类头 ----
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # 编码
        e1 = self.enc1(x)           # (B,  64, H,   W)
        e2 = self.enc2(self.pool(e1))  # (B, 128, H/2, W/2)
        e3 = self.enc3(self.pool(e2))  # (B, 256, H/4, W/4)
        e4 = self.enc4(self.pool(e3))  # (B, 512, H/8, W/8)

        # 瓶颈
        b = self.bottleneck(self.pool(e4))  # (B, 1024, H/16, W/16)

        # 解码 + 跳跃连接
        d4 = self.up4(b)                     # (B, 512, H/8, W/8)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))  # (B, 512, H/8, W/8)

        d3 = self.up3(d4)                    # (B, 256, H/4, W/4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))  # (B, 256, H/4, W/4)

        d2 = self.up2(d3)                    # (B, 128, H/2, W/2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # (B, 128, H/2, W/2)

        d1 = self.up1(d2)                    # (B,  64, H,   W)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # (B,  64, H,   W)

        out = self.head(d1)                  # (B,   2, H,   W)
        return out


# ======================== 对外调用函数 ========================
def get_unet_2class(device):
    """返回 U-Net 二分类模型，入参与返回值和基线 get_bisenetv2_2class 一致。"""
    model = UNet(num_classes=2)
    model = model.to(device)
    return model
