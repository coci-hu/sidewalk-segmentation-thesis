"""
model_enet.py
ENet（Efficient Neural Network）极轻量语义分割网络（二分类可通行区域/背景）。

论文：ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation (2016)
特点：实测约 0.26M 参数（官方 ENet 为 0.36M，本实现为精简版），编码器-解码器结构，专为实时低功耗场景设计，
      是轻量化语义分割领域的经典基线模型。

架构总览：
  输入 (B, 3, H, W)
    ↓
  ┌─────────────────────────────────────┐
  │  Initial Block                      │
  │  Conv 3→13, s=2 + MaxPool 3→3      │
  │  → 拼接 → (B, 16, H/2, W/2)       │
  └─────────────────────────────────────┘
    ↓
  ┌─────────────────────────────────────┐
  │  Encoder                            │
  │  Stage1: 5× Bottleneck(16→64)      │
  │    → 下采样1次 (H/4, W/4)          │
  │  Stage2: 3× Bottleneck(64→128) +   │
  │           空洞卷积扩展感受野         │
  │    → 下采样1次 (H/8, W/8)          │
  │  Stage3: 5× Bottleneck(128→128) +  │
  │           空洞卷积扩展感受野         │
  │    → 无下采样 (H/8, W/8)           │
  └─────────────────────────────────────┘
    ↓
  ┌─────────────────────────────────────┐
  │  Decoder                            │
  │  Stage4: 2× Bottleneck(128→64)     │
  │    → 上采样 (H/4, W/4)             │
  │  Stage5: 3× Bottleneck(64→16)      │
  │    → 上采样 (H/2, W/2)             │
  │  Conv 16→2 (分类头)                 │
  │  → 双线性插值回原始分辨率           │
  └─────────────────────────────────────┘
    → (B, 2, H, W)

对外接口：
  get_enet_2class(device)  →  返回已移至 device 的 ENet 模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================== 初始模块 ========================
class InitialBlock(nn.Module):
    """
    ENet 初始模块：并行执行卷积和最大池化，结果拼接。
    Conv(3→13, s=2) + MaxPool(s=2) → concat → (B, 16, H/2, W/2)

    设计思想：用极少的参数完成初步下采样和特征提取。
    """
    def __init__(self):
        super().__init__()
        # 卷积分支：3→13 通道，步幅2下采样
        self.conv = nn.Sequential(
            nn.Conv2d(3, 13, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(13),
            nn.PReLU(13),
        )
        # 池化分支：3→3 通道（保持通道不变），步幅2下采样
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        conv_out = self.conv(x)       # (B, 13, H/2, W/2)
        pool_out = self.pool(x)       # (B, 3,  H/2, W/2)
        return torch.cat([conv_out, pool_out], dim=1)  # (B, 16, H/2, W/2)


# ======================== 瓶颈模块 ========================
class Bottleneck(nn.Module):
    """
    ENet 核心瓶颈模块（严格遵循原论文设计）。

    下采样 Bottleneck（stride=2）：
      主路径：Conv2d(in→mid, 3×3, stride=2) → BN → PReLU → Conv2d(mid→out, 1×1) → BN
      支路：  MaxPool(stride=2) + 通道填充对齐
    非下采样 Bottleneck（stride=1）：
      主路径：Conv1×1(in→mid) → BN → PReLU → 主卷积(mid→mid) → BN → PReLU → Conv1×1(mid→out) → BN
      支路：  恒等映射（仅 in==out 时）

    参数：
        in_channels:   输入通道数
        out_channels:  输出通道数
        stride:        步幅（1 或 2），stride=2 时执行下采样
        dilation:      空洞率，>1 时使用空洞卷积
        asymmetric:    是否使用非对称卷积（5×1 + 1×5 代替 5×5）
        dropout_prob:  空间 Dropout 概率
    """
    def __init__(self, in_channels, out_channels, stride=1, dilation=1,
                 asymmetric=False, dropout_prob=0.01):
        super().__init__()
        self.is_downsample = (stride == 2)
        # 残差连接条件：不下采样 且 通道一致
        self.use_res_connect = (stride == 1 and in_channels == out_channels)

        # 瓶颈中间通道数：in_channels // 4（下采样时中间通道=in_channels）
        if self.is_downsample:
            mid_channels = in_channels
        else:
            mid_channels = max(in_channels // 4, 16)

        # ---- 下采样 Bottleneck：简化结构 ----
        if self.is_downsample:
            # 主卷积直接做下采样
            self.conv_main = nn.Conv2d(in_channels, mid_channels, 3, stride=2, padding=1, bias=False)
            self.bn_main = nn.BatchNorm2d(mid_channels)
            self.prelu_main = nn.PReLU(mid_channels)
            self.conv_expand = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
            self.bn_expand = nn.BatchNorm2d(out_channels)
        else:
            # ---- 非下采样 Bottleneck：标准瓶颈结构 ----
            self.conv_compress = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
            self.bn_compress = nn.BatchNorm2d(mid_channels)
            self.prelu_compress = nn.PReLU(mid_channels)

            # 主卷积类型选择
            if asymmetric:
                # 非对称卷积：5×1 + 1×5（分解 5×5 卷积，减少参数）
                # 注意：asymmetric 模式下 dilation 参数不生效
                self.conv_main = nn.Sequential(
                    nn.Conv2d(mid_channels, mid_channels, (5, 1), padding=(2, 0), bias=False),
                    nn.BatchNorm2d(mid_channels),
                    nn.PReLU(mid_channels),
                    nn.Conv2d(mid_channels, mid_channels, (1, 5), padding=(0, 2), bias=False),
                )
            elif dilation > 1:
                self.conv_main = nn.Sequential(
                    nn.Conv2d(mid_channels, mid_channels, 3, padding=dilation,
                              dilation=dilation, bias=False),
                )
            else:
                self.conv_main = nn.Sequential(
                    nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
                )
            self.bn_main = nn.BatchNorm2d(mid_channels)
            self.prelu_main = nn.PReLU(mid_channels)

            self.conv_expand = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
            self.bn_expand = nn.BatchNorm2d(out_channels)

        # 空间 Dropout
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

        # ---- 下采样支路（MaxPool + 0填充对齐通道） ----
        if self.is_downsample:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pad_channels = (out_channels - in_channels) if out_channels > in_channels else 0

        # PReLU 在残差连接之后
        self.prelu_out = nn.PReLU(out_channels)

    def forward(self, x):
        identity = x

        # 主路径
        if self.is_downsample:
            out = self.prelu_main(self.bn_main(self.conv_main(x)))
            out = self.bn_expand(self.conv_expand(out))
        else:
            out = self.prelu_compress(self.bn_compress(self.conv_compress(x)))
            out = self.prelu_main(self.bn_main(self.conv_main(out)))
            out = self.bn_expand(self.conv_expand(out))

        out = self.dropout(out)

        # 下采样支路
        if self.is_downsample:
            identity = self.pool(identity)
            if self.pad_channels > 0:
                identity = F.pad(identity, (0, 0, 0, 0, 0, self.pad_channels))

        # 残差连接
        if self.use_res_connect or self.is_downsample:
            out = out + identity

        return self.prelu_out(out)


# ======================== ENet 完整网络 ========================
class ENet(nn.Module):
    """
    ENet 完整网络。

    编码器（5 个阶段）+ 解码器（2 个阶段）。

    参数：
        num_classes: 分类数，默认 2（背景 + 可通行区域）
    """
    def __init__(self, num_classes=2):
        super().__init__()
        # ---- 初始模块 ----
        self.initial = InitialBlock()  # (B, 3, H, W) → (B, 16, H/2, W/2)

        # ---- 编码器 ----
        # Stage 1: 下采样 16→64，1 个下采样 + 4 个常规
        self.stage1 = nn.Sequential(
            Bottleneck(16, 64, stride=2, dropout_prob=0.01),   # 下采样 → H/4
            Bottleneck(64, 64, dropout_prob=0.01),
            Bottleneck(64, 64, dropout_prob=0.01),
            Bottleneck(64, 64, dropout_prob=0.01),
            Bottleneck(64, 64, dropout_prob=0.01),
        )
        # Stage 2: 下采样 64→128，1 个下采样 + 2 个空洞卷积
        self.stage2 = nn.Sequential(
            Bottleneck(64, 128, stride=2, dropout_prob=0.1),    # 下采样 → H/8
            Bottleneck(128, 128, dilation=2, dropout_prob=0.1),
            Bottleneck(128, 128, asymmetric=True, dropout_prob=0.1),
            Bottleneck(128, 128, dilation=4, dropout_prob=0.1),
        )
        # Stage 3: 不下采样 128→128，5 个空洞/非对称卷积
        self.stage3 = nn.Sequential(
            Bottleneck(128, 128, dropout_prob=0.1),
            Bottleneck(128, 128, dilation=2, dropout_prob=0.1),
            Bottleneck(128, 128, asymmetric=True, dropout_prob=0.1),
            Bottleneck(128, 128, dilation=4, dropout_prob=0.1),
            Bottleneck(128, 128, dropout_prob=0.1),
        )

        # ---- 解码器 ----
        # Stage 4: 先上采样 128→64，再 Bottleneck 处理
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=False)
        self.bn_up4 = nn.BatchNorm2d(64)
        self.prelu_up4 = nn.PReLU(64)
        self.stage4 = nn.Sequential(
            Bottleneck(64, 64, dropout_prob=0.1),
            Bottleneck(64, 64, dropout_prob=0.1),
        )

        # Stage 5: 先上采样 64→16，再 Bottleneck 处理
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

        # 解码器：先上采样，再 Bottleneck
        x = self.prelu_up4(self.bn_up4(self.up4(x)))    # (B, 64, H/4, W/4)
        x = self.stage4(x)                                # (B, 64, H/4, W/4)
        x = self.prelu_up5(self.bn_up5(self.up5(x)))    # (B, 16, H/2, W/2)
        x = self.stage5(x)                                # (B, 16, H/2, W/2)

        # 分类
        out = self.classifier(x)  # (B, 2, H/2, W/2)
        # 双线性插值回原始分辨率
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out


# ======================== 对外调用函数 ========================
def get_enet_2class(device):
    """返回 ENet 二分类模型，入参与返回值和其余模型一致。"""
    model = ENet(num_classes=2)
    model = model.to(device)
    return model
