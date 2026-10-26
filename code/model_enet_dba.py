"""
model_enet_dba.py
ENet-DBA: 方向边界协同优化（Directional Boundary-Aware）改进版。

核心创新：
  在 ENet 编码器最高语义层同时引入坐标注意力（CA）和边界感知分支（BA），
  两者形成正反馈协同：
    CA 提供水平/垂直方向先验 → 边界头从方向增强特征中学习边界 →
    边界损失反传强化 CA 的方向注意力 → 分割精度提升。

  ENet+CA 单独使用时 mIoU 下降（-0.36%），因为 CA 缺少显式监督信号
  来"激活"方向感知能力；而加入 BA 后，边界监督为 CA 提供了方向信息
  的学习目标，使 CA 的方向注意力被有效激活。

架构：
  输入 (B, 3, H, W)
    ↓
  ENet Encoder (Stage1→Stage2→Stage3)
    ↓ (B, 128, H/8, W/8)
  ┌──────────────────────────────────────┐
  │  CoordAtt(channels=128)             │  ← 方向注意力
  │  ↓ (B, 128, H/8, W/8)              │
  │  ┬→ 解码器 → 分割输出 (L_seg)       │
  │  └→ 边界检测头 → 边界输出 (L_bdy)   │  ← 边界感知
  └──────────────────────────────────────┘

  总损失: L = 0.4×CE + 0.6×Dice + λ_bdy×BCE(boundary)

参数增量：ENet 0.26M + CA 3.35K + BA 8.3K ≈ 0.27M（极轻量）

消融模型：
  - ENet-BA（model_enet_ba.py）：仅边界感知分支，无CA，验证BA单独效果
  - ENet-CA（model_enet_ca.py）：仅CA，无BA，已验证CA单独无效
  - ENet-DBA（本文件）：CA+BA协同，本文提出的完整方法

对外接口：
  get_enet_dba_2class(device)  →  返回已移至 device 的 ENet-DBA 模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_enet import InitialBlock, Bottleneck
from components.ca_attention import CoordAtt


# ======================== 边界检测头 ========================
class BoundaryHead(nn.Module):
    """
    轻量边界检测头。

    从编码器最高语义层预测像素级边界图（单通道，Sigmoid 输出）。
    训练时提供边界监督信号，推理时可选择保留或丢弃。

    结构：Conv1×1(128→64) → BN → ReLU → Conv1×1(64→1)
    参数量：128×64 + 64 + 64×1 + 1 = 8,257 ≈ 8.3K
    """
    def __init__(self, in_channels=128, mid_channels=64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
        )

    def forward(self, x):
        """
        参数:
            x: 编码器特征 (B, 128, H/8, W/8)
        返回:
            boundary: 边界预测 (B, 1, H/8, W/8)，Sigmoid 激活
        """
        return torch.sigmoid(self.head(x))


# ======================== 边界真值生成 ========================
def generate_boundary_gt(mask, kernel_size=3):
    """
    从分割掩码自动生成边界真值（形态学腐蚀法）。

    原理：boundary = mask - erode(mask)
    - mask 中值为 1（可通行区域）的区域，经腐蚀后边界向内收缩
    - 差值即为边界像素（1像素宽）

    参数:
        mask:        分割掩码 (B, H, W)，值为 0/1
        kernel_size: 腐蚀核大小，默认 3（1像素边界宽度）
    返回:
        boundary_gt: 边界真值 (B, 1, H, W)，值为 0/1
    """
    # 确保 mask 是 float 类型且形状为 (B, 1, H, W)
    if mask.dim() == 3:
        mask_float = mask.unsqueeze(1).float()  # (B, 1, H, W)
    else:
        mask_float = mask.float()

    # 构造腐蚀核
    padding = kernel_size // 2
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)

    # 腐蚀 = 最小值池化（用 -max_pool(-x) 近似）
    eroded = -F.max_pool2d(
        -mask_float,
        kernel_size=kernel_size,
        stride=1,
        padding=padding
    )
    # 边界 = 原始掩码 - 腐蚀掩码
    boundary = mask_float - eroded
    # 二值化（消除浮点误差）
    boundary = (boundary > 0.5).float()
    return boundary


# ======================== ENet-DBA 完整网络 ========================
class ENetDBA(nn.Module):
    """
    ENet-DBA：方向边界协同优化 ENet。

    在 ENet 编码器 Stage3 输出后：
    1. CoordAtt 增强水平/垂直方向特征
    2. 边界检测头从方向增强特征中预测边界
    3. 训练时联合优化分割损失 + 边界损失

    参数：
        num_classes: 分类数，默认 2（背景 + 可通行区域）
        boundary_weight: 边界损失权重 λ_bdy，默认 0.5
    """
    def __init__(self, num_classes=2, boundary_weight=0.5):
        super().__init__()
        self.boundary_weight = boundary_weight

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

        # ★ 方向注意力（CA）：增强编码器最高语义层的方向感知
        self.ca = CoordAtt(channels=128, reduction=32)

        # ★ 边界检测头（BA）：从方向增强特征中预测边界
        self.boundary_head = BoundaryHead(in_channels=128, mid_channels=64)

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

    def forward(self, x, mask=None):
        """
        前向传播。

        参数:
            x:    输入图像 (B, 3, H, W)
            mask: 分割真值 (B, H, W)，值为 0/1。
                  训练时提供，用于生成边界真值计算边界损失；
                  推理时为 None，不计算边界损失。

        返回:
            训练模式: (seg_out, boundary_out, boundary_gt, bdy_loss)
                - seg_out:      分割预测 (B, 2, H, W)
                - boundary_out: 边界预测 (B, 1, H/8, W/8)
                - boundary_gt:  边界真值 (B, 1, H/8, W/8)
                - bdy_loss:     边界损失标量
            推理模式: (seg_out,)
                - seg_out:      分割预测 (B, 2, H, W)
        """
        _, _, H, W = x.shape

        # 编码器
        x = self.initial(x)    # (B, 16, H/2, W/2)
        x = self.stage1(x)     # (B, 64, H/4, W/4)
        x = self.stage2(x)     # (B, 128, H/8, W/8)
        x = self.stage3(x)     # (B, 128, H/8, W/8)

        # ★ CA 方向注意力增强
        x = self.ca(x)         # (B, 128, H/8, W/8) → 形状不变

        # ★ BA 边界检测
        boundary_out = self.boundary_head(x)  # (B, 1, H/8, W/8)

        # 解码器
        x = self.prelu_up4(self.bn_up4(self.up4(x)))    # (B, 64, H/4, W/4)
        x = self.stage4(x)
        x = self.prelu_up5(self.bn_up5(self.up5(x)))    # (B, 16, H/2, W/2)
        x = self.stage5(x)

        # 分类
        seg_out = self.classifier(x)  # (B, 2, H/2, W/2)
        seg_out = F.interpolate(seg_out, size=(H, W), mode='bilinear', align_corners=False)

        # 边界损失计算（仅训练时）
        if self.training and mask is not None:
            # 生成边界真值（在原始分辨率上）
            boundary_gt_full = generate_boundary_gt(mask, kernel_size=3)  # (B, 1, H, W)
            # 将边界真值缩放到边界头输出尺寸 (H/8, W/8)
            boundary_gt = F.interpolate(
                boundary_gt_full, size=boundary_out.shape[2:],
                mode='bilinear', align_corners=False
            )
            # 二值化（插值后可能产生中间值）
            boundary_gt = (boundary_gt > 0.5).float()
            # BCE 损失
            bdy_loss = F.binary_cross_entropy(boundary_out, boundary_gt)
            return seg_out, boundary_out, boundary_gt, bdy_loss
        else:
            return (seg_out,)


def get_enet_dba_2class(device):
    """返回 ENet-DBA 二分类模型"""
    model = ENetDBA(num_classes=2)
    model = model.to(device)
    return model
