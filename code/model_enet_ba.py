"""
model_enet_ba.py
ENet-BA：仅边界感知分支（Boundary-Aware），不含 CA 注意力。

消融实验模型：验证边界感知分支单独的效果。
与 ENet-DBA 对比，可以分离 CA 和 BA 各自的贡献。

消融实验矩阵：
  - ENet（基线）：无 CA，无 BA
  - ENet+CA（已验证）：有 CA，无 BA → mIoU -0.36%（CA 单独无效）
  - ENet+BA（本文件）：无 CA，有 BA → 验证 BA 单独效果
  - ENet-DBA（完整方法）：有 CA，有 BA → 验证 CA+BA 协同效果

对外接口：
  get_enet_ba_2class(device)  →  返回已移至 device 的 ENet+BA 模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_enet import InitialBlock, Bottleneck
from model_enet_dba import BoundaryHead, generate_boundary_gt


class ENetBA(nn.Module):
    """
    ENet-BA：ENet + 边界感知分支（无 CA）。

    在 Stage3 输出后直接加边界检测头，不经过 CoordAtt。
    用于消融实验，验证边界感知分支的独立贡献。

    参数量：ENet 0.26M + BA 8.3K ≈ 0.27M
    """
    def __init__(self, num_classes=2, boundary_weight=0.5):
        super().__init__()
        self.boundary_weight = boundary_weight

        self.initial = InitialBlock()
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

        # ★ 边界检测头（BA）：直接从 Stage3 输出预测边界（无 CA）
        self.boundary_head = BoundaryHead(in_channels=128, mid_channels=64)

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
        _, _, H, W = x.shape

        x = self.initial(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)     # (B, 128, H/8, W/8)

        # ★ BA 边界检测（无 CA）
        boundary_out = self.boundary_head(x)

        x = self.prelu_up4(self.bn_up4(self.up4(x)))
        x = self.stage4(x)
        x = self.prelu_up5(self.bn_up5(self.up5(x)))
        x = self.stage5(x)

        seg_out = self.classifier(x)
        seg_out = F.interpolate(seg_out, size=(H, W), mode='bilinear', align_corners=False)

        if self.training and mask is not None:
            boundary_gt_full = generate_boundary_gt(mask, kernel_size=3)
            boundary_gt = F.interpolate(
                boundary_gt_full, size=boundary_out.shape[2:],
                mode='bilinear', align_corners=False
            )
            boundary_gt = (boundary_gt > 0.5).float()
            bdy_loss = F.binary_cross_entropy(boundary_out, boundary_gt)
            return seg_out, boundary_out, boundary_gt, bdy_loss
        else:
            return (seg_out,)


def get_enet_ba_2class(device):
    """返回 ENet-BA 二分类模型"""
    model = ENetBA(num_classes=2)
    model = model.to(device)
    return model
