"""
components/bgfr.py
边界引导特征精炼模块（Boundary-Guided Feature Refinement, BGFR）。

核心思想：
  传统的边界检测头（BA）只在训练时提供边界损失信号，推理时边界头被丢弃，
  对最终分割输出零贡献——这导致边界信息在推理阶段完全浪费。

  BGFR 的关键改进：
  1. 边界预测 → sigmoid → 生成边界注意力图（0~1）
  2. 边界注意力图与主特征逐元素相乘，增强边界区域特征响应
  3. 残差连接保留原始特征，避免信息丢失
  4. 推理时边界注意力仍然生效，不是训练独占的辅助损失

  训练时：额外计算边界 BCE 损失，提供显式边界监督
  推理时：边界注意力图自动聚焦于边界区域，精炼特征

公式：
  boundary_attn = sigmoid(boundary_head(x))      # 边界注意力图
  refined = x * boundary_attn + x                 # 残差精炼
  L_bdy = BCE(boundary_attn, boundary_gt)         # 边界损失（训练时）

参数量（以 in_channels=128 为例）：
  - BoundaryHead: 128×64 + 64 + 64×1 + 1 ≈ 8.3K
  - 总计 ≈ 8.3K（极轻量）

使用方式：
    from components.bgfr import BGFR
    bgfr = BGFR(in_channels=128, boundary_weight=0.5)
    # 训练时：
    refined, bdy_loss = bgfr(feature, mask=gt_mask)
    # 推理时：
    refined = bgfr(feature)  # 返回元组 (refined,)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_enet_dba import BoundaryHead, generate_boundary_gt


class BGFR(nn.Module):
    """
    边界引导特征精炼模块。

    从特征中预测边界注意力图，用边界注意力精炼主特征，
    同时在训练时提供边界监督损失。

    参数:
        in_channels:    输入特征通道数
        mid_channels:   边界头中间通道数，默认 64
        boundary_weight: 边界损失权重 λ_bdy，默认 0.5
    """
    def __init__(self, in_channels, mid_channels=64, boundary_weight=0.5):
        super().__init__()
        self.boundary_weight = boundary_weight

        # 边界检测头：预测边界注意力图
        self.boundary_head = BoundaryHead(in_channels=in_channels, mid_channels=mid_channels)

    def forward(self, x, mask=None):
        """
        边界引导特征精炼。

        参数:
            x:    输入特征 (B, C, H, W)
            mask: 分割真值 (B, H, W)，训练时提供

        返回:
            训练模式: (refined, bdy_loss)
                - refined:  精炼后的特征 (B, C, H, W)
                - bdy_loss: 边界损失标量
            推理模式: (refined,)
                - refined:  精炼后的特征 (B, C, H, W)
        """
        # 1) 边界注意力图预测
        boundary_attn = self.boundary_head(x)  # (B, 1, H', W'), sigmoid 输出 [0, 1]

        # 2) 边界注意力精炼特征（残差连接）
        #    boundary_attn 在边界处接近 1，非边界处接近 0
        #    x * boundary_attn 增强边界特征，+ x 保留原始信息
        refined = x * boundary_attn + x  # (B, C, H', W')

        # 3) 边界损失（仅训练时）
        if self.training and mask is not None:
            boundary_gt_full = generate_boundary_gt(mask, kernel_size=3)  # (B, 1, H, W)
            # 将边界真值缩放到特征图尺寸
            boundary_gt = F.interpolate(
                boundary_gt_full, size=boundary_attn.shape[2:],
                mode='bilinear', align_corners=False
            )
            boundary_gt = (boundary_gt > 0.5).float()
            bdy_loss = F.binary_cross_entropy(boundary_attn, boundary_gt)
            return refined, bdy_loss
        else:
            return (refined,)
