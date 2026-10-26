"""
components/laspp.py
轻量级空洞空间金字塔池化模块（Lightweight ASPP）。

论文参考：
  - DeepLabv3+ 中的 ASPP 使用 3×3 Conv + 不同 dilation rate 捕获多尺度上下文
  - Fast-SCNN / ContextNet 中的轻量化变体：使用 Depthwise Separable Conv 大幅减少参数

本模块采用 Depthwise Separable 空洞卷积（rates=[1,3,6]）+ 1×1 卷积融合，
相比标准 ASPP 参数量降低 90%+，适合嵌入轻量级网络。

参数量（以 in_channels=128 为例）：
  - DW Conv rate=1:  128×3×3 = 1,152
  - DW Conv rate=3:  128×3×3 = 1,152
  - DW Conv rate=6:  128×3×3 = 1,152
  - 1×1 Conv fusion:  128×4×128 = 65,536  (4个分支拼接后压缩)
  - BN:              128×2 = 256
  合计 ≈ 69K（标准 ASPP 在同样通道下 ≈ 700K+）

使用方式：
    from components.laspp import LightweightASPP
    laspp = LightweightASPP(in_channels=128)
    out = laspp(feature_map)  # 输入输出形状完全一致
"""

import torch
import torch.nn as nn


class LightweightASPP(nn.Module):
    """
    轻量级 ASPP 模块。

    采用 Depthwise Separable 空洞卷积 + 全局平均池化分支，
    捕获多尺度上下文信息，适合嵌入轻量级实时分割网络。

    参数:
        in_channels:  输入/输出通道数（本模块不改变通道数）
        rates:        空洞率列表，默认 [1, 3, 6]
    """
    def __init__(self, in_channels, rates=[1, 3, 6]):
        super().__init__()
        self.in_channels = in_channels

        # ---- 多尺度深度可分离空洞卷积分支 ----
        branches = []
        for rate in rates:
            branches.append(nn.Sequential(
                # Depthwise Conv：3×3 空洞卷积，每组通道独立卷积
                nn.Conv2d(in_channels, in_channels, kernel_size=3,
                          padding=rate, dilation=rate, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
            ))
        self.branches = nn.ModuleList(branches)

        # ---- 全局平均池化分支（捕获全局上下文） ----
        # 注意：全局池化后尺寸为 (B,C,1,1)，BatchNorm 在 batch_size=1 时会报错，
        # 因此全局池化分支不使用 BN，仅用 1×1 Conv + ReLU
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                          # (B, C, 1, 1)
            nn.Conv2d(in_channels, in_channels, 1, bias=False),  # 1×1 升维
            nn.ReLU(inplace=True),
        )

        # ---- 1×1 卷积融合（(len(rates)+1) 个分支拼接后压缩回 in_channels） ----
        num_branches = len(rates) + 1  # 空洞卷积分支 + 全局池化分支
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * num_branches, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        参数:
            x: 输入特征 (B, C, H, W)
        返回:
            out: 多尺度融合特征 (B, C, H, W)，形状与输入一致
        """
        h, w = x.shape[2:]

        # 1) 多尺度空洞卷积分支
        branch_outs = [branch(x) for branch in self.branches]

        # 2) 全局平均池化分支 → 上采样回原尺寸
        global_out = self.global_pool(x)
        global_out = nn.functional.interpolate(
            global_out, size=(h, w), mode='bilinear', align_corners=False
        )
        branch_outs.append(global_out)

        # 3) 拼接所有分支 → 1×1 融合
        concat = torch.cat(branch_outs, dim=1)  # (B, C*(len(rates)+1), H, W)
        out = self.fusion(concat)                 # (B, C, H, W)
        return out
