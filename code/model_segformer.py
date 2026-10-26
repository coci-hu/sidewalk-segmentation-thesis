"""
model_segformer.py
SegFormer-B0 轻量 Transformer 语义分割网络（二分类可通行区域/背景）。

论文：SegFormer: Simple and Efficient Design for Semantic Segmentation
      with Transformers (NeurIPS 2021)
特点：实测约 4.30M 参数，层次化 Transformer 编码器 + 轻量全 MLP 解码器，
      无位置编码（对输入分辨率更鲁棒），是轻量化 Transformer 分割的代表模型。

架构总览：
  输入 (B, 3, H, W)
    ↓
  ┌─────────────────────────────────────┐
  │  Mix Transformer (MiT-B0) 编码器    │
  │  Stage1: OverlapPatchEmbed 7×7,s=4  │
  │    → (B, 32, H/4, W/4)             │
  │  Stage2: OverlapPatchEmbed 3×3,s=2  │
  │    → (B, 64, H/8, W/8)             │
  │  Stage3: OverlapPatchEmbed 3×3,s=2  │
  │    → (B, 160, H/16, W/16)          │
  │  Stage4: OverlapPatchEmbed 3×3,s=2  │
  │    → (B, 256, H/32, W/32)          │
  └─────────────────────────────────────┘
    ↓ 四个尺度特征图
  ┌─────────────────────────────────────┐
  │  MLP 解码头                         │
  │  每个尺度: MLP 降维 → 上采样到 H/4  │
  │  拼接 → Conv 融合 → 分类           │
  └─────────────────────────────────────┘
    → (B, 2, H, W)

对外接口：
  get_segformer_b0_2class(device)  →  返回已移至 device 的 SegFormer-B0 模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================重叠 Patch 嵌入========================
class OverlapPatchEmbed(nn.Module):
    """
    重叠 Patch 嵌入：用卷积实现重叠的 patch 划分。
    不同于 ViT 的非重叠 patch，SegFormer 使用步幅 < 核大小的卷积，
    保留更多空间信息。

    参数：
        img_size:     输入尺寸（仅用于信息提示，实际不限制）
        patch_size:   卷积核大小
        stride:       步幅
        in_channels:  输入通道数
        embed_dim:    嵌入维度
    """
    def __init__(self, patch_size=7, stride=4, in_channels=3, embed_dim=32):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=stride,
            padding=(patch_size // 2, patch_size // 2)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # (B, C, H, W) → (B, embed_dim, H', W') → (B, H'*W', embed_dim)
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        x = self.norm(x)
        return x, H, W


# ========================高效自注意力========================
class EfficientSelfAttention(nn.Module):
    """
    高效自注意力：通过空间缩减比（SR ratio）降低 key/value 的空间分辨率，
    大幅减少注意力计算量。

    参数：
        dim:       输入维度
        num_heads: 注意力头数
        sr_ratio:  空间缩减比（1=不缩减，4=key/value 缩至 1/4 分辨率）
    """
    def __init__(self, dim, num_heads=1, sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            # 空间缩减：用卷积降采样 key/value
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        # Query
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Key/Value（可选空间缩减）
        if self.sr_ratio > 1:
            x_2d = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_sr = self.sr(x_2d)                                  # (B, C, H', W')
            x_sr = x_sr.reshape(B, C, -1).permute(0, 2, 1)       # (B, N', C)
            x_sr = self.norm(x_sr)                                 # LayerNorm on last dim
            kv = self.kv(x_sr).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out


# ========================Mix-FFN（混合前馈网络）========================
class MixFFN(nn.Module):
    """
    Mix-FFN：在 MLP 中加入 3×3 深度卷积，隐式编码位置信息。
    SegFormer 不使用显式位置编码，而是通过 Mix-FFN 让模型感知位置。

    参数：
        dim:         输入维度
        expand_ratio: 扩展倍率
    """
    def __init__(self, dim, expand_ratio=4):
        super().__init__()
        hidden_dim = int(dim * expand_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1,
                                 groups=hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x, H, W):
        x = self.fc1(x)
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        x = x.reshape(B, C, -1).transpose(1, 2)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


# ========================Transformer 块========================
class TransformerBlock(nn.Module):
    """
    SegFormer 的 Transformer 块：高效自注意力 + Mix-FFN。

    参数：
        dim:       输入维度
        num_heads: 注意力头数
        sr_ratio:  空间缩减比
        expand_ratio: FFN 扩展倍率
    """
    def __init__(self, dim, num_heads=1, sr_ratio=1, expand_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim, expand_ratio)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ffn(self.norm2(x), H, W)
        return x


# ========================MiT 编码器阶段========================
class MiTStage(nn.Module):
    """
    MiT 编码器的一个阶段：重叠 Patch 嵌入 + N 个 Transformer 块。

    参数：
        patch_size:    嵌入卷积核大小
        stride:        嵌入步幅
        in_channels:   输入通道数
        embed_dim:     嵌入维度
        depth:         Transformer 块的个数
        num_heads:     注意力头数
        sr_ratio:      空间缩减比
    """
    def __init__(self, patch_size, stride, in_channels, embed_dim,
                 depth=2, num_heads=1, sr_ratio=1):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(
            patch_size=patch_size, stride=stride,
            in_channels=in_channels, embed_dim=embed_dim
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, sr_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x, H, W = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x, H, W)
        x = self.norm(x)
        # 恢复为 (B, C, H, W) 格式
        x = x.transpose(1, 2).reshape(x.shape[0], -1, H, W)
        return x


# ========================MiT-B0 编码器========================
class MiT_B0(nn.Module):
    """
    MiT-B0 编码器：SegFormer-B0 的骨干网络。
    4 个阶段，通道数 [32, 64, 160, 256]，深度 [2, 2, 2, 2]。

    输入：(B, 3, H, W)
    输出：4 个尺度的特征图列表
    """
    def __init__(self):
        super().__init__()
        # Stage1: 3 → 32, H/4
        self.stage1 = MiTStage(
            patch_size=7, stride=4, in_channels=3, embed_dim=32,
            depth=2, num_heads=1, sr_ratio=8
        )
        # Stage2: 32 → 64, H/8
        self.stage2 = MiTStage(
            patch_size=3, stride=2, in_channels=32, embed_dim=64,
            depth=2, num_heads=2, sr_ratio=4
        )
        # Stage3: 64 → 160, H/16
        self.stage3 = MiTStage(
            patch_size=3, stride=2, in_channels=64, embed_dim=160,
            depth=2, num_heads=5, sr_ratio=2
        )
        # Stage4: 160 → 256, H/32
        self.stage4 = MiTStage(
            patch_size=3, stride=2, in_channels=160, embed_dim=256,
            depth=2, num_heads=8, sr_ratio=1
        )

    def forward(self, x):
        c1 = self.stage1(x)   # (B, 32,  H/4,  W/4)
        c2 = self.stage2(c1)  # (B, 64,  H/8,  W/8)
        c3 = self.stage3(c2)  # (B, 160, H/16, W/16)
        c4 = self.stage4(c3)  # (B, 256, H/32, W/32)
        return [c1, c2, c3, c4]


# ========================MLP 解码头========================
class SegFormerHead(nn.Module):
    """
    SegFormer 轻量 MLP 解码头。
    每个尺度特征先经 MLP 降维到统一嵌入维度，
    再上采样到 H/4 拼接融合，最后分类。

    参数：
        in_channels:  各阶段输出通道数列表 [32, 64, 160, 256]
        embed_dim:    统一嵌入维度（MLP 降维后的通道数）
        num_classes:  分类数
    """
    def __init__(self, in_channels=[32, 64, 160, 256], embed_dim=256, num_classes=2):
        super().__init__()
        # 每个阶段的 MLP 降维 + 上采样
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_ch, embed_dim),
                nn.GELU(),
            )
            for in_ch in in_channels
        ])
        # 融合后的分类头
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, features):
        """
        features: [c1, c2, c3, c4] 四个尺度的特征图
        """
        target_h, target_w = features[0].shape[2], features[0].shape[3]
        outs = []
        for i, feat in enumerate(features):
            B, C, H, W = feat.shape
            # (B, C, H, W) → (B, H*W, C) → MLP → (B, H*W, embed_dim) → (B, embed_dim, H, W)
            x = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
            x = self.mlps[i](x)                    # (B, H*W, embed_dim)
            x = x.transpose(1, 2).reshape(B, -1, H, W)  # (B, embed_dim, H, W)
            # 上采样到统一的 H/4 分辨率
            if H != target_h or W != target_w:
                x = F.interpolate(x, size=(target_h, target_w),
                                  mode='bilinear', align_corners=False)
            outs.append(x)

        # 拼接融合
        out = torch.cat(outs, dim=1)   # (B, embed_dim*4, H/4, W/4)
        out = self.fuse_conv(out)       # (B, embed_dim, H/4, W/4)
        out = self.classifier(out)      # (B, num_classes, H/4, W/4)
        return out


# ========================SegFormer-B0 完整网络========================
class SegFormerB0(nn.Module):
    """
    SegFormer-B0 完整网络。

    流程：
      1. MiT-B0 编码器：提取 4 个尺度特征 [H/4, H/8, H/16, H/32]
      2. MLP 解码头：统一降维 → 上采样对齐 → 拼接融合 → 分类
      3. 双线性插值上采样到原始分辨率

    参数：
        num_classes: 分类数，默认 2（背景 + 可通行区域）
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.encoder = MiT_B0()
        self.decode_head = SegFormerHead(
            in_channels=[32, 64, 160, 256],
            embed_dim=256,
            num_classes=num_classes
        )

    def forward(self, x):
        _, _, H, W = x.shape
        features = self.encoder(x)         # 4 个尺度特征
        out = self.decode_head(features)   # (B, 2, H/4, W/4)
        # 上采样回原始分辨率
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out


# ======================== 对外调用函数 ========================
def get_segformer_b0_2class(device):
    """返回 SegFormer-B0 二分类模型，入参与返回值和其余模型一致。"""
    model = SegFormerB0(num_classes=2)
    model = model.to(device)
    return model
