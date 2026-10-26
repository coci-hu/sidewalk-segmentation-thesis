"""
model_ppliteseg.py
PP-LiteSeg-T 式（PaddlePaddle Lite Segmentation Tiny）轻量化语义分割网络，按论文结构自行实现，实测 2.71M 参数。
无注意力模块、无自定义改动，严格复现论文结构。

论文：PP-LiteSeg: A Superior Real-Time Semantic Segmentation Model (CVPR 2022)

架构总览：
  输入 (B, 3, H, W)
    ↓
  Encoder（STDC1 主干）
    Stage1: Conv 3→32,   stride=2 → (B,  32, H/2,  W/2)
    Stage2: STDC 32→64,  stride=2 → (B,  64, H/4,  W/4)  ← s2
    Stage3: STDC 64→128, stride=2 → (B, 128, H/8,  W/8)  ← s3
    Stage4: STDC 128→256,stride=2 → (B, 256, H/16, W/16) ← s4
    Stage5: STDC 256→512,stride=2 → (B, 512, H/32, W/32) ← s5
    ↓
  SPPM（空间金字塔池化模块）
    多尺度池化 [1×1, 2×2, 4×4] → 拼接融合 → (B, 128, H/32, W/32)
    ↓
  Decoder（3 级 FFM 逐级上采样融合）
    FFM1: up(SPPM) + s4 → (B, 128, H/16, W/16)
    FFM2: up(FFM1) + s3 → (B,  96, H/8,  W/8)
    FFM3: up(FFM2) + s2 → (B,  64, H/4,  W/4)
    ↓
  Seg Head: Conv3×3+BN+ReLU → Conv1×1 → (B, 2, H/4, W/4)
    ↓
  双线性插值上采样 → (B, 2, H, W)

对外接口：
  get_ppliteseg_2class(device)  →  返回已移至 device 的 PP-LiteSeg-T 模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================== 基础卷积块 ========================
class ConvBNReLU(nn.Module):
    """Conv2d + BatchNorm2d + ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# ======================== STDC 模块 ========================
class STDCModule(nn.Module):
    """
    Short-Term Dense Concatenate 模块。

    将输入依次经过 n 个 3×3 卷积分支，各分支输出通道递减，
    最后将所有分支输出在通道维度拼接，再经 1×1 卷积融合。

    当 stride>1 时，第一个分支使用 stride 下采样，
    后续分支在已下采样的特征上依次卷积，保证拼接时空间尺寸一致。

    参数：
        in_channels:  输入通道数
        out_channels: 输出通道数（融合后通道数）
        stride:       步幅（1 或 2），默认 1
        n:            分支数量（默认 4，即 STDC1 配置）
    """
    def __init__(self, in_channels, out_channels, stride=1, n=4):
        super().__init__()
        assert n >= 2, "STDC 分支数 n 必须 >= 2"
        self.n = n

        # 各分支输出通道数：第1分支占一半，后续分支均分剩余
        ch_list = [out_channels // 2]
        ch_remaining = out_channels // 2
        for i in range(1, n - 1):
            ch_list.append(ch_remaining // 2)
            ch_remaining -= ch_list[-1]
        ch_list.append(ch_remaining)

        # 分支0：3×3卷积，携带 stride 下采样
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, ch_list[0], 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(ch_list[0]),
            nn.ReLU(inplace=True),
        )

        # 分支1~n-1：3×3卷积，stride=1，依次串联
        self.convs = nn.ModuleList()
        for i in range(1, n):
            self.convs.append(nn.Sequential(
                nn.Conv2d(ch_list[i - 1], ch_list[i], 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(ch_list[i]),
                nn.ReLU(inplace=True),
            ))

        # 融合 1×1 卷积
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out_list = [self.conv0(x)]
        feat = out_list[0]
        for conv in self.convs:
            feat = conv(feat)
            out_list.append(feat)
        out = torch.cat(out_list, dim=1)
        out = self.fuse(out)
        return out


# ======================== STDC1 主干网络 ========================
class STDC1Backbone(nn.Module):
    """
    STDC1 主干网络，5 个 Stage 逐级提取特征。

    Stage1: Conv 3→32,   stride=2, 输出 H/2
    Stage2: STDC 32→64,  stride=2, 输出 H/4   ← s2
    Stage3: STDC 64→128, stride=2, 输出 H/8   ← s3
    Stage4: STDC 128→256,stride=2, 输出 H/16  ← s4
    Stage5: STDC 256→512,stride=2, 输出 H/32  ← s5
    """
    def __init__(self):
        super().__init__()
        self.stage1 = ConvBNReLU(3, 32, kernel_size=3, stride=2, padding=1)
        self.stage2 = STDCModule(32, 64, stride=2, n=4)
        self.stage3 = STDCModule(64, 128, stride=2, n=4)
        self.stage4 = STDCModule(128, 256, stride=2, n=4)
        self.stage5 = STDCModule(256, 512, stride=2, n=4)

    def forward(self, x):
        s1 = self.stage1(x)    # (B,  32, H/2,  W/2)
        s2 = self.stage2(s1)   # (B,  64, H/4,  W/4)
        s3 = self.stage3(s2)   # (B, 128, H/8,  W/8)
        s4 = self.stage4(s3)   # (B, 256, H/16, W/16)
        s5 = self.stage5(s4)   # (B, 512, H/32, W/32)
        return s2, s3, s4, s5


# ======================== SPPM 空间金字塔池化模块 ========================
class SPPM(nn.Module):
    """
    Spatial Pyramid Pooling Module（空间金字塔池化模块）。
    对深层特征进行多尺度池化，捕获全局上下文信息。

    步骤：
      1. 在三个尺度 [1×1, 2×2, 4×4] 上进行自适应平均池化
      2. 每个尺度经 1×1 卷积降维
      3. 上采样回原始尺寸后拼接
      4. 1×1 卷积融合输出
    """
    def __init__(self, in_channels, out_channels, pool_scales=[1, 2, 4]):
        super().__init__()
        mid_channels = in_channels // 4
        self.pools = nn.ModuleList()
        for scale in pool_scales:
            self.pools.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                nn.ReLU(inplace=True),
            ))
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_channels * len(pool_scales), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[2:]
        pool_outs = []
        for pool in self.pools:
            p = pool(x)
            p = F.interpolate(p, size=(h, w), mode='bilinear', align_corners=False)
            pool_outs.append(p)
        out = torch.cat(pool_outs, dim=1)
        out = self.fuse(out)
        return out


# ======================== FFM 灵活特征融合模块 ========================
class FFM(nn.Module):
    """
    Flexible Feature Fusion Module（灵活特征融合模块）。
    将低级编码器特征与高级解码器特征融合。

    步骤：
      1. 高级特征双线性上采样2倍至低级特征尺寸
      2. 通道维度拼接
      3. 3×3 卷积融合 + BN + ReLU
    """
    def __init__(self, low_channels, high_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(low_channels + high_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, low_feat, high_feat):
        high_up = self.up(high_feat)
        # 若尺寸不完全对齐，插值到 low_feat 尺寸
        if high_up.shape[2:] != low_feat.shape[2:]:
            high_up = F.interpolate(high_up, size=low_feat.shape[2:],
                                    mode='bilinear', align_corners=False)
        fuse = torch.cat([low_feat, high_up], dim=1)
        out = self.conv_fuse(fuse)
        return out


# ======================== 分割头 Seg Head ========================
class SegHead(nn.Module):
    """
    分割输出头：Conv3×3+BN+ReLU → Conv1×1 → 输出 num_classes 通道。
    """
    def __init__(self, in_channels, mid_channels, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_classes, 1),
        )

    def forward(self, x):
        return self.head(x)


# ======================== PP-LiteSeg-T 完整网络 ========================
class PPLiteSeg(nn.Module):
    """
    PP-LiteSeg-T 完整网络。

    流程：
      1. STDC1 主干提取 s2(H/4)、s3(H/8)、s4(H/16)、s5(H/32) 四级特征
      2. SPPM 对 s5 做空间金字塔池化 → 128ch, H/32
      3. FFM1: 融合 s4(256ch) + SPPM(128ch) → 128ch, H/16
      4. FFM2: 融合 s3(128ch) + FFM1(128ch) → 96ch,  H/8
      5. FFM3: 融合 s2(64ch)  + FFM2(96ch)  → 64ch,  H/4
      6. Seg Head 输出分类结果 (B, 2, H/4, W/4)
      7. 双线性插值上采样回原始分辨率 (B, 2, H, W)
    """
    def __init__(self, num_classes=2):
        super().__init__()
        # 编码器
        self.encoder = STDC1Backbone()

        # 空间金字塔池化
        self.sppm = SPPM(in_channels=512, out_channels=128)

        # 解码器：3 级 FFM
        self.ffm1 = FFM(low_channels=256, high_channels=128, out_channels=128)
        self.ffm2 = FFM(low_channels=128, high_channels=128, out_channels=96)
        self.ffm3 = FFM(low_channels=64,  high_channels=96,  out_channels=64)

        # 分割头
        self.seg_head = SegHead(64, 64, num_classes)

    def forward(self, x):
        _, _, H, W = x.shape
        # 编码器
        s2, s3, s4, s5 = self.encoder(x)
        # s2:(B,64,H/4,W/4)  s3:(B,128,H/8,W/8)
        # s4:(B,256,H/16,W/16)  s5:(B,512,H/32,W/32)

        # SPPM
        sppm_out = self.sppm(s5)  # (B,128,H/32,W/32)

        # 解码器逐级融合
        d1 = self.ffm1(s4, sppm_out)  # (B,128,H/16,W/16)
        d2 = self.ffm2(s3, d1)        # (B,96, H/8, W/8)
        d3 = self.ffm3(s2, d2)        # (B,64, H/4, W/4)

        # 分割头
        out = self.seg_head(d3)       # (B,2, H/4, W/4)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out


# ======================== 对外调用函数 ========================
def get_ppliteseg_2class(device):
    """返回 PP-LiteSeg-T 二分类模型，入参与返回值和基线 get_bisenetv2_2class 一致。"""
    model = PPLiteSeg(num_classes=2)
    model = model.to(device)
    return model
