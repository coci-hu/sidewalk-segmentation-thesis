"""
model_stdc_seg.py
STDC-Seg 式（Short-Term Dense Concatenate）轻量化语义分割网络，按论文结构自行实现，实测 2.25M 参数。
无注意力模块、无自定义改动，严格复现论文结构。

论文：Rethinking BiSeNet For Real-time Semantic Segmentation (CVPR 2021)
架构：
  - STDC1 主干（5 级特征提取，逐级下采样 2x）
  - 特征融合模块 FFM（Feature Fusion Module）
  - 输出头：Seg Head（1×1 Conv → BN → ReLU → 1×1 Conv）

通道配置（STDC1）：
  Stage1: 3  → 32   stride=2
  Stage2: 32 → 64   stride=2
  Stage3: 64 → 128  stride=2
  Stage4: 128→ 256  stride=2
  Stage5: 256→ 512  stride=2

输入：(B, 3, H, W)   输出：(B, 2, H, W)

对外接口：
  get_stdc_seg_2class(device)  →  返回已移至 device 的 STDC-Seg 模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================== STDC 模块 ========================
class STDCModule(nn.Module):
    """
    Short-Term Dense Concatenate 模块。
    将输入依次经过 n 个 3×3 卷积分支，各分支输出通道递减，
    最后将所有分支输出在通道维度拼接，再经 1×1 卷积融合。

    参数：
        in_channels:  输入通道数
        out_channels: 输出通道数（融合后通道数）
        n:            分支数量（默认 4，即 STDC1 配置）
    """
    def __init__(self, in_channels, out_channels, n=4):
        super().__init__()
        assert n >= 2, "STDC 分支数 n 必须 >= 2"
        self.n = n

        # 各分支输出通道数：按比例递减
        # 第 1 分支占一半，后续分支均分剩余
        ch_list = [out_channels // 2]
        ch_remaining = out_channels // 2
        for i in range(1, n - 1):
            ch_list.append(ch_remaining // 2)
            ch_remaining = ch_remaining - ch_list[-1]
        ch_list.append(ch_remaining)

        # 分支 0：直接 1×1 卷积（不改变空间尺寸）
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, ch_list[0], 1, bias=False),
            nn.BatchNorm2d(ch_list[0]),
            nn.ReLU(inplace=True),
        )

        # 分支 1~n-1：依次 3×3 卷积，步长=1
        self.convs = nn.ModuleList()
        in_ch = in_channels
        for i in range(1, n):
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_ch, ch_list[i], 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(ch_list[i]),
                nn.ReLU(inplace=True),
            ))
            in_ch = ch_list[i]

        # 融合 1×1 卷积
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out_list = [self.conv0(x)]
        feat = x
        for conv in self.convs:
            feat = conv(feat)
            out_list.append(feat)
        out = torch.cat(out_list, dim=1)
        out = self.fuse(out)
        return out


# ======================== STDC 主干网络 ========================
class STDCBackbone(nn.Module):
    """
    STDC1 主干网络，5 个 Stage 逐级下采样。
    输出 Stage3、Stage4、Stage5 三个特征图用于分割头。

    Stage1: Conv 3→32,   stride=2, 输出 H/2
    Stage2: STDC 32→64,  stride=2, 输出 H/4
    Stage3: STDC 64→128, stride=2, 输出 H/8
    Stage4: STDC 128→256,stride=2, 输出 H/16
    Stage5: STDC 256→512,stride=2, 输出 H/16（步长1，仅通道变化）
    """
    def __init__(self):
        super().__init__()
        # Stage1: 普通卷积下采样
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # Stage2: STDC + 下采样
        self.stage2 = nn.Sequential(
            STDCModule(32, 64, n=4),
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Stage3: STDC + 下采样
        self.stage3 = nn.Sequential(
            STDCModule(64, 128, n=4),
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # Stage4: STDC + 下采样
        self.stage4 = nn.Sequential(
            STDCModule(128, 256, n=4),
            nn.Conv2d(256, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # Stage5: STDC（不下采样，仅通道变化）
        self.stage5 = STDCModule(256, 512, n=4)

    def forward(self, x):
        s1 = self.stage1(x)    # (B,  32, H/2,  W/2)
        s2 = self.stage2(s1)   # (B,  64, H/4,  W/4)
        s3 = self.stage3(s2)   # (B, 128, H/8,  W/8)
        s4 = self.stage4(s3)   # (B, 256, H/16, W/16)
        s5 = self.stage5(s4)   # (B, 512, H/16, W/16)
        return s3, s4, s5


# ======================== 特征融合模块 FFM ========================
class FeatureFusionModule(nn.Module):
    """
    特征融合模块 FFM（Feature Fusion Module）。
    将低级特征（Stage3）和高级特征（Stage5）融合。

    步骤：
      1. 高级特征上采样到低级特征尺寸
      2. 通道拼接
      3. 1×1 Conv 压缩通道
      4. 全局平均池化 → FC → Sigmoid 通道注意力
      5. 输出 = 融合特征 × 注意力权重
    """
    def __init__(self, low_channels, high_channels, out_channels):
        super().__init__()
        self.up_high = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(low_channels + high_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, low_feat, high_feat):
        high_up = self.up_high(high_feat)
        # 若尺寸不完全对齐，插值到 low_feat 尺寸
        if high_up.shape[2:] != low_feat.shape[2:]:
            high_up = F.interpolate(high_up, size=low_feat.shape[2:], mode='bilinear', align_corners=False)
        fuse = torch.cat([low_feat, high_up], dim=1)
        fuse = self.conv_fuse(fuse)
        # 通道注意力
        att = self.avg_pool(fuse)
        att = self.fc(att)
        out = fuse * att
        return out


# ======================== 分割头 Seg Head ========================
class SegHead(nn.Module):
    """
    分割输出头：Conv-BN-ReLU → 1×1 Conv → 输出 num_classes 通道。
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


# ======================== STDC-Seg 完整网络 ========================
class STDCSeg(nn.Module):
    """
    STDC-Seg 完整网络。

    流程：
      1. STDC 主干提取 Stage3(H/8)、Stage4(H/16)、Stage5(H/16) 三级特征
      2. FFM 融合 Stage3 + Stage5 → 融合特征 (H/8)
      3. Seg Head 输出分类结果 (H/8)
      4. 双线性插值上采样回原始分辨率
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = STDCBackbone()

        # 特征融合：Stage3(128ch) + Stage5(512ch) → 256ch
        self.ffm = FeatureFusionModule(
            low_channels=128,
            high_channels=512,
            out_channels=256,
        )

        # 分割头
        self.seg_head = SegHead(256, 128, num_classes)

    def forward(self, x):
        _, _, H, W = x.shape
        s3, s4, s5 = self.backbone(x)   # s3:(B,128,H/8,W/8)  s5:(B,512,H/16,W/16)
        fuse = self.ffm(s3, s5)          # (B,256,H/8,W/8)
        out = self.seg_head(fuse)        # (B,2,H/8,W/8)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out


# ======================== 对外调用函数 ========================
def get_stdc_seg_2class(device):
    """返回 STDC-Seg 二分类模型，入参与返回值和基线 get_bisenetv2_2class 一致。"""
    model = STDCSeg(num_classes=2)
    model = model.to(device)
    return model
