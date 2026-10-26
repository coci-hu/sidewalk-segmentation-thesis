"""
model_fastscnn.py
Fast-SCNN 轻量化语义分割网络（按官方论文结构自行实现，二分类可通行区域/背景）。

论文：Fast-SCNN: Fast Semantic Segmentation Network (BMVC 2019)
特点：实测约 1.47M 参数，两分支共享浅层特征，速度极快，适合实时场景。

架构总览：
  输入 (B, 3, H, W)
    ↓
  ┌─────────────────────────────────────┐
  │  Learning to Downsample（浅层下采样）│
  │  Conv 3→32, s=2  → (B, 32, H/2,  W/2) │
  │  DWConv 32→32, s=2 → (B, 32, H/4,  W/4) │
  │  DWConv 32→32, s=2 → (B, 32, H/8,  W/8) │
  │  Conv 32→64, 1×1 → (B, 64, H/8,  W/8)  │ ← 浅层特征送入 FFM
  └─────────────────────────────────────┘
    ↓
  ┌─────────────────────────────────────┐
  │  Global Feature Extractor（全局特征）│
  │  InvertedResidual 64→128, s=2 ×1    │
  │  InvertedResidual 128→128, s=1 ×2   │
  │  InvertedResidual 128→128, s=2 ×1   │
  │  InvertedResidual 128→128, s=1 ×2   │
  │  PPM [1,2,4] → Conv融合             │
  └─────────────────────────────────────┘
    → (B, 128, H/32, W/32) ← 深层特征送入 FFM
    ↓
  ┌─────────────────────────────────────┐
  │  Feature Fusion Module（特征融合）  │
  │  浅层: Conv1×1 64→128               │
  │  深层: 上采样4× → 拼接               │
  │  Conv3×3 256→128 融合               │
  └─────────────────────────────────────┘
    → (B, 128, H/8, W/8)
    ↓
  ┌─────────────────────────────────────┐
  │  Classifier（分割头）               │
  │  DWConv3×3 128→128 + Conv1×1 → 2ch │
  │  双线性插值回原始分辨率              │
  └─────────────────────────────────────┘
    → (B, 2, H, W)

对外接口：
  get_fastscnn_2class(device)  →  返回已移至 device 的 Fast-SCNN 模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================== 深度可分离卷积 ========================
class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积 = DepthwiseConv + PointwiseConv(1×1)。
    大幅减少参数量和计算量，是轻量化网络的核心构建块。

    参数：
        in_channels:  输入通道数
        out_channels: 输出通道数
        kernel_size:  深度卷积核大小，默认 3
        stride:       步幅，默认 1
        padding:      填充，默认 1（对应 kernel_size=3）
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # 深度卷积：每个输入通道独立卷积
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.relu_dw = nn.ReLU(inplace=True)
        # 逐点卷积：1×1 卷积跨通道融合
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_channels)
        self.relu_pw = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu_dw(self.bn_dw(self.depthwise(x)))
        x = self.relu_pw(self.bn_pw(self.pointwise(x)))
        return x


# ======================== 倒残差模块（MobileNetV2风格） ========================
class InvertedResidual(nn.Module):
    """
    倒残差模块：先扩展通道 → 深度卷积 → 压缩通道。
    当 stride=1 且输入输出通道相同时，使用残差连接。

    参数：
        in_channels:  输入通道数
        out_channels: 输出通道数
        stride:       步幅（1 或 2）
        expand_ratio: 扩展倍率，默认 6
    """
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = (stride == 1 and in_channels == out_channels)

        layers = []
        # 扩展阶段：1×1 卷积升维
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ])
        # 深度卷积阶段：3×3 深度卷积
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        ])
        # 压缩阶段：1×1 卷积降维（不使用 ReLU）
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


# ======================== 金字塔池化模块（PPM） ========================
class PyramidPoolingModule(nn.Module):
    """
    金字塔池化模块：在多个尺度上捕获全局上下文信息。

    对输入特征分别在 [1×1, 2×2, 4×4] 三个尺度做自适应平均池化，
    每个尺度经 1×1 卷积降维后上采样回原尺寸，最后拼接并融合。

    参数：
        in_channels:  输入通道数（Fast-SCNN 中为 128）
        out_channels: 输出通道数
        pool_scales:  池化尺度列表
    """
    def __init__(self, in_channels, out_channels, pool_scales=[1, 2, 4]):
        super().__init__()
        mid_channels = in_channels // 4  # 每个分支的中间通道数

        self.pools = nn.ModuleList()
        for scale in pool_scales:
            self.pools.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                nn.ReLU(inplace=True),
            ))
        # 融合卷积：拼接后的通道数 = mid_channels × len(pool_scales)
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


# ======================== 浅层下采样模块 ========================
class LearningToDownsample(nn.Module):
    """
    浅层下采样模块：快速提取空间细节特征。
    使用深度可分离卷积大幅降低计算量，同时保留浅层几何/边缘信息，
    供后续 Feature Fusion Module 使用。

    输出：
        (B, 64, H/8, W/8) — 浅层特征
    """
    def __init__(self):
        super().__init__()
        # 第1层：标准卷积，3→32，步幅2下采样
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # 第2层：深度可分离卷积，32→32，步幅2下采样
        self.dsconv1 = DepthwiseSeparableConv(32, 32, kernel_size=3, stride=2, padding=1)
        # 第3层：深度可分离卷积，32→32，步幅2下采样
        self.dsconv2 = DepthwiseSeparableConv(32, 32, kernel_size=3, stride=2, padding=1)
        # 第4层：1×1 卷积升维，32→64
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)       # (B, 32, H/2,  W/2)
        x = self.dsconv1(x)     # (B, 32, H/4,  W/4)
        x = self.dsconv2(x)     # (B, 32, H/8,  W/8)
        x = self.conv2(x)       # (B, 64, H/8,  W/8)
        return x


# ======================== 全局特征提取模块 ========================
class GlobalFeatureExtractor(nn.Module):
    """
    全局特征提取模块：在低分辨率上捕获上下文语义信息。
    使用 MobileNetV2 风格的倒残差模块 + 金字塔池化模块。

    输入：(B, 64, H/8, W/8)
    输出：(B, 128, H/32, W/32)
    """
    def __init__(self):
        super().__init__()
        # ---- 步幅2下采样：64 → 128 ----
        self.bottleneck1 = InvertedResidual(64, 128, stride=2, expand_ratio=6)
        # ---- 步幅1重复 ×2：128 → 128 ----
        self.bottleneck2 = nn.Sequential(
            InvertedResidual(128, 128, stride=1, expand_ratio=6),
            InvertedResidual(128, 128, stride=1, expand_ratio=6),
        )
        # ---- 步幅2下采样：128 → 128 ----
        self.bottleneck3 = InvertedResidual(128, 128, stride=2, expand_ratio=6)
        # ---- 步幅1重复 ×2：128 → 128 ----
        self.bottleneck4 = nn.Sequential(
            InvertedResidual(128, 128, stride=1, expand_ratio=6),
            InvertedResidual(128, 128, stride=1, expand_ratio=6),
        )
        # ---- 金字塔池化 ----
        self.ppm = PyramidPoolingModule(in_channels=128, out_channels=128)

    def forward(self, x):
        x = self.bottleneck1(x)   # (B, 128, H/16, W/16)
        x = self.bottleneck2(x)   # (B, 128, H/16, W/16)
        x = self.bottleneck3(x)   # (B, 128, H/32, W/32)
        x = self.bottleneck4(x)   # (B, 128, H/32, W/32)
        x = self.ppm(x)           # (B, 128, H/32, W/32)
        return x


# ======================== 特征融合模块 ========================
class FeatureFusionModule(nn.Module):
    """
    特征融合模块：将浅层空间细节与深层语义信息融合。

    流程：
      1. 浅层特征经 1×1 卷积降通道（与深层对齐）
      2. 深层特征双线性上采样 4 倍（对齐空间尺寸）
      3. 通道维度拼接 → 3×3 卷积融合

    参数：
        low_channels:  浅层特征通道数（64）
        high_channels: 深层特征通道数（128）
        out_channels:  融合输出通道数（128）
    """
    def __init__(self, low_channels, high_channels, out_channels):
        super().__init__()
        # 浅层分支：1×1 卷积调整通道数
        self.low_conv = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # 深层分支：上采样
        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        # 融合卷积
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(high_channels + out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, low_feat, high_feat):
        # 浅层：1×1 降通道
        low = self.low_conv(low_feat)       # (B, 128, H/8, W/8)
        # 深层：上采样4倍到 H/8
        high = self.up(high_feat)           # (B, 128, H/8, W/8)
        # 尺寸对齐（防止除不尽导致差1像素）
        if high.shape[2:] != low.shape[2:]:
            high = F.interpolate(high, size=low.shape[2:], mode='bilinear', align_corners=False)
        # 拼接融合
        out = torch.cat([low, high], dim=1)  # (B, 256, H/8, W/8)
        out = self.fuse_conv(out)             # (B, 128, H/8, W/8)
        return out


# ======================== 分割分类头 ========================
class Classifier(nn.Module):
    """
    分类头：深度可分离卷积细化 + 1×1 卷积输出类别。

    参数：
        in_channels:  输入通道数（128）
        num_classes:  类别数（2：背景 + 可通行区域）
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            # 深度可分离卷积细化特征
            DepthwiseSeparableConv(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            # 1×1 卷积输出类别
            nn.Conv2d(in_channels, num_classes, 1),
        )

    def forward(self, x):
        return self.conv(x)


# ======================== Fast-SCNN 完整网络 ========================
class FastSCNN(nn.Module):
    """
    Fast-SCNN 完整网络。

    流程：
      1. Learning to Downsample：提取浅层空间特征 (B, 64, H/8, W/8)
      2. Global Feature Extractor：提取深层语义特征 (B, 128, H/32, W/32)
      3. Feature Fusion Module：融合浅层+深层特征 (B, 128, H/8, W/8)
      4. Classifier：输出分类 logits (B, 2, H/8, W/8)
      5. 双线性插值上采样到原始分辨率 (B, 2, H, W)

    参数：
        num_classes: 分类数，默认 2（背景 + 可通行区域）
    """
    def __init__(self, num_classes=2):
        super().__init__()
        # 浅层下采样
        self.learning_to_downsample = LearningToDownsample()
        # 全局特征提取
        self.global_feature_extractor = GlobalFeatureExtractor()
        # 特征融合
        self.feature_fusion = FeatureFusionModule(
            low_channels=64, high_channels=128, out_channels=128
        )
        # 分类头
        self.classifier = Classifier(in_channels=128, num_classes=num_classes)

    def forward(self, x):
        _, _, H, W = x.shape
        # 浅层特征
        low_feat = self.learning_to_downsample(x)        # (B, 64, H/8, W/8)
        # 深层特征
        high_feat = self.global_feature_extractor(low_feat)  # (B, 128, H/32, W/32)
        # 特征融合
        fused = self.feature_fusion(low_feat, high_feat)     # (B, 128, H/8, W/8)
        # 分类
        out = self.classifier(fused)                          # (B, 2, H/8, W/8)
        # 上采样回原始分辨率
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out


# ======================== 对外调用函数 ========================
def get_fastscnn_2class(device):
    """返回 Fast-SCNN 二分类模型，入参与返回值和其余模型一致。"""
    model = FastSCNN(num_classes=2)
    model = model.to(device)
    return model
