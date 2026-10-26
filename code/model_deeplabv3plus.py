"""
model_deeplabv3plus.py
DeepLabv3+ 二分类语义分割网络（ResNet-50 骨干，随机初始化，无预训练权重）。

架构（output_stride = 16）：
  Backbone  : ResNet-50（conv1 stride=2 + maxpool stride=2 → 1/4；
              layer3 stride=2 → 1/16；layer4 stride=1 + dilation=2 → 保持 1/16）
  ASPP      : Atrous Spatial Pyramid Pooling（rates=[6,12,18] + 全局池化分支）
  Decoder   : 1×1 卷积上采样 + layer1 低级特征融合

注意：ASPP 的全局池化分支输出 1×1 特征图，其后接 BatchNorm2d。
     训练模式下 batch size 必须 ≥ 2，否则会抛
     ValueError: Expected more than 1 value per channel。
     因此 train_deeplab.py 的 train_loader 必须设置 drop_last=True。

  输入：(B, 3, H, W)   输出：(B, 2, H, W)

对外接口：
  get_deeplabv3plus_2class(device)  →  返回已移至 device 的 DeepLabv3+ 模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================== ResNet-50 基础组件 ========================
class Bottleneck(nn.Module):
    """ResNet-50 残差块（expansion=4）"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet50(nn.Module):
    """
    ResNet-50 骨干网络，输出多尺度特征：
      - low_level: layer1 输出 (B, 256, H/4, W/4)  — 解码器低级特征
      - high_level: layer4 输出 (B, 2048, H/16, W/16) — 送入 ASPP
    为适配 DeepLabv3+，使用 output_stride=16（论文标准配置）：
      conv1 stride=2 + maxpool stride=2 → 1/4 分辨率
      layer3 stride=2 → 1/16 分辨率
      layer4 stride=1 + dilation=2 → 保持 1/16 分辨率
    """

    def __init__(self):
        super().__init__()
        self.inplanes = 64

        # conv1: stride=2（标准下采样）
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # maxpool: stride=2（标准下采样）
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # layer1: stride=1 → 1/4 分辨率
        self.layer1 = self._make_layer(64, 3, stride=1, dilation=1)
        # layer2: stride=2 → 1/8 分辨率
        self.layer2 = self._make_layer(128, 4, stride=2, dilation=1)
        # layer3: stride=2 → 1/16 分辨率
        self.layer3 = self._make_layer(256, 6, stride=2, dilation=1)
        # layer4: stride=1, dilation=2 → 保持 1/16 分辨率（output_stride=16）
        self.layer4 = self._make_layer(512, 3, stride=1, dilation=2)

    def _make_layer(self, planes, blocks, stride, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )
        layers = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # (B, 64, H/2, W/2)
        x = self.maxpool(x)                      # (B, 64, H/4, W/4)
        low_level = self.layer1(x)               # (B, 256, H/4, W/4)
        x = self.layer2(low_level)               # (B, 512, H/8, W/8)
        x = self.layer3(x)                       # (B, 1024, H/16, W/16)
        x = self.layer4(x)                       # (B, 2048, H/16, W/16)
        return low_level, x  # low: (B,256,H/4,W/4), high: (B,2048,H/16,W/16)


# ======================== ASPP 模块 ========================
class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling（rates=[6,12,18]）"""

    def __init__(self, in_channels=2048, out_channels=256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        h, w = x.shape[2:]
        p1 = self.conv1(x)
        p2 = self.conv2(x)
        p3 = self.conv3(x)
        p4 = self.conv4(x)
        p5 = F.interpolate(self.pool(x), size=(h, w), mode='bilinear', align_corners=False)
        return self.project(torch.cat([p1, p2, p3, p4, p5], dim=1))


# ======================== DeepLabv3+ 完整网络 ========================
class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = ResNet50()
        self.aspp = ASPP(in_channels=2048, out_channels=256)

        # 解码器：融合 ASPP 输出 + low_level 特征
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        input_size = x.shape[2:]
        low_level, high_level = self.backbone(x)

        # ASPP 编码
        aspp_out = self.aspp(high_level)  # (B, 256, H/4, W/4)

        # 解码：上采样 ASPP 输出 + 投影后的 low_level 特征
        aspp_up = F.interpolate(aspp_out, size=low_level.shape[2:],
                                mode='bilinear', align_corners=False)
        low_feat = self.low_level_proj(low_level)
        dec_in = torch.cat([aspp_up, low_feat], dim=1)
        dec_out = self.decoder(dec_in)

        out = self.head(dec_out)
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        return out


# ======================== 对外调用函数 ========================
def get_deeplabv3plus_2class(device):
    """返回 DeepLabv3+ 二分类模型，入参与返回值和基线 get_bisenetv2_2class 一致。"""
    model = DeepLabV3Plus(num_classes=2)
    model = model.to(device)
    return model
