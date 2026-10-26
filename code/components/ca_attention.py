"""
components/ca_attention.py
坐标注意力（Coordinate Attention）通用可复用组件。

论文: Coordinate Attention for Efficient Mobile Network Design (CVPR 2021)
可独立插入任意 CNN 骨干或特征融合层，不改变输入通道数和空间尺寸。

使用方式：
    from components.ca_attention import CoordAtt
    ca = CoordAtt(channels=256, reduction=32)
    out = ca(feature_map)  # 输入输出形状完全一致

注意事项：
    - 类名 CoordAtt（与原论文命名一致）
    - 内部参数名与 model_bisenetv2_ca.py 中 CoordinateAttention 完全对齐，
      保证 BiSeNetV2_CA 旧权重可直接 load。
"""

import torch
import torch.nn as nn


class CoordAtt(nn.Module):
    """
    坐标注意力模块（Coordinate Attention）。

    沿 X/Y 两个方向分别做全局平均池化，捕获长距离依赖的同时保留位置信息。
    输出与输入形状完全一致，可作为即插即用模块插入任意 CNN。

    参数:
        channels   — 输入/输出通道数（本模块不改变通道数）
        reduction  — 中间瓶颈通道的缩减因子，默认 32
        inp        — [向后兼容] 等同于 channels（旧版 model_bisenetv2_ca.py 使用 inp/oup）
        oup        — [向后兼容] 等同于 channels（旧版中 inp == oup）
    """

    def __init__(self, channels=None, reduction=32, inp=None, oup=None):
        super().__init__()
        # 向后兼容：支持旧版 inp/oup 参数名
        if channels is None and inp is not None:
            channels = inp
        if channels is None:
            raise ValueError("CoordAtt 需要指定 channels 参数（或使用旧版 inp 参数名）")
        # oup 参数兼容（旧版中 oup == channels）
        _ = oup  # 忽略 oup，输出通道与输入一致

        # 中间瓶颈通道数，至少 8
        mip = max(8, channels // reduction)

        # 1×1 卷积：通道压缩 + 信息融合
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)

        # 两个 1×1 卷积分别生成 H / W 方向的注意力权重
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # 1) 沿 H 方向平均池化 → (B, C, 1, W)
        x_h = x.mean(dim=2, keepdim=True)
        # 2) 沿 W 方向平均池化 → (B, C, H, 1) → 转置为 (B, C, 1, H) 方便拼接
        x_w = x.mean(dim=3, keepdim=True).permute(0, 1, 3, 2)

        # 3) 拼接 → (B, C, 1, W+H)
        y = torch.cat([x_h, x_w], dim=3)
        # 4) 1×1 卷积压缩通道 + BN + ReLU
        y = self.act(self.bn1(self.conv1(y)))

        # 5) 拆分回两个方向
        x_h, x_w = torch.split(y, [w, h], dim=3)
        # 6) x_w 恢复为 (B, C_mid, H, 1)
        x_w = x_w.permute(0, 1, 3, 2)

        # 7) 生成注意力权重并 sigmoid 归一化
        a_h = self.conv_h(x_h).sigmoid()   # (B, C, 1, W)
        a_w = self.conv_w(x_w).sigmoid()   # (B, C, H, 1)

        # 8) 逐通道加权
        return identity * a_w * a_h


# ======================== 向后兼容别名 ========================
# 旧版 model_bisenetv2_ca.py 中使用的是 CoordinateAttention 类名，
# 此别名保证从 components.ca_attention 导入时也能识别旧名
CoordinateAttention = CoordAtt
