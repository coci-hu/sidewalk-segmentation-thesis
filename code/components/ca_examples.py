"""
components/ca_examples.py
演示如何将 CoordAtt（坐标注意力）插入其他基线模型。

本文件仅为示例参考，不参与训练流程。
用法：将 CoordAtt 模块插入到 CNN 骨干的特征输出层或融合层，
      不改变通道数和空间尺寸，即可即插即用。

插入原则：
  1. CA 放在"特征提取完毕、融合/上采样之前"效果最好
  2. CA 输入输出形状完全一致，不需要修改后续层的通道数
  3. 适用于任意 CNN 特征图，推荐放在 256~512 通道的深层特征上
"""

import torch
import torch.nn as nn
from components.ca_attention import CoordAtt


# ====================== 示例1：STDC-Seg + CA ======================
# 在 STDC-Seg 的 SPPM 输出后插入 CA，增强多尺度特征的通道-空间注意力
class STDCSegWithCA(nn.Module):
    """
    STDC-Seg + CA 示例骨架（仅演示插入位置，非完整实现）。
    实际使用时请基于 model_stdc_seg.py 修改。
    """
    def __init__(self, num_classes=2):
        super().__init__()
        # 原始编码器（示意）
        self.encoder = nn.Identity()  # 实际替换为 STDC1Backbone
        # CA 模块插入点：SPPM 输出后
        self.ca = CoordAtt(channels=128, reduction=16)
        # 后续融合和解码头
        self.decode_head = nn.Identity()  # 实际替换为 FFM + SegHead

    def forward(self, x):
        feats = self.encoder(x)
        # ★ 在特征融合前插入 CA
        feats = self.ca(feats)
        out = self.decode_head(feats)
        return out


# ====================== 示例2：U-Net + CA ======================
# 在 U-Net 的瓶颈层（bottleneck）输出后插入 CA
class UNetBottleneckWithCA(nn.Module):
    """
    U-Net + CA 示例骨架（仅演示插入位置）。
    在 bottleneck 1024 通道特征上应用 CA，增强编码器-解码器桥接的注意力。
    """
    def __init__(self, num_classes=2):
        super().__init__()
        # U-Net 编码器到瓶颈层（示意）
        self.encoder = nn.Identity()
        # ★ CA 插入瓶颈层
        self.ca_bottleneck = CoordAtt(channels=1024, reduction=32)
        # 解码器
        self.decoder = nn.Identity()

    def forward(self, x):
        feat = self.encoder(x)
        feat = self.ca_bottleneck(feat)  # ★ 瓶颈层增强
        out = self.decoder(feat)
        return out


# ====================== 示例3：通用 CNN + CA 插入方法 ======================
def insert_ca_after_layer(model, layer_name, channels, reduction=32):
    """
    通用函数：在已有模型的指定层后插入 CoordAtt。

    参数:
        model     — 目标模型实例
        layer_name — 要在其后插入 CA 的属性名（str）
        channels  — 该层输出通道数
        reduction — CA 缩减因子

    用法示例（伪代码）:
        model = get_bisenetv2_2class(device)
        insert_ca_after_layer(model, 'semantic_branch', channels=256, reduction=32)
        # 此时 model.semantic_branch 输出后自动经过 CA 注意力
    """
    original_layer = getattr(model, layer_name)
    ca_module = CoordAtt(channels=channels, reduction=reduction)

    # 将原始层和 CA 包装为 Sequential
    setattr(model, layer_name, nn.Sequential(original_layer, ca_module))
    return model


if __name__ == "__main__":
    # 验证 CA 模块前向传播
    x = torch.randn(2, 256, 48, 64)
    ca = CoordAtt(channels=256, reduction=32)
    y = ca(x)
    print(f"CoordAtt 验证: input={x.shape}, output={y.shape}, params={sum(p.numel() for p in ca.parameters())}")

    # 验证插入不影响形状
    x2 = torch.randn(2, 1024, 24, 32)
    ca2 = CoordAtt(channels=1024, reduction=32)
    y2 = ca2(x2)
    print(f"CoordAtt-1024 验证: input={x2.shape}, output={y2.shape}")
