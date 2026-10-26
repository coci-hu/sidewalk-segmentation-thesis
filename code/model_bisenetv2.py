import torch
import torch.nn as nn
# 离线本地实现：本文自写的精简双分支网络，论文显示名为 SBNet（非官方 BiSeNetV2）。
# 命名与标签口径见 ../code/NAMING_AND_LABELS.md。类名沿用历史名称，仅内部标识符，不进入论文。
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# 精简BiSeNetV2主干，输出改为2分类
class BiSeNetV2(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # 浅层细节分支
        self.detail_branch = nn.Sequential(
            ConvBNReLU(3, 64, 3, 2, 1),
            ConvBNReLU(64, 64, 3, 1, 1),
            ConvBNReLU(64, 128, 3, 2, 1),
        )
        # 深层语义分支（简化）
        self.semantic_branch = nn.Sequential(
            ConvBNReLU(3, 32, 3, 2, 1),
            ConvBNReLU(32, 64, 3, 2, 1),
            ConvBNReLU(64, 128, 3, 2, 1),
            ConvBNReLU(128, 256, 3, 2, 1),
        )
        # 融合头 + 分类输出
        self.head = nn.Sequential(
            ConvBNReLU(384, 128, 3, 1, 1),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
    def forward(self, x):
        detail = self.detail_branch(x)
        semantic = self.semantic_branch(x)
        semantic = nn.functional.interpolate(semantic, size=detail.shape[2:], mode='bilinear', align_corners=False)
        fuse = torch.cat([detail, semantic], dim=1)
        out = self.head(fuse)
        out = nn.functional.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out

# 对外调用函数，train.py直接调用不用改
def get_bisenetv2_2class(device):
    # 二分类：0背景，1人行道
    model = BiSeNetV2(num_classes=2)
    model = model.to(device)
    return model