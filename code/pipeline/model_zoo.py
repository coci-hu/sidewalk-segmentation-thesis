"""
pipeline/model_zoo.py
统一模型注册与加载入口。
支持加载预训练权重、迁移学习、主干冻结。
可以直接导入 components 下的 CA 注意力组件。
新增模型只需在此注册，不需要改训练循环。
"""

import os
import sys
import torch
import torch.nn as nn

# 确保能从上级目录导入模型文件
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


# ======================== 模型注册表 ========================
# 每个模型注册为 (构建函数, 参数量描述)
# 构建函数签名: build_fn(num_classes, device) -> nn.Module

def _build_bisenetv2(num_classes, device):
    from model_bisenetv2 import BiSeNetV2
    return BiSeNetV2(num_classes=num_classes).to(device)

def _build_bisenetv2_ca(num_classes, device):
    from model_bisenetv2_ca import BiSeNetV2_CA
    return BiSeNetV2_CA(num_classes=num_classes).to(device)

def _build_unet(num_classes, device):
    from model_unet import UNet
    return UNet(num_classes=num_classes).to(device)

def _build_deeplabv3plus(num_classes, device):
    from model_deeplabv3plus import DeepLabV3Plus
    return DeepLabV3Plus(num_classes=num_classes).to(device)

def _build_stdc_seg(num_classes, device):
    from model_stdc_seg import STDCSeg
    return STDCSeg(num_classes=num_classes).to(device)

def _build_ppliteseg(num_classes, device):
    from model_ppliteseg import PPLiteSeg
    return PPLiteSeg(num_classes=num_classes).to(device)

def _build_fastscnn(num_classes, device):
    from model_fastscnn import FastSCNN
    return FastSCNN(num_classes=num_classes).to(device)

def _build_enet(num_classes, device):
    from model_enet import ENet
    return ENet(num_classes=num_classes).to(device)

def _build_enet_ca(num_classes, device):
    from model_enet_ca import ENetCA
    return ENetCA(num_classes=num_classes).to(device)

def _build_segformer(num_classes, device):
    from model_segformer import SegFormerB0
    return SegFormerB0(num_classes=num_classes).to(device)

def _build_enet_ba(num_classes, device):
    from model_enet_ba import ENetBA
    return ENetBA(num_classes=num_classes).to(device)

def _build_enet_dba(num_classes, device):
    from model_enet_dba import ENetDBA
    return ENetDBA(num_classes=num_classes).to(device)

def _build_enet_laspp(num_classes, device):
    from model_enet_laspp import ENetLASSP
    return ENetLASSP(num_classes=num_classes).to(device)

def _build_enet_cbr(num_classes, device):
    from model_enet_cbr import ENetCBR
    return ENetCBR(num_classes=num_classes).to(device)

def _build_bisenetv2_laspp(num_classes, device):
    from model_bisenetv2_laspp import BiSeNetV2LASSP
    return BiSeNetV2LASSP(num_classes=num_classes).to(device)

def _build_bisenetv2_cbr(num_classes, device):
    from model_bisenetv2_cbr import BiSeNetV2CBR
    return BiSeNetV2CBR(num_classes=num_classes).to(device)


# 模型注册表：名称 → 构建函数
MODEL_REGISTRY = {
    "bisenetv2":       _build_bisenetv2,
    "bisenetv2_ca":    _build_bisenetv2_ca,
    "unet":            _build_unet,
    "deeplabv3plus":   _build_deeplabv3plus,
    "stdc_seg":        _build_stdc_seg,
    "ppliteseg":       _build_ppliteseg,
    "fastscnn":        _build_fastscnn,
    "enet":            _build_enet,
    "enet_ca":         _build_enet_ca,
    "enet_ba":         _build_enet_ba,
    "enet_dba":        _build_enet_dba,
    "enet_laspp":      _build_enet_laspp,
    "enet_cbr":        _build_enet_cbr,
    "bisenetv2_laspp": _build_bisenetv2_laspp,
    "bisenetv2_cbr":   _build_bisenetv2_cbr,
    "segformer_b0":    _build_segformer,
}


def build_model(cfg, device=None):
    """
    根据配置构建模型，支持预训练权重加载和骨干冻结。

    参数:
        cfg    — Config 实例，包含 model 子配置
        device — torch.device，默认自动选择

    返回:
        model (nn.Module)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = cfg.model.name
    num_classes = cfg.model.num_classes

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"未知模型 '{model_name}'，可选: {list(MODEL_REGISTRY.keys())}\n"
            f"新增模型请在 pipeline/model_zoo.py 的 MODEL_REGISTRY 中注册。"
        )

    # 构建模型
    build_fn = MODEL_REGISTRY[model_name]
    model = build_fn(num_classes, device)
    print(f"模型构建: {model_name} | 参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 加载预训练权重
    if cfg.model.pretrained and os.path.exists(cfg.model.pretrained):
        state_dict = torch.load(cfg.model.pretrained, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"已加载预训练权重: {cfg.model.pretrained}")

    # 冻结骨干
    if cfg.model.freeze_backbone:
        _freeze_backbone(model, model_name)
        print("骨干网络已冻结（requires_grad=False）")

    return model


def _freeze_backbone(model, model_name):
    """冻结骨干网络参数（仅训练解码头）"""
    # 通用策略：冻结前几个大模块，保留最后的 head/decode 部分
    freeze_keywords = {
        "bisenetv2":     ["detail_branch", "semantic_branch"],
        "bisenetv2_ca":  ["detail_branch", "semantic_branch"],
        "unet":          ["enc1", "enc2", "enc3", "enc4"],
        "deeplabv3plus": ["backbone"],
        "stdc_seg":      ["encoder"],
        "ppliteseg":     ["encoder"],
        "fastscnn":      ["learning_to_down", "global_feature"],
        "enet":          ["initial_block", "downsample1", "downsample2"],
        "segformer_b0":  ["patch_embed", "block"],
    }

    keywords = freeze_keywords.get(model_name, [])
    frozen_count = 0
    for name, param in model.named_parameters():
        if any(kw in name for kw in keywords):
            param.requires_grad = False
            frozen_count += 1
    print(f"冻结了 {frozen_count} 个参数")
