"""
pipeline/data_module.py
统一 DataLoader 构建，封装 transform 预处理、数据增强。
支持 train/val/test，内置性能优化参数。
直接导入现有 CityscapesSegDataset 和 HaikouBlindDataset。
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# 确保能从上级目录导入数据集类
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from cityscapes_dataset import CityscapesSegDataset
from dataset_haikou import HaikouBlindDataset


def create_dataloaders(cfg):
    """
    根据配置创建训练集和验证集 DataLoader。

    参数:
        cfg — Config 实例，包含 data 子配置

    返回:
        (train_loader, val_loader, train_dataset, val_dataset)
    """
    # 选择数据集类
    dataset_cls = {
        'cityscapes': CityscapesSegDataset,
        'haikou': HaikouBlindDataset,
    }.get(cfg.data.dataset, CityscapesSegDataset)

    crop_size = cfg.data.crop_size

    # 创建数据集
    train_dataset = dataset_cls(
        root_dir=cfg.data.root_dir,
        split="train",
        crop_size=crop_size,
    )
    val_dataset = dataset_cls(
        root_dir=cfg.data.root_dir,
        split="val",
        crop_size=crop_size,
    )

    print(f"数据集: {cfg.data.dataset} | 训练集: {len(train_dataset)} 张 | 验证集: {len(val_dataset)} 张")

    # 构建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers if cfg.data.num_workers > 0 else False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers if cfg.data.num_workers > 0 else False,
    )

    return train_loader, val_loader, train_dataset, val_dataset
