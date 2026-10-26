"""
dataset_haikou.py
海口可通行区域自建数据集加载类。
接口、预处理、裁剪尺寸与 cityscapes_dataset.py 完全一致，可一键切换数据集。

目录结构（由 split_haikou_dataset.py 生成）：
  dataset/haikou_blind/split/
    train/
      images/<场景名>/*.jpg
      masks/<场景名>/*.png   （可通行区域=1，背景=0）
    val/
      images/<场景名>/*.jpg
      masks/<场景名>/*.png

使用方式：
  from dataset_haikou import HaikouBlindDataset
  dataset = HaikouBlindDataset("../dataset/haikou_blind/split", split="train", crop_size=(768, 1024))
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class HaikouBlindDataset(Dataset):
    """
    海口可通行区域二分类数据集。
    接口与 CityscapesSegDataset 完全一致：
      - __init__(root_dir, split, crop_size)
      - __getitem__ 返回 (img_tensor, mask_tensor)
      - img_tensor: (3, H, W) 标准化浮点
      - mask_tensor: (H, W) long，可通行区域=1 背景=0
    """

    def __init__(self, root_dir, split="train", crop_size=(768, 1024)):
        self.img_dir = os.path.join(root_dir, split, "images")
        self.mask_dir = os.path.join(root_dir, split, "masks")
        self.crop_h, self.crop_w = crop_size

        self.img_paths = []
        self.mask_paths = []

        # 遍历场景子文件夹，匹配原图与掩码
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"图片目录不存在: {self.img_dir}")

        for scene in os.listdir(self.img_dir):
            scene_img_dir = os.path.join(self.img_dir, scene)
            scene_mask_dir = os.path.join(self.mask_dir, scene)
            if not os.path.isdir(scene_img_dir):
                continue
            for img_name in sorted(os.listdir(scene_img_dir)):
                if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    continue
                img_path = os.path.join(scene_img_dir, img_name)
                # 掩码同名但扩展名为 .png
                mask_name = os.path.splitext(img_name)[0] + ".png"
                mask_path = os.path.join(scene_mask_dir, mask_name)
                if os.path.exists(mask_path):
                    self.img_paths.append(img_path)
                    self.mask_paths.append(mask_path)

        # 图像标准化预处理（与 Cityscapes 完全一致）
        self.transform_img = transforms.Compose([
            transforms.Resize((self.crop_h, self.crop_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        # 掩码二值化：确保只有 0 和 1
        mask_np = np.array(mask)
        binary_mask = np.zeros_like(mask_np, dtype=np.uint8)
        binary_mask[mask_np > 0] = 1  # 任何非零像素视为可通行区域

        mask_pil = Image.fromarray(binary_mask)
        mask_pil = mask_pil.resize((self.crop_w, self.crop_h), Image.NEAREST)

        img_tensor = self.transform_img(img)
        mask_tensor = torch.tensor(np.array(mask_pil), dtype=torch.long)
        return img_tensor, mask_tensor
