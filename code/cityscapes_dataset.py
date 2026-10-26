## 自定义 PyTorch 数据集类，专门读取 Cityscapes 街景数据集，自动配对原图与像素标注，把多类别原始标签转为「人行道 / 背景」二分类掩码（人行道作为"视障可通行区域"的代理标签），同时完成图像标准化预处理，直接适配后续轻量分割网络训练

import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import torch

# Cityscapes人行道像素ID
SIDEWALK_ID = 8

class CityscapesSegDataset(Dataset):
    def __init__(self, root_dir, split="train", crop_size=(768, 1024)):
        self.img_dir = os.path.join(root_dir, "leftImg8bit", split)
        self.label_dir = os.path.join(root_dir, "gtFine", split)
        self.crop_h, self.crop_w = crop_size

        self.img_paths = []
        self.label_paths = []
        # 遍历城市子文件夹匹配原图与标注
        for city in os.listdir(self.img_dir):
            city_img = os.path.join(self.img_dir, city)
            city_label = os.path.join(self.label_dir, city)
            for img_name in os.listdir(city_img):
                img_path = os.path.join(city_img, img_name)
                label_name = img_name.replace("leftImg8bit", "gtFine_labelIds")
                label_path = os.path.join(city_label, label_name)
                self.img_paths.append(img_path)
                self.label_paths.append(label_path)

        # 图像标准化预处理
        self.transform_img = transforms.Compose([
            transforms.Resize((self.crop_h, self.crop_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        label = cv2.imread(self.label_paths[idx], cv2.IMREAD_GRAYSCALE)

        # 二分类掩码：人行道=1，其余背景=0
        binary_mask = np.zeros_like(label, dtype=np.uint8)
        binary_mask[label == SIDEWALK_ID] = 1
        mask_pil = Image.fromarray(binary_mask)
        mask_pil = mask_pil.resize((self.crop_w, self.crop_h), Image.NEAREST)

        img_tensor = self.transform_img(img)
        mask_tensor = torch.tensor(np.array(mask_pil), dtype=torch.long)
        return img_tensor, mask_tensor