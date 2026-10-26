"""
eval_common.py
评估 / 部署实验的公共工具模块（被 deploy_benchmark / eval_boundary_iou /
eval_cross_scene / visualize_attention 四个脚本 import）。

统一提供：
  1. 模型注册表 + 权重加载（兼容 model_state / state_dict / 纯 dict，strict=False）
  2. Cityscapes 二分类数据集（支持按城市过滤，用于跨场景实验）
  3. 统一推理接口（兼容返回元组的 CBR/BA/DBA 模型）
  4. 混淆矩阵 + mIoU / PixelAcc 指标

约定：脚本在 code/ 目录外任意位置运行均可，路径基于本文件绝对路径推导。
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 避免 Anaconda 下 OpenMP 重复加载
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# 确保本目录（code/）在 sys.path，以便 import model_* 与 components 包
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

# ---------------- 全局配置 ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CROP_SIZE = (768, 1024)          # 与训练一致的输入尺寸 (高, 宽)
SIDEWALK_ID = 8                  # Cityscapes sidewalk(id=8)；
                                 # 本文以人行道作为"视障可通行区域"的代理标签（数据集无盲道类别）
NUM_CLASSES = 2                  # 0=背景, 1=可通行区域

# 基于本文件绝对路径推导项目根目录，避免依赖运行 cwd
WEIGHT_DIR = os.path.join(CODE_DIR, "..", "weights")
DATA_ROOT = os.path.join(CODE_DIR, "..", "dataset", "cityscapes")

# ---------------- 模型注册表 ----------------
# 命名约定（详见 code/NAMING_AND_LABELS.md）：
#   SBNet = 精简双分支网络（Slim Bilateral Network），是自写的精简双分支结构，
#           并非官方 BiSeNetV2（官方含 ARM/CE/BGA，参数量为 M 量级）。
#           权重文件名沿用历史命名 bisenetv2_*，仅文件名未改，不改变任何结构。
# 每个条目: {"build": 构建函数, "weight": 权重文件名}
MODEL_SPECS = {
    # ---- ENet 系列（参数量 0.26M ~ 0.36M）----
    "ENet": {
        "build": lambda: __import__("model_enet").get_enet_2class(DEVICE),
        "weight": "enet_best.pth",
    },
    "ENet+CA": {
        "build": lambda: __import__("model_enet_ca").get_enet_ca_2class(DEVICE),
        "weight": "enet_ca_best.pth",
    },
    "ENet+BA": {
        "build": lambda: __import__("model_enet_ba").get_enet_ba_2class(DEVICE),
        "weight": "enet_ba_best.pth",
    },
    "ENet-DBA": {
        "build": lambda: __import__("model_enet_dba").get_enet_dba_2class(DEVICE),
        "weight": "enet_dba_best.pth",
    },
    "ENet+L-ASPP": {
        "build": lambda: __import__("model_enet_laspp").ENetLASSP(num_classes=2).to(DEVICE),
        "weight": "enet_laspp_best.pth",
    },
    "ENet-CBR": {
        "build": lambda: __import__("model_enet_cbr").ENetCBR(num_classes=2).to(DEVICE),
        "weight": "enet_cbr_best.pth",
    },
    # ---- SBNet 系列（精简双分支网络，参数量 0.94M ~ 1.32M）----
    "SBNet": {
        "build": lambda: __import__("model_bisenetv2").get_bisenetv2_2class(DEVICE),
        "weight": "best_bisenet_100epoch_bin.pth",
    },
    "SBNet+CA": {
        "build": lambda: __import__("model_bisenetv2_ca").get_bisenetv2_ca_2class(DEVICE),
        "weight": "best_bisenet_ca_100epoch_bin.pth",
    },
    "SBNet+L-ASPP": {
        "build": lambda: __import__("model_bisenetv2_laspp").BiSeNetV2LASSP(num_classes=2).to(DEVICE),
        "weight": "bisenetv2_laspp_best.pth",
    },
    "SBNet-CBR": {
        "build": lambda: __import__("model_bisenetv2_cbr").BiSeNetV2CBR(num_classes=2).to(DEVICE),
        "weight": "bisenetv2_cbr_best.pth",
    },
    # ---- 其他轻量基线 ----
    "Fast-SCNN": {
        "build": lambda: __import__("model_fastscnn").get_fastscnn_2class(DEVICE),
        "weight": "fastscnn_best.pth",
    },
    "SegFormer-B0": {
        "build": lambda: __import__("model_segformer").get_segformer_b0_2class(DEVICE),
        "weight": "segformer_best.pth",
    },
    "STDC-Seg": {
        "build": lambda: __import__("model_stdc_seg").get_stdc_seg_2class(DEVICE),
        "weight": "best_stdc_seg_100epoch_bin.pth",
    },
    "PP-LiteSeg-T": {
        "build": lambda: __import__("model_ppliteseg").get_ppliteseg_2class(DEVICE),
        "weight": "best_ppliteseg_100epoch_bin.pth",
    },
    # 说明：U-Net 已从论文中移除。原因：batch=2、768×1024 时峰值显存 9030 MiB，
    # 超过本机 8 GB 显存上限（触发内存交换，单 epoch 需 170 分钟），无法完成 100 轮
    # 公平对比训练；其现有权重 best_unet_100epoch_bin.pth 实测 val mIoU 仅 66.0%，
    # 属未充分训练的中间产物，不可与已收敛模型同表比较。详见导师审阅报告。
}


def resolve_weight_path(name):
    """返回模型权重的绝对路径（不存在则返回 None）。"""
    if name not in MODEL_SPECS:
        return None
    p = os.path.join(WEIGHT_DIR, MODEL_SPECS[name]["weight"])
    return p if os.path.exists(p) else None


def load_weight(model, weight_path):
    """加载权重，兼容不同存储格式。成功返回 True。"""
    if weight_path is None or not os.path.exists(weight_path):
        return False
    ckpt = torch.load(weight_path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=False)
    return True


def build_model(name):
    """构建模型并加载权重，返回 (model, loaded_bool)。"""
    spec = MODEL_SPECS[name]
    model = spec["build"]()
    model.eval()
    loaded = load_weight(model, resolve_weight_path(name))
    return model, loaded


def model_forward(model, x):
    """统一前向推理，兼容返回元组的模型（CBR/BA/DBA eval 时返回 (seg_out,)）。"""
    out = model(x)
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


# ---------------- Cityscapes 二分类数据集 ----------------
class CityscapesBlindDataset(Dataset):
    """
    Cityscapes 二分类数据集（可通行区域=1 / 背景=0），支持按城市过滤。

    参数:
        root:      Cityscapes 根目录（含 leftImg8bit/ 与 gtFine/）
        split:     "train" 或 "val"
        crop_size: 统一 resize 到的 (高, 宽)
        city:      城市名（如 "frankfurt"），None 表示使用全部城市
    """
    def __init__(self, root, split="val", crop_size=CROP_SIZE, city=None):
        self.crop_h, self.crop_w = crop_size
        self.img_dir = os.path.join(root, "leftImg8bit", split)
        self.label_dir = os.path.join(root, "gtFine", split)

        cities = [city] if city else sorted(os.listdir(self.img_dir))
        self.img_paths = []
        self.label_paths = []
        for c in cities:
            c_img_dir = os.path.join(self.img_dir, c)
            c_label_dir = os.path.join(self.label_dir, c)
            if not os.path.isdir(c_img_dir):
                continue
            for img_name in sorted(os.listdir(c_img_dir)):
                if not img_name.endswith(".png"):
                    continue
                label_name = img_name.replace("leftImg8bit", "gtFine_labelIds")
                label_path = os.path.join(c_label_dir, label_name)
                if os.path.exists(label_path):
                    self.img_paths.append(os.path.join(c_img_dir, img_name))
                    self.label_paths.append(label_path)

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
        label = np.array(Image.open(self.label_paths[idx]))
        binary = np.zeros_like(label, dtype=np.uint8)
        binary[label == SIDEWALK_ID] = 1
        mask = Image.fromarray(binary).resize((self.crop_w, self.crop_h), Image.NEAREST)
        img_tensor = self.transform_img(img)
        mask_tensor = torch.tensor(np.array(mask), dtype=torch.long)
        return img_tensor, mask_tensor


# ---------------- 指标 ----------------
def confusion_matrix(pred, gt, num_classes=NUM_CLASSES):
    """计算单张图的混淆矩阵 (num_classes, num_classes)。pred/gt 为 (H, W) torch.long。"""
    mask = (gt >= 0) & (gt < num_classes)
    idx = num_classes * gt[mask] + pred[mask]
    count = torch.bincount(idx, minlength=num_classes * num_classes)
    return count.reshape(num_classes, num_classes)


def mious_from_confusion(confusion):
    """从累积混淆矩阵计算每类 IoU 与 mIoU。"""
    ious = []
    for c in range(confusion.shape[0]):
        inter = confusion[c, c]
        union = confusion[c, :].sum() + confusion[:, c].sum() - inter
        ious.append(inter / union if union > 0 else 0.0)
    return ious, float(np.mean(ious))


def pixel_acc_from_confusion(confusion):
    """从累积混淆矩阵计算像素精度。"""
    correct = np.trace(confusion)
    total = confusion.sum()
    return correct / total if total > 0 else 0.0


def evaluate(model, dataset, device=DEVICE, verbose=False):
    """
    在给定数据集上评估模型，返回 (mIoU, pixel_acc, confusion)：
      - mIoU:      平均 IoU
      - pixel_acc: 像素精度
      - confusion: 累积混淆矩阵 numpy (num_classes, num_classes)
    """
    model.eval()
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    with torch.no_grad():
        for i, (img, mask) in enumerate(dataset):
            img = img.unsqueeze(0).to(device)
            out = model_forward(model, img)
            pred = torch.argmax(out, dim=1)[0]
            conf += confusion_matrix(pred, mask.to(device))
            if verbose and (i + 1) % 20 == 0:
                print(f"    ... {i + 1}/{len(dataset)}")
    conf = conf.cpu().numpy()
    ious, miou = mious_from_confusion(conf)
    pa = pixel_acc_from_confusion(conf)
    return miou, pa, conf