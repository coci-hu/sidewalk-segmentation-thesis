"""
visualize_prediction.py
可视化推理脚本 —— 输入图片，输出模型分割结果对比图。

使用条件：
  - 至少有一个模型已训练完成并保存了权重
  - 需要 GPU 或 CPU 均可运行（CPU 较慢但可用）
  - 输入图片可以是验证集图片或自定义图片

作用：
  1. 对指定图片（或验证集随机图片）进行推理
  2. 将原图、真值、预测结果并排展示，方便论文使用
  3. 支持单模型推理和多模型对比推理
  4. 支持批量生成验证集可视化结果

运行方式：
  # 单模型推理（默认 ENet-CBR）
  cd code
  conda activate blind_road
  python visualize_prediction.py

  # 指定模型
  python visualize_prediction.py --model enet_cbr

  # 指定输入图片
  python visualize_prediction.py --model enet_cbr --input ../dataset/cityscapes/leftImg8bit/val/frankfurt_000000_000294_leftImg8bit.png

  # 多模型对比（生成一行多列对比图）
  python visualize_prediction.py --compare enet,enet_cbr,bisenetv2_cbr

  # 批量生成验证集可视化（随机选 N 张）
  python visualize_prediction.py --batch 10 --model enet_cbr

  # 指定输出目录
  python visualize_prediction.py --model enet_cbr --output ../vis_results

输出：
  - 单模型: {output_dir}/{model_name}_{image_name}.png
  - 多模型对比: {output_dir}/compare_{models}_{image_name}.png
"""

from project_paths import resolve_code_path
import os
import sys
import argparse
import random

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms

# ============================================================
# 颜色映射：背景=黑色，可通行区域=黄色
# ============================================================
COLOR_MAP = np.array([
    [0, 0, 0],        # 类别0: 背景 → 黑色
    [255, 255, 0],     # 类别1: 可通行区域 → 黄色
], dtype=np.uint8)


# ============================================================
# 模型注册表
# ============================================================
def build_model(name, device):
    """根据名称构建模型并加载权重"""
    registry = {
        "enet": ("model_enet", "get_enet_2class", resolve_code_path("../weights/enet_best.pth"), False),
        "enet_ca": ("model_enet_ca", "get_enet_ca_2class", resolve_code_path("../weights/enet_ca_best.pth"), False),
        "enet_ba": ("model_enet_ba", "get_enet_ba_2class", resolve_code_path("../weights/enet_ba_best.pth"), False),
        "enet_dba": ("model_enet_dba", "get_enet_dba_2class", resolve_code_path("../weights/enet_dba_best.pth"), False),
        "enet_laspp": ("model_enet_laspp", "get_enet_laspp_2class", resolve_code_path("../weights/enet_laspp_best.pth"), False),
        "enet_cbr": ("model_enet_cbr", "get_enet_cbr_2class", resolve_code_path("../weights/enet_cbr_best.pth"), False),
        "fastscnn": ("model_fastscnn", "get_fastscnn_2class", resolve_code_path("../weights/fastscnn_best.pth"), False),
        "segformer": ("model_segformer", "get_segformer_b0_2class", resolve_code_path("../weights/segformer_best.pth"), False),
        "bisenetv2": ("model_bisenetv2", "get_bisenetv2_2class", resolve_code_path("../weights/best_bisenet_100epoch_bin.pth"), False),
        "bisenetv2_ca": ("model_bisenetv2_ca", "get_bisenetv2_ca_2class", resolve_code_path("../weights/best_bisenet_ca_100epoch_bin.pth"), False),
        "bisenetv2_laspp": ("model_bisenetv2_laspp", "get_bisenetv2_laspp_2class", resolve_code_path("../weights/bisenetv2_laspp_best.pth"), False),
        "bisenetv2_cbr": ("model_bisenetv2_cbr", "get_bisenetv2_cbr_2class", resolve_code_path("../weights/bisenetv2_cbr_best.pth"), False),
        "ppliteseg": ("model_ppliteseg", "get_ppliteseg_2class", resolve_code_path("../weights/best_ppliteseg_100epoch_bin.pth"), False),
        "stdc_seg": ("model_stdc_seg", "get_stdc_seg_2class", resolve_code_path("../weights/best_stdc_seg_100epoch_bin.pth"), False),
        "unet": ("model_unet", "get_unet_2class", resolve_code_path("../weights/best_unet_100epoch_bin.pth"), False),
    }
    if name not in registry:
        raise ValueError(f"未知模型: {name}，可选: {list(registry.keys())}")

    mod_name, fn_name, weight_path, _ = registry[name]
    mod = __import__(mod_name)
    build_fn = getattr(mod, fn_name)
    model = build_fn(device)

    # 加载权重
    if os.path.exists(weight_path):
        ckpt = torch.load(weight_path, map_location=device, weights_only=False)
        state_dict = ckpt if not isinstance(ckpt, dict) else ckpt.get("model_state", ckpt.get("state_dict", ckpt))
        model.load_state_dict(state_dict, strict=False)
        print(f"  [加载] {name} 权重: {weight_path}")
    else:
        print(f"  [警告] 权重不存在: {weight_path}，使用随机权重")

    model.eval()
    return model


def preprocess_image(image_path, crop_size=(768, 1024)):
    """
    预处理图片：resize + normalize（与训练时一致）。
    返回: tensor (1,3,H,W), 原始 PIL Image
    """
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((crop_size[1], crop_size[0]), Image.BILINEAR)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0

    # ImageNet 标准化（与训练时一致）
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = (img_np - mean) / std

    # HWC → CHW
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float().unsqueeze(0)
    return img_tensor, img


def predict(model, img_tensor, device):
    """单张图片推理，返回预测掩码 (H, W) numpy"""
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        output = model(img_tensor)
        # 处理返回元组的情况（CBR 等模型推理时返回 (seg_out,)）
        if isinstance(output, tuple):
            output = output[0]
        pred = output.argmax(dim=1).squeeze(0).cpu().numpy()
    return pred


def mask_to_color(mask):
    """将类别掩码转为 RGB 彩色图"""
    return COLOR_MAP[mask]


def load_gt_mask(gt_path, crop_size=(768, 1024)):
    """加载真值掩码并 resize"""
    gt = Image.open(gt_path).convert("L")
    gt = gt.resize((crop_size[1], crop_size[0]), Image.NEAREST)
    return np.array(gt, dtype=np.int64)


def find_val_images(dataset_root, n=None):
    """
    从验证集中找到 (image, gt_mask) 配对。
    数据集结构: dataset_root/leftImg8bit/val/ 和 dataset_root/gtFine/val/
    """
    img_dir = os.path.join(dataset_root, "leftImg8bit", "val")
    gt_dir = os.path.join(dataset_root, "gtFine", "val")

    pairs = []
    if not os.path.exists(img_dir):
        print(f"[错误] 验证集图片目录不存在: {img_dir}")
        return pairs

    for subfolder in sorted(os.listdir(img_dir)):
        sub_img_dir = os.path.join(img_dir, subfolder)
        sub_gt_dir = os.path.join(gt_dir, subfolder)
        if not os.path.isdir(sub_img_dir):
            continue
        for fname in sorted(os.listdir(sub_img_dir)):
            if not fname.endswith("_leftImg8bit.png"):
                continue
            img_path = os.path.join(sub_img_dir, fname)
            gt_name = fname.replace("_leftImg8bit.png", "_gtFine_labelTrainIds.png")
            gt_path = os.path.join(sub_gt_dir, gt_name)
            if os.path.exists(gt_path):
                pairs.append((img_path, gt_path))

    if n and n < len(pairs):
        pairs = random.sample(pairs, n)
    return pairs


def visualize_single(model_name, img_path, gt_path, output_dir, device, crop_size=(768, 1024)):
    """单模型可视化：原图 | 真值 | 预测"""
    model = build_model(model_name, device)
    img_tensor, orig_img = preprocess_image(img_path, crop_size)
    pred = predict(model, img_tensor, device)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 原图
    axes[0].imshow(orig_img.resize((crop_size[1], crop_size[0])))
    axes[0].set_title("Input Image", fontsize=14)
    axes[0].axis("off")

    # 真值
    if gt_path and os.path.exists(gt_path):
        gt = load_gt_mask(gt_path, crop_size)
        axes[1].imshow(mask_to_color(gt))
        axes[1].set_title("Ground Truth", fontsize=14)
    else:
        axes[1].text(0.5, 0.5, "No GT", ha="center", va="center", fontsize=16)
        axes[1].set_title("Ground Truth (N/A)", fontsize=14)
    axes[1].axis("off")

    # 预测
    axes[2].imshow(mask_to_color(pred))
    axes[2].set_title(f"Prediction ({model_name})", fontsize=14)
    axes[2].axis("off")

    plt.tight_layout()
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(output_dir, f"{model_name}_{img_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [保存] {save_path}")

    del model
    torch.cuda.empty_cache()


def visualize_compare(model_names, img_path, gt_path, output_dir, device, crop_size=(768, 1024)):
    """多模型对比可视化：原图 | 真值 | 模型1预测 | 模型2预测 | ..."""
    img_tensor, orig_img = preprocess_image(img_path, crop_size)
    n_cols = 2 + len(model_names)  # 原图 + 真值 + N个模型
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))

    # 原图
    axes[0].imshow(orig_img.resize((crop_size[1], crop_size[0])))
    axes[0].set_title("Input", fontsize=14)
    axes[0].axis("off")

    # 真值
    if gt_path and os.path.exists(gt_path):
        gt = load_gt_mask(gt_path, crop_size)
        axes[1].imshow(mask_to_color(gt))
        axes[1].set_title("Ground Truth", fontsize=14)
    else:
        axes[1].text(0.5, 0.5, "No GT", ha="center", va="center")
        axes[1].set_title("GT (N/A)", fontsize=14)
    axes[1].axis("off")

    # 各模型预测
    for i, name in enumerate(model_names):
        model = build_model(name, device)
        pred = predict(model, img_tensor, device)
        axes[2 + i].imshow(mask_to_color(pred))
        axes[2 + i].set_title(name, fontsize=14)
        axes[2 + i].axis("off")
        del model
        torch.cuda.empty_cache()

    plt.tight_layout()
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    compare_str = "vs_".join(model_names)
    save_path = os.path.join(output_dir, f"compare_{compare_str}_{img_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [保存] {save_path}")


def main():
    parser = argparse.ArgumentParser(description="可通行区域分割可视化推理脚本")
    parser.add_argument("--model", type=str, default="enet_cbr",
                        help="单模型推理的模型名称 (默认: enet_cbr)")
    parser.add_argument("--compare", type=str, default=None,
                        help="多模型对比，逗号分隔 (例: enet,enet_cbr,bisenetv2_cbr)")
    parser.add_argument("--input", type=str, default=None,
                        help="输入图片路径 (默认: 从验证集随机选取)")
    parser.add_argument("--gt", type=str, default=None,
                        help="真值掩码路径 (默认: 自动从数据集目录查找)")
    parser.add_argument("--dataset", type=str, default=resolve_code_path("../dataset/cityscapes"),
                        help="数据集根目录 (默认: ../dataset/cityscapes)")
    parser.add_argument("--batch", type=int, default=None,
                        help="批量生成N张验证集可视化 (默认: None，仅生成1张)")
    parser.add_argument("--output", type=str, default=resolve_code_path("../vis_results"),
                        help="输出目录 (默认: ../vis_results)")
    parser.add_argument("--crop_size", type=int, nargs=2, default=[768, 1024],
                        help="输入尺寸 H W (默认: 768 1024)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crop_size = tuple(args.crop_size)
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  可通行区域分割可视化推理")
    print(f"  设备: {device}")
    print(f"  输入尺寸: {crop_size}")
    print(f"  输出目录: {args.output}")
    print("=" * 60)

    # 确定输入图片列表
    if args.input:
        pairs = [(args.input, args.gt)]
    else:
        n = args.batch if args.batch else 1
        pairs = find_val_images(args.dataset, n=n)
        if not pairs:
            print("[错误] 未找到验证集图片，请指定 --input 参数")
            return

    # 执行可视化
    for img_path, gt_path in pairs:
        print(f"\n处理: {os.path.basename(img_path)}")
        if args.compare:
            model_names = [m.strip() for m in args.compare.split(",")]
            visualize_compare(model_names, img_path, gt_path, args.output, device, crop_size)
        else:
            visualize_single(args.model, img_path, gt_path, args.output, device, crop_size)

    print(f"\n完成！结果保存在: {args.output}")


if __name__ == "__main__":
    main()
