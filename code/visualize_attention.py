"""
visualize_attention.py
实验4：注意力可视化 —— 提取 CA 的方向注意力图，证明 CA 确实学到方向信息。

使用条件：
  - ENet-CBR / BiSeNetV2-CBR 训练完成，weights/ 下有权重
  - 依赖 matplotlib + opencv（已随项目安装）

作用：
  1. 通过 forward hook 提取 CoordAtt 内部的方向注意力权重：
       a_h (水平方向) 与 a_w (垂直方向)
  2. 可视化原图 + 空间注意力热力图 + 方向权重（横条/竖条）
  3. 对比说明：ENet 基线无 CA 模块，ENet-CBR 的 CA 方向注意力被激活，
     用于回应"CA 单独使用无效"的质疑（CBR 中 L-ASPP 提供的上下文让 CA 更有效）

关键说明：
  - CA 的 a_h/a_w 是学习到的方向权重，可视化它们能直观展示网络
    对水平/垂直方向（可通行区域通常是沿视线的纵向结构）的响应。

运行方式：
  conda activate blind_road
  python code/visualize_attention.py                 # 默认取 val 第 0 张图
  python code/visualize_attention.py --index 100     # 指定第 100 张图

输出：
  - ../train_logs/attention_<模型名>.png   ：原图 + 空间注意力热力图 + 方向权重
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 避免 Anaconda 下 OpenMP 重复加载
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

from eval_common import (DEVICE, CROP_SIZE, DATA_ROOT, build_model,
                         resolve_weight_path, CityscapesBlindDataset, model_forward)

OUTPUT_DIR = os.path.join(DATA_ROOT, "..", "..", "train_logs")
VIS_MODELS = ["ENet-CBR", "SBNet-CBR"]   # 注意：只有含 CoordAtt 的模型才能可视化（SBNet+L-ASPP 无 CA 模块）

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def extract_ca_attn(ca_module, feat):
    """
    复现 CoordAtt 前向的前 7 步，单独取出方向注意力权重。

    参数:
        ca_module: CoordAtt 模块实例
        feat:      CA 的输入特征 (1, C, H, W)
    返回:
        a_h: 水平方向注意力 (1, C, 1, W)
        a_w: 垂直方向注意力 (1, C, H, 1)
    """
    n, c, h, w = feat.size()
    x_h = feat.mean(dim=2, keepdim=True)                # (1, C, 1, W)
    x_w = feat.mean(dim=3, keepdim=True).permute(0, 1, 3, 2)  # (1, C, 1, H)
    y = torch.cat([x_h, x_w], dim=3)                    # (1, C, 1, W+H)
    y = ca_module.act(ca_module.bn1(ca_module.conv1(y)))
    x_h2, x_w2 = torch.split(y, [w, h], dim=3)
    x_w2 = x_w2.permute(0, 1, 3, 2)                     # (1, C, H, 1)
    a_h = ca_module.conv_h(x_h2).sigmoid()              # (1, C, 1, W)
    a_w = ca_module.conv_w(x_w2).sigmoid()              # (1, C, H, 1)
    return a_h, a_w


def make_ca_hook(ca_module, captured):
    """返回一个 forward hook，用于捕获 CA 的方向注意力。"""
    def hook(module, inp, out):
        with torch.no_grad():
            a_h, a_w = extract_ca_attn(ca_module, inp[0])
            captured["a_h"] = a_h.detach().cpu()
            captured["a_w"] = a_w.detach().cpu()
    return hook


def load_raw_image(img_path, crop_size):
    """读取原始图片并 resize 到 crop_size，返回 RGB numpy (H, W, 3)。"""
    img = Image.open(img_path).convert("RGB").resize((crop_size[1], crop_size[0]))
    return np.array(img)


def visualize(model_name, model, img_path, img_tensor, out_dir):
    """可视化单模型的 CA 方向注意力。"""
    captured = {}
    handle = model.ca.register_forward_hook(make_ca_hook(model.ca, captured))
    with torch.no_grad():
        _ = model_forward(model, img_tensor.unsqueeze(0).to(DEVICE))
    handle.remove()

    if "a_h" not in captured:
        print(f"  [警告] {model_name} 未能捕获 CA 注意力（模块名为 .ca ？）")
        return

    a_h = captured["a_h"][0]          # (C, 1, W')
    a_w = captured["a_w"][0]          # (C, H', 1)

    # 对通道求均值 → 方向权重向量
    h_weight = a_h.mean(dim=0)[0].numpy()           # (W',)
    w_weight = a_w.mean(dim=0)[:, 0].numpy()        # (H',)

    # 空间注意力图：外积 (H', W')
    spatial = w_weight[:, None] * h_weight[None, :]

    # 归一化到原图尺寸
    H, W = CROP_SIZE
    h_w = np.interp(np.linspace(0, 1, W), np.linspace(0, 1, len(h_weight)), h_weight)
    w_w = np.interp(np.linspace(0, 1, H), np.linspace(0, 1, len(w_weight)), w_weight)
    spatial_full = w_w[:, None] * h_w[None, :]

    orig = load_raw_image(img_path, CROP_SIZE)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    # (1) 原图
    axes[0].imshow(orig)
    axes[0].set_title("Input Image", fontsize=13)
    axes[0].axis("off")

    # (2) 空间注意力热力图
    im = axes[1].imshow(spatial_full, cmap="jet", alpha=0.9)
    axes[1].set_title("CA Spatial Attention (h x w)", fontsize=13)
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # (3) 方向权重：上方竖条（垂直方向），下方横条（水平方向）
    ax3 = axes[2]
    ax3_top = ax3.inset_axes([0.0, 0.55, 1.0, 0.40])
    ax3_top.imshow(w_w[:, None].T, cmap="jet", aspect="auto")
    ax3_top.set_title("Vertical (a_w)", fontsize=11)
    ax3_top.set_xticks([]); ax3_top.set_yticks([])
    ax3_bot = ax3.inset_axes([0.0, 0.05, 1.0, 0.40])
    ax3_bot.imshow(h_w[None, :], cmap="jet", aspect="auto")
    ax3_bot.set_title("Horizontal (a_h)", fontsize=11)
    ax3_bot.set_xticks([]); ax3_bot.set_yticks([])
    ax3.axis("off")

    fig.suptitle(f"{model_name} — Coordinate Attention Direction Weights", fontsize=14)
    save_path = os.path.join(out_dir, f"attention_{model_name}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [保存] {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="val 集图片索引")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("  实验4：CA 方向注意力可视化")
    print(f"  设备: {DEVICE}    图片索引: {args.index}")
    print("=" * 70)

    val_ds = CityscapesBlindDataset(DATA_ROOT, split="val")
    if args.index >= len(val_ds):
        print(f"索引超出范围（共 {len(val_ds)} 张）")
        return
    img_tensor, _ = val_ds[args.index]
    img_path = val_ds.img_paths[args.index]
    print(f"  图片: {os.path.basename(img_path)}\n")

    for name in VIS_MODELS:
        model, loaded = build_model(name)
        if not loaded:
            print(f"  [跳过] {name} 权重不存在: {resolve_weight_path(name)}")
            continue
        if not hasattr(model, "ca"):
            print(f"  [跳过] {name} 不含 CA 模块（CoordAtt），无法做方向注意力可视化")
            continue
        print(f">>> {name} ...")
        visualize(name, model, img_path, img_tensor, OUTPUT_DIR)

    print("\n说明：ENet 基线不含 CA 模块，作为'无方向注意力'的对照；")
    print("      ENet-CBR 的 CA 在 L-ASPP 上下文基础上学习方向权重。")


if __name__ == "__main__":
    main()