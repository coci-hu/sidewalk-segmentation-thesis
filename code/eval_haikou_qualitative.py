"""
eval_haikou_qualitative.py
实验5：海口实景零样本定性验证（Cityscapes 训练 → 海口实景直接推理）

背景与定位：
  海口自建数据只有 23 张原图、没有像素标注，不足以训练或做定量 IoU 对比。
  但"跨域测试"本身不需要训练集和真值标注，因此这 23 张非常适合用来回答一个
  论文必须回答的问题：在 Cityscapes 上训练的轻量模型，换到真实城市实景后
  会退化到什么程度、退化是否体现在"可通行区域的形状"上。

本脚本产出（全部写入 train_logs/）：
  1. haikou_zero_shot_<模型名>.png  —— 每个模型一张 23 图总览（原图叠加预测可通行区域）
  2. haikou_zero_shot_scene_<场景>.png —— 每个场景一张图，横向对比全部模型
  3. haikou_zero_shot_stats.csv      —— 无需真值的统计量：可通行区域预测占比、
                                         平均最大 softmax 置信度（跨域置信度下降的代理指标）

运行方式：
  cd code
  python eval_haikou_qualitative.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

from eval_common import (DEVICE, CROP_SIZE, CODE_DIR, build_model, model_forward)

# ===================== 配置 =====================
HAIKOU_ROOT = os.path.join(CODE_DIR, "..", "dataset", "haikou_blind", "raw_all")
OUTPUT_DIR = os.path.join(CODE_DIR, "..", "train_logs")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# 对比模型：ENet 骨干与 SBNet 骨干各取基线与完整方法
MODELS = ["ENet", "ENet-CBR", "SBNet", "SBNet-CBR"]

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

_norm = T.Compose([
    T.Resize((CROP_SIZE[0], CROP_SIZE[1])),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def collect_images():
    """返回 [(场景名, 图片绝对路径), ...]，按场景名与文件名排序。"""
    items = []
    if not os.path.isdir(HAIKOU_ROOT):
        raise SystemExit(f"海口数据目录不存在: {HAIKOU_ROOT}")
    for scene in sorted(os.listdir(HAIKOU_ROOT)):
        scene_dir = os.path.join(HAIKOU_ROOT, scene)
        if not os.path.isdir(scene_dir):
            continue
        for fn in sorted(os.listdir(scene_dir)):
            if fn.endswith(IMG_EXTS):
                items.append((scene, os.path.join(scene_dir, fn)))
    return items


def load_image(path):
    """返回 (原图 uint8 (H,W,3), 网络输入张量 (3,H,W))"""
    img = Image.open(path).convert("RGB")
    raw = np.array(img.resize((CROP_SIZE[1], CROP_SIZE[0])))
    tensor = _norm(img)
    return raw, tensor


def infer(model, tensor):
    """返回 (前景概率图 (H,W) float, 二值掩码 (H,W) bool)"""
    with torch.no_grad():
        out = model_forward(model, tensor.unsqueeze(0).to(DEVICE))
        prob = torch.softmax(out, dim=1)[0, 1].cpu().numpy()   # 可通行区域类概率
    return prob, prob > 0.5


def overlay(raw, mask, alpha=0.45):
    """在原图上叠加半透明红色表示预测的可通行区域。"""
    vis = raw.astype(np.float32).copy()
    color = np.array([255.0, 40.0, 40.0])
    vis[mask] = (1 - alpha) * vis[mask] + alpha * color
    return vis.astype(np.uint8)


def save_grid(images, titles, save_path, ncols=5, figsize_scale=(3.2, 2.6)):
    """把若干张图排成网格保存。"""
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_scale[0] * ncols, figsize_scale[1] * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for i, (im, t) in enumerate(zip(images, titles)):
        axes[i].imshow(im)
        axes[i].set_title(t, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [保存] {save_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    items = collect_images()
    scenes = sorted({s for s, _ in items})
    print("=" * 78)
    print("  实验5：海口实景零样本定性验证")
    print(f"  设备: {DEVICE}    图片: {len(items)} 张 / {len(scenes)} 个场景")
    print(f"  场景: {', '.join(scenes)}")
    print(f"  说明: 无真值标注，仅做定性可视化与无需真值的统计")
    print("=" * 78)

    # ---- 逐图逐模型推理 ----
    raws, tensors, names = [], [], []
    for scene, path in items:
        raw, tensor = load_image(path)
        raws.append(raw)
        tensors.append(tensor)
        names.append((scene, os.path.basename(path)))

    stats = []          # (model, scene, n, fg_ratio_mean, conf_mean)
    pred_masks = {}     # model -> list[bool HxW]

    for name in MODELS:
        model, loaded = build_model(name)
        if not loaded:
            print(f"  [跳过] {name} 权重不存在")
            continue
        print(f"\n>>> 推理 {name} ...")
        masks, fg_ratios, confs = [], [], []
        for tensor in tensors:
            prob, mask = infer(model, tensor)
            masks.append(mask)
            fg_ratios.append(float(mask.mean()))
            confs.append(float(prob.max()) if prob.size else 0.0)
        pred_masks[name] = masks

        for scene in scenes:
            idx = [i for i, (s, _) in enumerate(names) if s == scene]
            stats.append({
                "model": name,
                "scene": scene,
                "n_images": len(idx),
                "fg_ratio_mean": round(float(np.mean([fg_ratios[i] for i in idx])), 4),
                "conf_mean": round(float(np.mean([confs[i] for i in idx])), 4),
            })

        # 该模型的海口全图总览
        overlays = [overlay(raws[i], masks[i]) for i in range(len(items))]
        titles = [f"{s}/{n}" for s, n in names]
        save_grid(overlays, titles,
                  os.path.join(OUTPUT_DIR, f"haikou_zero_shot_{name.replace('+', '_')}.png"))

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # ---- 逐场景横向对比（全部模型）----
    print("\n>>> 生成逐场景横向对比图 ...")
    for scene in scenes:
        idx = [i for i, (s, _) in enumerate(names) if s == scene]
        col = []
        for i in idx:
            col.append(raws[i])
            for name in MODELS:
                if name in pred_masks:
                    col.append(overlay(raws[i], pred_masks[name][i]))
        title_col = []
        for i in idx:
            title_col.append(f"{os.path.basename(names[i][1])}\n(Input)")
            for name in MODELS:
                if name in pred_masks:
                    title_col.append(name)
        save_grid(col, title_col,
                  os.path.join(OUTPUT_DIR, f"haikou_zero_shot_scene_{scene}.png"),
                  ncols=len(MODELS) + 1, figsize_scale=(2.6, 2.2))

    # ---- 统计 CSV ----
    csv_path = os.path.join(OUTPUT_DIR, "haikou_zero_shot_stats.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "scene", "n_images",
                                               "fg_ratio_mean", "conf_mean"])
        writer.writeheader()
        writer.writerows(stats)
    print(f"\n  [保存] {csv_path}")

    # ---- 控制台汇总 ----
    print("\n" + "=" * 78)
    print(f"  {'模型':<12s} {'可通行区域预测占比':<22s} {'平均最大置信度':<16s}")
    print("-" * 78)
    for name in MODELS:
        rows = [r for r in stats if r["model"] == name]
        if not rows:
            continue
        print(f"  {name:<12s} {np.mean([r['fg_ratio_mean'] for r in rows]):<22.4f} "
              f"{np.mean([r['conf_mean'] for r in rows]):<16.4f}")
    print("=" * 78)
    print("说明：本实验无真值标注，故不给出 IoU。可通行区域预测占比过低通常意味着")
    print("      模型把海口实景误判为背景，平均最大置信度下降则反映跨域置信度退化。")


if __name__ == "__main__":
    main()
