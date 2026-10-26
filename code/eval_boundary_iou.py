"""
eval_boundary_iou.py
实验2：边界质量评估（Boundary IoU, BIoU）—— 针对性证明 BGFR 的边界精炼贡献。

使用条件：
  - 模型训练完成，weights/ 下有权重
  - 依赖 opencv（cv2）做形态学腐蚀

作用：
  1. 在 Cityscapes val 上计算每个模型的 mIoU 与两种口径的边界 IoU（BIoU）
  2. 对比 ENet / SBNet 两条骨干上的完整消融（基线 / +CA / +BA / +DBA / +L-ASPP / +CBR）
  3. 说明：mIoU 提升有限时，BIoU 提升更能体现 CBR 边界精炼的价值

【两种边界 IoU 口径】—— 论文中必须写清用的是哪一种
  官方口径（Cheng et al., "Boundary IoU", CVPR 2021，对应 boundary-iou-api）：
    边界带先对 mask 做 1 像素 copyMakeBorder 以闭合图像边缘轮廓，再腐蚀：
        B(M) = M - erode(pad(M), d)[1:h+1, 1:w+1]
        BIoU_c = |(G∩P) ∩ (B(G)∪B(P))| / |(G∪P) ∩ (B(G)∪B(P))|
    按“每张图先算、再对数据集求平均”的协议统计。
  旧口径（本项目早期实现，保留用于对照）：
        inter = |B(G)∩P| + |B(P)∩G|
        union = |B(G)| + |B(P)|
    且不做 copyMakeBorder，按“全局累积计数”统计。
  两种口径数值不等价，须在论文中明确标注，不可混用。

运行方式：
  conda activate blind_road
  cd code
  python eval_boundary_iou.py

输出：
  - ../train_logs/eval_boundary_iou.csv        : BIoU/mIoU 对比表（两种口径并列）
  - ../train_logs/eval_boundary_iou_chart.png  : 分组柱状图（mIoU vs 官方 BIoU vs 旧口径 BIoU）
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 避免 Anaconda 下 OpenMP 重复加载
import csv
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_common import (DEVICE, CROP_SIZE, NUM_CLASSES, DATA_ROOT,
                         build_model, resolve_weight_path, CityscapesBlindDataset,
                         model_forward, confusion_matrix,
                         mious_from_confusion, pixel_acc_from_confusion)

# ---------------- 配置 ----------------
OUTPUT_CSV = os.path.join(DATA_ROOT, "..", "..", "train_logs", "eval_boundary_iou.csv")
OUTPUT_CHART = os.path.join(DATA_ROOT, "..", "..", "train_logs", "eval_boundary_iou_chart.png")
DILATION_RATIO = 0.02        # 边界带宽度相对图像对角线的比例（论文常用 0.02）
TRAIN_FG_NAME = "可通行区域"

# 对比模型：两条骨干上的完整消融
COMPARE_MODELS = ["ENet", "ENet+CA", "ENet+BA", "ENet-DBA", "ENet+L-ASPP", "ENet-CBR",
                  "SBNet", "SBNet+CA", "SBNet+L-ASPP", "SBNet-CBR"]

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------- 官方口径 ----------------
def mask_to_boundary_official(mask_c, d):
    """
    官方 Boundary IoU 边界带：pad 1 像素后 mask - erode(mask, d)。
    参数 mask_c: (H,W) bool/uint8；返回 (H,W) bool。
    """
    m = mask_c.astype(np.uint8)
    h, w = m.shape
    # 官方实现：先补 1 像素边框以闭合图像边缘的轮廓
    padded = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(padded, kernel, iterations=d)
    eroded = eroded[1:h + 1, 1:w + 1]
    return (m - eroded) > 0


def boundary_iou_official(gt_b, pred_b, gt, pred):
    """单张图单类别的官方 Boundary IoU。gt_b/pred_b 为已算好的边界带。"""
    inter = int(np.sum((gt & pred) & (gt_b | pred_b)))
    union = int(np.sum((gt | pred) & (gt_b | pred_b)))
    return (inter / union) if union > 0 else None


# ---------------- 旧口径（保留对照） ----------------
def mask_to_boundary_legacy(mask_c, d):
    """旧口径：不补边框，直接 mask - erode(mask)。"""
    m = mask_c.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(m, kernel, iterations=d)
    return (m - eroded) > 0


def main():
    print("=" * 78)
    print("  实验2：边界质量评估（Boundary IoU）")
    print(f"  设备: {DEVICE}    扩张比例: {DILATION_RATIO}")
    print(f"  对比模型: {len(COMPARE_MODELS)} 个")
    print("=" * 78)

    val_dataset = CityscapesBlindDataset(DATA_ROOT, split="val")
    n_val = len(val_dataset)
    print(f"  验证集: {n_val} 张\n")

    H, W = CROP_SIZE
    d = max(1, int(round(DILATION_RATIO * np.sqrt(H ** 2 + W ** 2))))
    print(f"  边界带宽度 d = round({DILATION_RATIO} * sqrt({H}^2+{W}^2)) = {d} px\n")

    # ---- 预计算 GT（与模型无关，算一次复用 10 次，明显省时）----
    print(">>> 预计算 GT 边界带 ...")
    gts, gt_b_official, gt_b_legacy = [], [], []
    for i, (_, mask) in enumerate(val_dataset):
        gt = mask.numpy()
        gts.append(gt)
        gt_b_official.append([mask_to_boundary_official(gt == c, d) for c in range(NUM_CLASSES)])
        gt_b_legacy.append([mask_to_boundary_legacy(gt == c, d) for c in range(NUM_CLASSES)])
        if (i + 1) % 100 == 0:
            print(f"    ... {i + 1}/{n_val}")
    print("    GT 边界带完成\n")

    results = []
    for name in COMPARE_MODELS:
        print(f">>> {name} ...")
        wp = resolve_weight_path(name)
        if wp is None:
            print(f"  [跳过] 权重不存在")
            continue
        model, loaded = build_model(name)
        if not loaded:
            print(f"  [跳过] 权重加载失败")
            continue
        model.eval()

        conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        # 官方口径：每张图算一次 BIoU，最后求平均
        off_per_img = {c: [] for c in range(NUM_CLASSES)}
        # 旧口径：全局累积
        lg_inter = {c: 0 for c in range(NUM_CLASSES)}
        lg_union = {c: 0 for c in range(NUM_CLASSES)}

        with torch.no_grad():
            for idx, (img, _) in enumerate(val_dataset):
                out = model_forward(model, img.unsqueeze(0).to(DEVICE))
                pred = torch.argmax(out, dim=1)[0].cpu().numpy()
                gt = gts[idx]
                conf += confusion_matrix(torch.from_numpy(pred), torch.from_numpy(gt)).numpy()

                for c in range(NUM_CLASSES):
                    p_c = pred == c
                    g_c = gt == c
                    p_b_off = mask_to_boundary_official(p_c, d)
                    v = boundary_iou_official(gt_b_official[idx][c], p_b_off, g_c, p_c)
                    if v is not None:
                        off_per_img[c].append(v)

                    p_b_lg = mask_to_boundary_legacy(p_c, d)
                    lg_inter[c] += int(np.sum(gt_b_legacy[idx][c] & p_c) + np.sum(p_b_lg & g_c))
                    lg_union[c] += int(np.sum(gt_b_legacy[idx][c]) + np.sum(p_b_lg))

        per_class_iou, miou = mious_from_confusion(conf)
        pa = pixel_acc_from_confusion(conf)

        off_per_class = [(float(np.mean(off_per_img[c])) if off_per_img[c] else 0.0)
                         for c in range(NUM_CLASSES)]
        off_mean = float(np.mean(off_per_class))
        lg_per_class = [(lg_inter[c] / lg_union[c]) if lg_union[c] > 0 else 0.0
                        for c in range(NUM_CLASSES)]
        lg_mean = float(np.mean(lg_per_class))

        results.append({
            "model": name,
            "miou": round(miou * 100, 2),
            "pixel_acc": round(pa * 100, 2),
            "iou_bg": round(per_class_iou[0] * 100, 2),
            "iou_fg": round(per_class_iou[1] * 100, 2),
            "biou_official": round(off_mean * 100, 2),
            "biou_official_bg": round(off_per_class[0] * 100, 2),
            "biou_official_fg": round(off_per_class[1] * 100, 2),
            "biou_legacy": round(lg_mean * 100, 2),
            "biou_legacy_bg": round(lg_per_class[0] * 100, 2),
            "biou_legacy_fg": round(lg_per_class[1] * 100, 2),
            "d_px": d,
        })
        print(f"    mIoU={miou*100:.2f}%  官方BIoU={off_mean*100:.2f}%  旧口径BIoU={lg_mean*100:.2f}%  "
              f"({TRAIN_FG_NAME}类 IoU={per_class_iou[1]*100:.2f}%, 官方边界BIoU={off_per_class[1]*100:.2f}%)")
        del model
        torch.cuda.empty_cache()

    # ---- 保存 CSV ----
    fieldnames = ["model", "miou", "pixel_acc", "iou_bg", "iou_fg",
                  "biou_official", "biou_official_bg", "biou_official_fg",
                  "biou_legacy", "biou_legacy_bg", "biou_legacy_fg"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\n结果已保存: {OUTPUT_CSV}")

    # ---- 打印对比表 ----
    print("\n" + "=" * 100)
    print(f"  {'模型':<16s} {'mIoU(%)':<9s} {'官方BIoU':<10s} {'旧BIoU':<9s} "
          f"{TRAIN_FG_NAME+'IoU':<12s} {TRAIN_FG_NAME+'官方BIoU':<14s}")
    print("-" * 100)
    for r in results:
        print(f"  {r['model']:<16s} {r['miou']:<9.2f} {r['biou_official']:<10.2f} "
              f"{r['biou_legacy']:<9.2f} {r['iou_fg']:<12.2f} {r['biou_official_fg']:<14.2f}")
    print("=" * 100)

    # ---- 分组柱状图 ----
    x = np.arange(len(results))
    names = [r["model"] for r in results]
    mious = [r["miou"] for r in results]
    off = [r["biou_official"] for r in results]
    lg = [r["biou_legacy"] for r in results]

    fig, ax = plt.subplots(figsize=(15, 6))
    w = 0.27
    ax.bar(x - w, mious, w, label="mIoU (%)", color="#2196F3", edgecolor="black", linewidth=0.5)
    ax.bar(x, off, w, label="Boundary IoU (%) — 官方口径", color="#FF5722", edgecolor="black", linewidth=0.5)
    ax.bar(x + w, lg, w, label="Boundary IoU (%) — 旧口径（对照）", color="#9E9E9E",
           edgecolor="black", linewidth=0.5)
    for i in range(len(results)):
        ax.text(i - w, mious[i] + 0.4, f"{mious[i]:.1f}", ha="center", fontsize=7)
        ax.text(i, off[i] + 0.4, f"{off[i]:.1f}", ha="center", fontsize=7)
        ax.text(i + w, lg[i] + 0.4, f"{lg[i]:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("IoU (%)", fontsize=12)
    ax.set_title(f"mIoU vs Boundary IoU (d={d}px) — 证明 CBR 边界精炼贡献", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_CHART, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"图表已保存: {OUTPUT_CHART}")


if __name__ == "__main__":
    main()
