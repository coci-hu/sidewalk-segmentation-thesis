"""
eval_main_table.py
论文主表统一口径评估：用**全局混淆矩阵**口径评估全部模型，同时算出**逐图平均**
口径，把两者的差值显式记录下来，从而一次性解决"两种 mIoU 口径混用"的问题。

输出：
  ../train_logs/eval_main_table.csv
    model, params_m, flops_g, fps, best_epoch, miou_global, miou_per_image,
    miou_gap, pixel_acc, iou_bg, iou_fg, weight

口径说明（写入 CSV 的每行）：
  miou_global    = 先累积 500 张图的混淆矩阵，再算每类 IoU 后取平均  ← 论文主表推荐
  miou_per_image = 每张图先算 mIoU，再对 500 张取平均               ← 训练 CSV 的口径
  miou_gap       = miou_global - miou_per_image（论文中需说明的差值）

运行方式（在任意目录均可）：
  E:\\anaconda\\envs\\blind_road\\python.exe eval_main_table.py
"""

import os
import csv
import sys
import numpy as np
import torch

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import eval_common as ec
from eval_common import (MODEL_SPECS, DEVICE, DATA_ROOT, CROP_SIZE,
                         CityscapesBlindDataset, build_model, evaluate,
                         mious_from_confusion, pixel_acc_from_confusion,
                         resolve_weight_path, model_forward, confusion_matrix)

LOG_DIR = os.path.join(CODE_DIR, "..", "train_logs")
OUT_CSV = os.path.join(LOG_DIR, "eval_main_table.csv")

# 论文显示名 -> 训练 CSV（用于读取最优轮次）
TRAIN_CSV = {
    "ENet": "enet_metrics.csv",
    "ENet+CA": "enet_ca_metrics.csv",
    "ENet+BA": "enet_ba_metrics.csv",
    "ENet-DBA": "enet_dba_metrics.csv",
    "ENet+L-ASPP": "enet_laspp_metrics.csv",
    "ENet-CBR": "enet_cbr_metrics.csv",
    "SBNet": "train_log_100epoch_bin.csv",
    "SBNet+CA": "train_log_ca_100epoch_bin.csv",
    "SBNet+L-ASPP": "bisenetv2_laspp_metrics.csv",
    "SBNet-CBR": "bisenetv2_cbr_metrics.csv",
    "Fast-SCNN": "fastscnn_metrics.csv",
    "SegFormer-B0": "segformer_metrics.csv",
    "STDC-Seg": "train_log_stdc_seg_100epoch_bin.csv",
    "PP-LiteSeg-T": "train_log_ppliteseg_100epoch_bin.csv",
    # DeepLabv3+ 训练完成后自动纳入（权重存在即评估）
    "DeepLabv3+": "deeplab_metrics.csv",
}

# DeepLabv3+ 未注册在 eval_common 里（它是训练中新增的），在此补充。
# 权重不存在时会被自动跳过，因此本脚本可在训练完成前后反复运行。
EXTRA_SPECS = {
    "DeepLabv3+": {
        "build": lambda: __import__("model_deeplabv3plus").get_deeplabv3plus_2class(DEVICE),
        "weight": "best_deeplab_100epoch_bin.pth",
    },
}


def read_fps_table():
    """从 benchmark_fps.csv 读取参数量 / FLOPs / FPS。"""
    path = os.path.join(LOG_DIR, "benchmark_fps.csv")
    table = {}
    if not os.path.exists(path):
        return table
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            table[r["model"]] = {
                "params_m": r.get("params_m", "N/A"),
                "flops_g": r.get("flops_g", "N/A"),
                "fps": r.get("fps", "N/A"),
            }
    return table


def read_best_epoch(model_name):
    """从训练 CSV 读取最优轮次（用于论文表格的"最优轮"列）。"""
    fn = TRAIN_CSV.get(model_name)
    if not fn:
        return "N/A"
    path = os.path.join(LOG_DIR, fn)
    if not os.path.exists(path):
        return "N/A"
    best_v, best_ep = -1.0, "N/A"
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            key = "val_mIoU" if "val_mIoU" in r else "val_miou"
            try:
                v = float(r[key]); ep = int(r["epoch"])
            except (KeyError, TypeError, ValueError):
                continue
            if v > best_v:
                best_v, best_ep = v, ep
    return best_ep


def eval_one(model, dataset):
    """
    返回 (miou_global, miou_per_image, pixel_acc, iou_bg, iou_fg)。
    一次遍历同时算出两种口径，避免重复推理。
    """
    model.eval()
    conf_total = torch.zeros(ec.NUM_CLASSES, ec.NUM_CLASSES, dtype=torch.long, device=DEVICE)
    per_image_mious = []
    with torch.no_grad():
        for img, mask in dataset:
            img = img.unsqueeze(0).to(DEVICE)
            mask = mask.to(DEVICE)
            pred = torch.argmax(model_forward(model, img), dim=1)[0]
            c = confusion_matrix(pred, mask)
            conf_total += c
            # 逐图口径：单张图的 mIoU
            _, mi = mious_from_confusion(c.cpu().numpy())
            per_image_mious.append(mi)

    conf = conf_total.cpu().numpy()
    ious, miou_global = mious_from_confusion(conf)
    pa = pixel_acc_from_confusion(conf)
    miou_per_image = float(np.mean(per_image_mious)) if per_image_mious else 0.0
    return miou_global, miou_per_image, pa, ious[0], ious[1]


def main():
    # 把 DeepLabv3+ 注入注册表（权重不存在则本脚本跳过它）
    for k, v in EXTRA_SPECS.items():
        MODEL_SPECS.setdefault(k, v)

    print(f"设备: {DEVICE}")
    print(f"验证集: {DATA_ROOT}  split=val  crop={CROP_SIZE}")
    dataset = CityscapesBlindDataset(DATA_ROOT, split="val", crop_size=CROP_SIZE)
    print(f"验证集图片数: {len(dataset)}\n")

    fps_table = read_fps_table()
    rows = []
    for name in MODEL_SPECS:
        wpath = resolve_weight_path(name)
        if wpath is None:
            print(f"[跳过] {name:<14s} 权重不存在")
            continue
        print(f"[评估] {name:<14s} ...", end="", flush=True)
        model, loaded = build_model(name)
        if not loaded:
            print(" 权重加载失败，跳过")
            continue
        miou_g, miou_p, pa, iou_bg, iou_fg = eval_one(model, dataset)
        meta = fps_table.get(name, {})
        rows.append({
            "model": name,
            "params_m": meta.get("params_m", "N/A"),
            "flops_g": meta.get("flops_g", "N/A"),
            "fps": meta.get("fps", "N/A"),
            "best_epoch": read_best_epoch(name),
            "miou_global": round(miou_g * 100, 2),
            "miou_per_image": round(miou_p * 100, 2),
            "miou_gap": round((miou_g - miou_p) * 100, 2),
            "pixel_acc": round(pa * 100, 2),
            "iou_bg": round(iou_bg * 100, 2),
            "iou_fg": round(iou_fg * 100, 2),
            "weight": os.path.basename(wpath),
        })
        print(f" 全局={miou_g*100:.2f}%  逐图={miou_p*100:.2f}%  差={100*(miou_g-miou_p):+.2f}pp")

    fields = ["model", "params_m", "flops_g", "fps", "best_epoch", "miou_global",
              "miou_per_image", "miou_gap", "pixel_acc", "iou_bg", "iou_fg", "weight"]
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 104)
    print(f"  {'模型':<14s}{'参数(M)':>9s}{'最优轮':>7s}{'全局mIoU':>10s}{'逐图mIoU':>10s}{'差(pp)':>8s}{'像素精度':>9s}{'前景IoU':>9s}")
    print("-" * 104)
    for r in sorted(rows, key=lambda x: -x["miou_global"]):
        print(f"  {r['model']:<14s}{str(r['params_m']):>9s}{str(r['best_epoch']):>7s}"
              f"{r['miou_global']:>10.2f}{r['miou_per_image']:>10.2f}{r['miou_gap']:>+8.2f}"
              f"{r['pixel_acc']:>9.2f}{r['iou_fg']:>9.2f}")
    print("=" * 104)
    print(f"\n结果已保存: {OUT_CSV}")
    print("论文主表请引用 miou_global 列（全局混淆矩阵 + 最优轮），并在表注写明口径与最优轮次。")


if __name__ == "__main__":
    main()
