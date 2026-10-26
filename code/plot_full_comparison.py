"""
plot_full_comparison.py
全模型对比绘图脚本 —— 生成15个模型的综合对比图表。

使用条件：
  - 所有训练已完成，train_logs/ 目录下有全部 CSV 文件
  - benchmark_fps.py 已运行，train_logs/benchmark_fps.csv 已生成（可选，无FPS数据时只画mIoU/参数量）

作用：
  1. 从 train_logs/ 读取所有模型的训练 CSV，提取 Best mIoU
  2. 从 benchmark_fps.csv 读取 FPS/参数量/FLOPs（如果有）
  3. 生成4张论文级对比图：
     - 图1: mIoU 对比柱状图（按 mIoU 排序）
     - 图2: 参数量 vs mIoU 散点图
     - 图3: FPS vs mIoU 散点图（需 benchmark_fps.csv）
     - 图4: 训练曲线对比图（所有模型的 mIoU 随 epoch 变化）

运行方式：
  cd code
  conda activate blind_road
  python plot_full_comparison.py

输出：
  - ../train_logs/full_comparison_miou_bar.png
  - ../train_logs/full_comparison_params_scatter.png
  - ../train_logs/full_comparison_fps_scatter.png
  - ../train_logs/full_comparison_curves.png
"""

from project_paths import resolve_code_path
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 控制台中文/符号输出：Windows 默认 GBK，遇到 ✓ 等字符会抛 UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ============================================================
# 配置
# ============================================================
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
LOG_DIR = resolve_code_path("../train_logs")
OUTPUT_DIR = resolve_code_path("../train_logs")

# ============================================================
# 模型注册表：名称 → (CSV文件名, 参数量M, 颜色, 分组)
# ============================================================
MODEL_INFO = {
    # ---- 基线模型 ----
    "ENet":            {"csv": "enet_metrics.csv",              "params": 0.26,  "color": "#2196F3", "group": "baseline"},
    "Fast-SCNN":       {"csv": "fastscnn_metrics.csv",          "params": 1.47,  "color": "#4CAF50", "group": "baseline"},
    "SegFormer-B0":    {"csv": "segformer_metrics.csv",         "params": 4.30,  "color": "#9C27B0", "group": "baseline"},
    "SBNet":       {"csv": "train_log_100epoch_bin.csv",    "params": 0.95,  "color": "#FF5722", "group": "baseline"},
    "PP-LiteSeg-T":    {"csv": "train_log_ppliteseg_100epoch_bin.csv", "params": 2.71, "color": "#FF9800", "group": "baseline"},
    "STDC-Seg":        {"csv": "train_log_stdc_seg_100epoch_bin.csv",  "params": 2.25,  "color": "#795548", "group": "baseline"},
    # ---- SBNet 改进 ----
    "SBNet+CA":    {"csv": "train_log_ca_100epoch_bin.csv", "params": 0.95,  "color": "#FF7043", "group": "bisenetv2_ablation"},
    "SBNet+L-ASPP":{"csv": "bisenetv2_laspp_metrics.csv",  "params": 1.28,  "color": "#E64A19", "group": "bisenetv2_ablation"},
    "SBNet-CBR":   {"csv": "bisenetv2_cbr_metrics.csv",    "params": 1.32,  "color": "#BF360C", "group": "bisenetv2_ablation"},
    # ---- ENet 改进 ----
    "ENet+CA":         {"csv": "enet_ca_metrics.csv",           "params": 0.27,  "color": "#42A5F5", "group": "enet_ablation"},
    "ENet+BA":         {"csv": "enet_ba_metrics.csv",           "params": 0.27,  "color": "#64B5F6", "group": "enet_ablation"},
    "ENet-DBA":        {"csv": "enet_dba_metrics.csv",          "params": 0.28,  "color": "#90CAF9", "group": "enet_ablation"},
    "ENet+L-ASPP":     {"csv": "enet_laspp_metrics.csv",        "params": 0.35,  "color": "#1565C0", "group": "enet_ablation"},
    "ENet-CBR":        {"csv": "enet_cbr_metrics.csv",          "params": 0.36,  "color": "#0D47A1", "group": "enet_ablation"},
}


def read_best_miou(csv_path):
    """从 CSV 中读取最佳 mIoU"""
    if not os.path.exists(csv_path):
        return None, None, []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None, None, []

    # 兼容不同列名
    miou_key = "val_mIoU" if "val_mIoU" in reader.fieldnames else "val_miou"
    epoch_key = "epoch"

    best_miou = 0
    best_epoch = 0
    miou_curve = []
    for r in rows:
        try:
            miou = float(r[miou_key])
            miou_curve.append(miou)
            if miou > best_miou:
                best_miou = miou
                best_epoch = int(r[epoch_key])
        except (ValueError, KeyError):
            continue
    return best_miou, best_epoch, miou_curve


def read_fps_csv():
    """从 benchmark_fps.csv 读取 FPS 数据"""
    fps_path = os.path.join(LOG_DIR, "benchmark_fps.csv")
    fps_map = {}
    if not os.path.exists(fps_path):
        return fps_map
    with open(fps_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                fps_map[r["model"]] = {
                    "fps": float(r["fps"]),
                    "latency_ms": float(r["latency_ms"]),
                    "params_m": float(r["params_m"]),
                    "flops_g": float(r["flops_g"]) if r["flops_g"] != "N/A" else None,
                }
            except (ValueError, KeyError):
                continue
    return fps_map


def plot_miou_bar(results, save_path):
    """图1: mIoU 对比柱状图（按 mIoU 排序）"""
    sorted_results = sorted(results.items(), key=lambda x: x[1]["best_miou"] if x[1]["best_miou"] else 0)

    names = [r[0] for r in sorted_results]
    mious = [r[1]["best_miou"] * 100 if r[1]["best_miou"] else 0 for r in sorted_results]
    colors = [r[1]["color"] for r in sorted_results]

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.barh(range(len(names)), mious, color=colors, edgecolor="black", linewidth=0.5, height=0.7)

    # 标注 mIoU 值
    for i, (bar, miou) in enumerate(zip(bars, mious)):
        if miou > 0:
            ax.text(miou + 0.3, i, f"{miou:.2f}%", va="center", fontsize=9)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Best Val mIoU (%)", fontsize=12)
    ax.set_title("Blind Road Segmentation - Model Comparison", fontsize=14)
    ax.set_xlim(0, max(mious) + 5 if mious else 100)
    ax.grid(axis="x", alpha=0.3)

    # 分组图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", edgecolor="black", label="ENet 系列"),
        Patch(facecolor="#FF5722", edgecolor="black", label="SBNet 系列"),
        Patch(facecolor="#4CAF50", edgecolor="black", label="其他基线"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[保存] mIoU柱状图: {save_path}")


def plot_params_scatter(results, save_path):
    """图2: 参数量 vs mIoU 散点图"""
    fig, ax = plt.subplots(figsize=(12, 8))

    for name, info in results.items():
        if info["best_miou"] is None:
            continue
        params = info["params"]
        miou = info["best_miou"] * 100
        color = info["color"]
        size = max(params * 30, 50)
        marker = "D" if "CBR" in name or "L-ASPP" in name else "o"
        ax.scatter(params, miou, s=size, c=color, marker=marker,
                   edgecolors="black", linewidth=0.5, alpha=0.85, zorder=5)
        # 标注模型名称
        offset_y = 0.3 if "ENet" in name else -0.5
        ax.annotate(name, (params, miou), fontsize=7, ha="center", va="bottom",
                    xytext=(0, 8), textcoords="offset points")

    ax.set_xlabel("Parameters (M)", fontsize=12)
    ax.set_ylabel("Best Val mIoU (%)", fontsize=12)
    ax.set_title("Parameters vs. Accuracy", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 50)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[保存] 参数量散点图: {save_path}")


def plot_fps_scatter(results, fps_map, save_path):
    """图3: FPS vs mIoU 散点图"""
    if not fps_map:
        print("[跳过] FPS散点图: benchmark_fps.csv 不存在，请先运行 benchmark_fps.py")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    for name, info in results.items():
        if info["best_miou"] is None or name not in fps_map:
            continue
        fps = fps_map[name]["fps"]
        miou = info["best_miou"] * 100
        color = info["color"]
        size = max(info["params"] * 30, 50)
        ax.scatter(fps, miou, s=size, c=color, edgecolors="black", linewidth=0.5, alpha=0.85, zorder=5)
        ax.annotate(name, (fps, miou), fontsize=7, ha="center", va="bottom",
                    xytext=(0, 8), textcoords="offset points")

    ax.set_xlabel("Inference Speed (FPS) ↑", fontsize=12)
    ax.set_ylabel("Best Val mIoU (%) ↑", fontsize=12)
    ax.set_title("Speed vs. Accuracy (Bubble Size = Params)", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[保存] FPS散点图: {save_path}")


def plot_training_curves(results, save_path):
    """图4: 训练曲线对比图（mIoU vs Epoch）"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # 左图：ENet 系列
    enet_models = {k: v for k, v in results.items() if "ENet" in k and v["miou_curve"]}
    ax1 = axes[0]
    for name, info in enet_models.items():
        curve = info["miou_curve"]
        epochs = list(range(1, len(curve) + 1))
        miou_pct = [m * 100 for m in curve]
        ax1.plot(epochs, miou_pct, label=name, color=info["color"], linewidth=1.5, alpha=0.85)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Val mIoU (%)", fontsize=12)
    ax1.set_title("ENet Ablation Study", fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 右图：BiSeNetV2 系列 + 其他基线
    other_models = {k: v for k, v in results.items()
                    if ("SBNet" in k or k in ["Fast-SCNN", "PP-LiteSeg-T", "STDC-Seg", "SegFormer-B0"])
                    and v["miou_curve"]}
    ax2 = axes[1]
    for name, info in other_models.items():
        curve = info["miou_curve"]
        epochs = list(range(1, len(curve) + 1))
        miou_pct = [m * 100 for m in curve]
        style = "--" if "CBR" in name or "L-ASPP" in name or "+CA" in name else "-"
        lw = 2.5 if "CBR" in name else 1.5
        ax2.plot(epochs, miou_pct, label=name, color=info["color"],
                 linestyle=style, linewidth=lw, alpha=0.85)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Val mIoU (%)", fontsize=12)
    ax2.set_title("SBNet Ablation & Baselines", fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[保存] 训练曲线图: {save_path}")


def main():
    print("=" * 60)
    print("  全模型对比绘图")
    print(f"  日志目录: {LOG_DIR}")
    print("=" * 60)

    # 读取所有模型数据
    results = {}
    for name, info in MODEL_INFO.items():
        csv_path = os.path.join(LOG_DIR, info["csv"])
        best_miou, best_epoch, miou_curve = read_best_miou(csv_path)
        results[name] = {
            "best_miou": best_miou,
            "best_epoch": best_epoch,
            "miou_curve": miou_curve,
            "params": info["params"],
            "color": info["color"],
            "group": info["group"],
        }
        if best_miou is not None:
            print(f"  {name:<20s}  Best mIoU: {best_miou*100:.2f}% (Epoch {best_epoch})  "
                  f"Params: {info['params']:.2f}M  曲线长度: {len(miou_curve)}")
        else:
            print(f"  {name:<20s}  [未找到] {info['csv']}")

    # 读取 FPS 数据
    fps_map = read_fps_csv()

    # 生成4张图
    plot_miou_bar(results, os.path.join(OUTPUT_DIR, "full_comparison_miou_bar.png"))
    plot_params_scatter(results, os.path.join(OUTPUT_DIR, "full_comparison_params_scatter.png"))
    plot_fps_scatter(results, fps_map, os.path.join(OUTPUT_DIR, "full_comparison_fps_scatter.png"))
    plot_training_curves(results, os.path.join(OUTPUT_DIR, "full_comparison_curves.png"))

    # 打印汇总表格
    print("\n" + "=" * 90)
    print(f"  {'模型':<20s} {'参数量(M)':<12s} {'Best mIoU':<12s} {'Epoch':<8s} {'状态'}")
    print("-" * 90)
    for name, info in sorted(results.items(), key=lambda x: x[1]["best_miou"] if x[1]["best_miou"] else 0, reverse=True):
        miou_str = f"{info['best_miou']*100:.2f}%" if info["best_miou"] else "N/A"
        epoch_str = str(info["best_epoch"]) if info["best_epoch"] else "-"
        status = "✓" if info["best_miou"] else "✗ 待训练"
        print(f"  {name:<20s} {info['params']:<12.2f} {miou_str:<12s} {epoch_str:<8s} {status}")
    print("=" * 90)


if __name__ == "__main__":
    main()
