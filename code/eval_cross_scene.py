"""
eval_cross_scene.py
实验3：跨场景（跨城市）泛化分析 —— 每个城市单独算 mIoU，生成"城市 × 模型"热力图。

使用条件：
  - 模型训练完成，weights/ 下有权重
  - Cityscapes val 集包含多个城市（frankfurt / lindau / munster）

作用：
  1. 对 Cityscapes 验证集中的每个城市，分别评估各模型的 mIoU
  2. 生成"城市 × 模型"热力图，揭示哪些场景（城市）更难分割
  3. 用于论文实验章节的"跨场景泛化分析"，增加论文深度

说明：
  训练集与验证集是互不重叠的城市（train: aachen/bochum/bremen...；
  val: frankfurt/lindau/munster），因此本实验本质上是 cross-city 泛化测试。

运行方式：
  conda activate blind_road
  python code/eval_cross_scene.py

输出：
  - ../train_logs/eval_cross_scene.csv       : 城市 × 模型 mIoU 表
  - ../train_logs/eval_cross_scene_heatmap.png : 热力图
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 避免 Anaconda 下 OpenMP 重复加载
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_common import (DEVICE, DATA_ROOT, NUM_CLASSES, build_model,
                         resolve_weight_path, CityscapesBlindDataset, evaluate)

# ---------------- 配置 ----------------
OUTPUT_CSV = os.path.join(DATA_ROOT, "..", "..", "train_logs", "eval_cross_scene.csv")
OUTPUT_CHART = os.path.join(DATA_ROOT, "..", "..", "train_logs", "eval_cross_scene_heatmap.png")

# 对比模型：基线 + 改进
COMPARE_MODELS = ["ENet", "ENet+CA", "ENet+BA", "ENet-DBA", "ENet+L-ASPP", "ENet-CBR",
                 "SBNet", "SBNet+CA", "SBNet+L-ASPP", "SBNet-CBR"]

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def discover_val_cities():
    """发现 Cityscapes val 的所有城市目录。"""
    val_img_dir = os.path.join(DATA_ROOT, "leftImg8bit", "val")
    return sorted([d for d in os.listdir(val_img_dir) if os.path.isdir(os.path.join(val_img_dir, d))])


def main():
    cities = discover_val_cities()
    print("=" * 70)
    print("  实验3：跨城市泛化分析（Cityscapes val）")
    print(f"  设备: {DEVICE}    城市: {cities}")
    print("=" * 70)

    # 预构建所有模型（每模型只加载一次）
    print("\n加载模型 ...")
    models = {}
    for name in COMPARE_MODELS:
        model, loaded = build_model(name)
        if loaded:
            models[name] = model
            print(f"  {name}: 已加载")
        else:
            print(f"  {name}: 权重不存在，跳过 ({resolve_weight_path(name)})")

    # 逐城市评估
    matrix = {}  # name -> list of miou per city
    for city in cities:
        ds = CityscapesBlindDataset(DATA_ROOT, split="val", city=city)
        print(f"\n>>> 城市 {city} ({len(ds)} 张) ...")
        for name, model in models.items():
            miou, pa, _ = evaluate(model, ds)
            matrix.setdefault(name, []).append(round(miou * 100, 2))
            print(f"    {name:<16s} mIoU={miou*100:.2f}%  PixelAcc={pa*100:.2f}%")

    # ---- 保存 CSV ----
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + cities + ["mean"])
        for name in models:
            mean = sum(matrix[name]) / len(matrix[name])
            writer.writerow([name] + matrix[name] + [round(mean, 2)])
    print(f"\n结果已保存: {OUTPUT_CSV}")

    # ---- 热力图 ----
    names = list(models.keys())
    M = np.array([matrix[n] for n in names])  # (模型数, 城市数)
    fig, ax = plt.subplots(figsize=(len(cities) * 2.2 + 4, len(names) * 1.1 + 2))
    im = ax.imshow(M, cmap="RdYlGn", aspect="auto", vmin=np.min(M) - 2, vmax=np.max(M) + 2)
    ax.set_xticks(range(len(cities)))
    ax.set_xticklabels(cities, fontsize=11)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("City (scene)", fontsize=12)
    ax.set_title("Cross-City Generalization (Val mIoU %)", fontsize=13)
    for i in range(len(names)):
        for j in range(len(cities)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=10, color="black")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mIoU (%)", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_CHART, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"热力图已保存: {OUTPUT_CHART}")

    # ---- 打印摘要（每个模型的城市间方差，反映稳定性） ----
    print("\n" + "=" * 70)
    print(f"  {'模型':<16s} " + "".join(f"{c:<10s}" for c in cities) + f"{'均值':<8s}{'方差':<8s}")
    print("-" * 70)
    for n in names:
        row = matrix[n]
        mean = np.mean(row)
        var = np.var(row)
        print(f"  {n:<16s} " + "".join(f"{v:<10.2f}" for v in row) + f"{mean:<8.2f}{var:<8.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()