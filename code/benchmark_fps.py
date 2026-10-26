"""
benchmark_fps.py
推理速度测试脚本 —— 统一测试所有模型的 FPS、参数量、FLOPs。

使用条件：
  - 所有模型训练完成后运行
  - 需要GPU环境（CPU测试结果不具备论文参考价值）
  - 需要安装 thop 库：pip install thop

作用：
  1. 统一输入尺寸 (768, 1024)，测试每个模型的推理速度 (FPS)
  2. 统计参数量 (Params) 和计算量 (FLOPs)
  3. 结果保存为 CSV 和对比图表

运行方式：
  cd code
  conda activate blind_road
  python benchmark_fps.py

输出：
  - ../train_logs/benchmark_fps.csv     : 速度/参数量/FLOPs 数据表
  - ../train_logs/benchmark_fps_chart.png : 速度-精度散点图
"""

from project_paths import resolve_code_path
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 避免 Anaconda 下 OpenMP 重复加载
import sys
import time
import csv
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")          # 必须在 pyplot 之前设置：无界面后端，避免绘图时加载 GUI 运行时
import matplotlib.pyplot as plt

# ============================================================
# 配置
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = (1, 3, 768, 1024)       # 统一输入尺寸（与训练一致）
WARMUP_ITERS = 50                     # GPU 预热迭代次数
BENCH_ITERS = 200                     # 正式测试迭代次数
OUTPUT_CSV = resolve_code_path("../train_logs/benchmark_fps.csv")
OUTPUT_CHART = resolve_code_path("../train_logs/benchmark_fps_chart.png")

# ============================================================
# 模型注册表：名称 → (构建函数, 权重路径, 是否有边界损失)
# ============================================================
MODEL_REGISTRY = {
    # ---- 基线模型 ----
    "ENet": {
        "build_fn": lambda: __import__("model_enet").get_enet_2class(DEVICE),
        "weight": resolve_code_path("../weights/enet_best.pth"),
        "has_bdy_loss": False,
    },
    "Fast-SCNN": {
        "build_fn": lambda: __import__("model_fastscnn").get_fastscnn_2class(DEVICE),
        "weight": resolve_code_path("../weights/fastscnn_best.pth"),
        "has_bdy_loss": False,
    },
    "SegFormer-B0": {
        "build_fn": lambda: __import__("model_segformer").get_segformer_b0_2class(DEVICE),
        "weight": resolve_code_path("../weights/segformer_best.pth"),
        "has_bdy_loss": False,
    },
    "SBNet": {
        "build_fn": lambda: __import__("model_bisenetv2").get_bisenetv2_2class(DEVICE),
        "weight": resolve_code_path("../weights/best_bisenet_100epoch_bin.pth"),
        "has_bdy_loss": False,
    },
    "SBNet+CA": {
        "build_fn": lambda: __import__("model_bisenetv2_ca").get_bisenetv2_ca_2class(DEVICE),
        "weight": resolve_code_path("../weights/best_bisenet_ca_100epoch_bin.pth"),
        "has_bdy_loss": False,
    },
    "PP-LiteSeg-T": {
        "build_fn": lambda: __import__("model_ppliteseg").get_ppliteseg_2class(DEVICE),
        "weight": resolve_code_path("../weights/best_ppliteseg_100epoch_bin.pth"),
        "has_bdy_loss": False,
    },
    "STDC-Seg": {
        "build_fn": lambda: __import__("model_stdc_seg").get_stdc_seg_2class(DEVICE),
        "weight": resolve_code_path("../weights/best_stdc_seg_100epoch_bin.pth"),
        "has_bdy_loss": False,
    },
    # ---- ENet 消融实验 ----
    "ENet+CA": {
        "build_fn": lambda: __import__("model_enet_ca").get_enet_ca_2class(DEVICE),
        "weight": resolve_code_path("../weights/enet_ca_best.pth"),
        "has_bdy_loss": False,
    },
    "ENet+BA": {
        "build_fn": lambda: __import__("model_enet_ba").get_enet_ba_2class(DEVICE),
        "weight": resolve_code_path("../weights/enet_ba_best.pth"),
        "has_bdy_loss": False,
    },
    "ENet-DBA": {
        "build_fn": lambda: __import__("model_enet_dba").get_enet_dba_2class(DEVICE),
        "weight": resolve_code_path("../weights/enet_dba_best.pth"),
        "has_bdy_loss": False,
    },
    "ENet+L-ASPP": {
        "build_fn": lambda: __import__("model_enet_laspp").ENetLASSP(num_classes=2).to(DEVICE),
        "weight": resolve_code_path("../weights/enet_laspp_best.pth"),
        "has_bdy_loss": False,
    },
    "ENet-CBR": {
        "build_fn": lambda: __import__("model_enet_cbr").ENetCBR(num_classes=2).to(DEVICE),
        "weight": resolve_code_path("../weights/enet_cbr_best.pth"),
        "has_bdy_loss": False,
    },
    # ---- BiSeNetV2 消融实验 ----
    "SBNet+L-ASPP": {
        "build_fn": lambda: __import__("model_bisenetv2_laspp").BiSeNetV2LASSP(num_classes=2).to(DEVICE),
        "weight": resolve_code_path("../weights/bisenetv2_laspp_best.pth"),
        "has_bdy_loss": False,
    },
    "SBNet-CBR": {
        "build_fn": lambda: __import__("model_bisenetv2_cbr").BiSeNetV2CBR(num_classes=2).to(DEVICE),
        "weight": resolve_code_path("../weights/bisenetv2_cbr_best.pth"),
        "has_bdy_loss": False,
    },
}


def count_params(model):
    """统计模型参数量（M）"""
    return sum(p.numel() for p in model.parameters()) / 1e6


def _count_flops_thop(model, dummy):
    """thop 口径：统计数据集的乘加次数（MAC），业界论文的 GFLOPs 惯例"""
    from thop import profile
    flops, _ = profile(model, inputs=(dummy,), verbose=False)
    return flops / 1e9


def _count_flops_torch(model, dummy):
    """torch 内置口径：乘、加各计一次运算，数值约为 MAC 口径的 1.4~2 倍"""
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc:
        _model_forward(model, dummy)
    return fc.get_total_flops() / 1e9


def count_flops(model, input_size):
    """
    统计模型 FLOPs（G）。
    主口径：thop（乘加计 1 次，与文献 GFLOPs 惯例一致）；
    备用口径：torch.utils.flop_counter（thop 失败时才使用，数值不可与主口径混用）。
    """
    dummy = torch.randn(*input_size).to(DEVICE)
    model.eval()
    with torch.no_grad():
        try:
            val = _count_flops_thop(model, dummy)
            print(f"  FLOPs:  {val:.2f}G (thop/MAC 口径)")
            return val
        except Exception as e:
            print(f"  [警告] thop 统计失败: {type(e).__name__}: {e}")
        try:
            val = _count_flops_torch(model, dummy)
            print(f"  FLOPs:  {val:.2f}G (torch 内置口径，仅作参考)")
            return val
        except Exception as e:
            print(f"  [警告] FLOPs 统计全部失败: {type(e).__name__}: {e}")
            return -1.0


def _model_forward(model, x):
    """统一前向推理，兼容返回元组的模型（CBR/BA/DBA 等 eval 时返回 (seg_out,)）"""
    output = model(x)
    if isinstance(output, tuple):
        return output[0]
    return output


def benchmark_fps(model, input_size, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """
    测试模型推理速度 (FPS)。
    流程：GPU预热 → 正式测速 → 取平均值
    """
    model.eval()
    dummy = torch.randn(*input_size).to(DEVICE)

    # GPU 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = _model_forward(model, dummy)
    torch.cuda.synchronize()

    # 正式测速
    with torch.no_grad():
        start = time.time()
        for _ in range(iters):
            _ = _model_forward(model, dummy)
        torch.cuda.synchronize()
        elapsed = time.time() - start

    fps = iters / elapsed
    latency = elapsed / iters * 1000  # ms
    return fps, latency


def load_weight(model, weight_path, has_bdy_loss=False):
    """加载权重，兼容不同格式"""
    if not os.path.exists(weight_path):
        print(f"  [跳过] 权重不存在: {weight_path}")
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


def plot_fps_chart(results, save_path):
    """
    绘制速度-精度散点图（FPS vs mIoU）+ 参数量气泡图。
    需要先跑完全部训练，从 CSV 中读取 mIoU。
    """
    # 尝试从 CSV 读取 mIoU
    miou_map = {}
    csv_files = {
        "ENet": "enet_metrics.csv",
        "Fast-SCNN": "fastscnn_metrics.csv",
        "SegFormer-B0": "segformer_metrics.csv",
        "SBNet": "train_log_100epoch_bin.csv",
        "SBNet+CA": "train_log_ca_100epoch_bin.csv",
        "PP-LiteSeg-T": "train_log_ppliteseg_100epoch_bin.csv",
        "STDC-Seg": "train_log_stdc_seg_100epoch_bin.csv",
        "ENet+CA": "enet_ca_metrics.csv",
        "ENet+BA": "enet_ba_metrics.csv",
        "ENet-DBA": "enet_dba_metrics.csv",
        "ENet+L-ASPP": "enet_laspp_metrics.csv",
        "ENet-CBR": "enet_cbr_metrics.csv",
        "SBNet+L-ASPP": "bisenetv2_laspp_metrics.csv",
        "SBNet-CBR": "bisenetv2_cbr_metrics.csv",
    }
    for name, fname in csv_files.items():
        fpath = os.path.join(resolve_code_path("../train_logs"), fname)
        if os.path.exists(fpath):
            with open(fpath, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    miou_key = "val_mIoU" if "val_mIoU" in reader.fieldnames else "val_miou"
                    best_miou = max(float(r[miou_key]) for r in rows if r[miou_key])
                    miou_map[name] = best_miou * 100  # 转百分比

    # 绘图
    fig, ax = plt.subplots(figsize=(14, 8))

    # 颜色分组
    color_map = {
        "ENet": "#2196F3", "ENet+CA": "#42A5F5", "ENet+BA": "#64B5F6",
        "ENet-DBA": "#90CAF9", "ENet+L-ASPP": "#1565C0", "ENet-CBR": "#0D47A1",
        "SBNet": "#FF5722", "SBNet+CA": "#FF7043",
        "SBNet+L-ASPP": "#E64A19", "SBNet-CBR": "#BF360C",
        "Fast-SCNN": "#4CAF50", "SegFormer-B0": "#9C27B0",
        "PP-LiteSeg-T": "#FF9800", "STDC-Seg": "#795548",
    }

    skipped = []
    for r in results:
        name = r["model"]
        fps = r["fps"]
        params = r["params_m"]
        flops = r["flops_g"]
        miou = miou_map.get(name, None)

        color = color_map.get(name, "#999999")
        size = max(params * 80, 30)  # 气泡大小与参数量成正比

        if miou is None:
            # 没有训练指标 CSV 的模型不画点（否则会落在 y=0，污染散点图）
            skipped.append(name)
            continue
        ax.scatter(fps, miou, s=size, c=color, alpha=0.8, edgecolors="black", linewidth=0.5, zorder=5)
        ax.annotate(name, (fps, miou), fontsize=7, ha="center", va="bottom",
                    xytext=(0, 8), textcoords="offset points")

    ax.set_xlabel("Inference Speed (FPS) ↑", fontsize=12)
    ax.set_ylabel("Best Val mIoU (%) ↑", fontsize=12)
    ax.set_title("Speed-Accuracy Trade-off (Bubble Size = Params)", fontsize=14)
    ax.grid(True, alpha=0.3)
    if miou_map:
        ax.set_ylim(max(60, min(miou_map.values()) - 5), max(miou_map.values()) + 3)
    if skipped:
        print(f"  [提示] 无训练指标 CSV，未画入散点图: {', '.join(skipped)}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"\n速度-精度图已保存: {save_path}")


def main():
    print("=" * 70)
    print("  推理速度测试 (Benchmark FPS)")
    print(f"  设备: {DEVICE}")
    print(f"  输入尺寸: {INPUT_SIZE}")
    print(f"  预热: {WARMUP_ITERS} iters, 测试: {BENCH_ITERS} iters")
    print("=" * 70)

    if DEVICE.type == "cpu":
        print("\n[警告] 当前使用 CPU，FPS 结果不具备论文参考价值！建议在 GPU 环境运行。")

    results = []
    for name, cfg in MODEL_REGISTRY.items():
        print(f"\n>>> 测试 {name} ...")
        try:
            model = cfg["build_fn"]()
            model.eval()

            # 加载权重
            loaded = load_weight(model, cfg["weight"], cfg["has_bdy_loss"])
            if not loaded:
                print(f"  [跳过] {name} - 权重未找到")

            # 参数量
            params_m = count_params(model)
            print(f"  参数量: {params_m:.4f}M")

            # FLOPs（count_flops 内部已打印口径）
            flops_g = count_flops(model, INPUT_SIZE)

            # FPS
            fps, latency = benchmark_fps(model, INPUT_SIZE)
            print(f"  FPS:    {fps:.1f}")
            print(f"  延迟:   {latency:.2f}ms")

            results.append({
                "model": name,
                "params_m": round(params_m, 4),
                "flops_g": round(flops_g, 2) if flops_g > 0 else "N/A",
                "fps": round(fps, 1),
                "latency_ms": round(latency, 2),
            })

            # 释放GPU显存
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  [错误] {name}: {e}")
            results.append({
                "model": name,
                "params_m": "ERROR",
                "flops_g": "ERROR",
                "fps": "ERROR",
                "latency_ms": "ERROR",
            })

    # 保存 CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "params_m", "flops_g", "fps", "latency_ms"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n结果已保存: {OUTPUT_CSV}")

    # 打印结果表格
    print("\n" + "=" * 70)
    print(f"  {'模型':<20s} {'参数量(M)':<12s} {'FLOPs(G)':<12s} {'FPS':<10s} {'延迟(ms)':<10s}")
    print("-" * 70)
    for r in results:
        print(f"  {r['model']:<20s} {str(r['params_m']):<12s} {str(r['flops_g']):<12s} "
              f"{str(r['fps']):<10s} {str(r['latency_ms']):<10s}")
    print("=" * 70)

    # 绘制速度-精度图
    try:
        plot_fps_chart(results, OUTPUT_CHART)
    except Exception as e:
        print(f"[警告] 绘图失败: {e}")


if __name__ == "__main__":
    main()
