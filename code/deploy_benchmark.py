"""
deploy_benchmark.py
实验1：部署量化闭环 —— ONNX 导出 → ONNX Runtime FP32 / INT8 推理 → 精度与速度对比。

使用条件：
  - 已安装 onnx + onnxruntime-gpu（pip install onnx onnxruntime-gpu）
  - 目标模型已完成训练，weights/ 下有权重文件
  - GPU 环境（速度对比需 GPU；纯 CPU 也会跑，但 FPS 不具备论文参考价值）

作用：
  1. 将 PyTorch 模型导出为 ONNX
  2. 用 ONNX Runtime 跑 FP32 与 INT8（动态量化）推理
  3. 在 Cityscapes 验证集上对比 PyTorch FP32 / ORT FP32 / ORT INT8 的 mIoU
  4. 对比三者的推理速度（FPS / 延迟）
  5. 结果保存 CSV，供论文"工程化与部署"章节直接引用

运行方式（在 code 目录或项目根目录均可）：
  conda activate blind_road
  python code/deploy_benchmark.py

输出：
  - ../train_logs/deploy_benchmark.csv    : 量化前后精度/速度对比表
  - ../train_logs/deploy/*.onnx           : 导出的 ONNX 模型
  - ../train_logs/deploy/*_int8.onnx      : INT8 量化模型

说明（INT8 方案）：
  本脚本采用 onnxruntime 动态量化（quantize_dynamic，仅量化权重）。
  如需更小的精度损失，可改用静态量化（quantize_static + 校准数据集），
  但静态量化需要额外校准流程，此处先以动态量化完成"量化前后精度对比"的闭环。
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 避免 Anaconda 下 OpenMP 重复加载
import time
import csv
import numpy as np
import torch
import torch.nn as nn

from eval_common import (DEVICE, CROP_SIZE, NUM_CLASSES, MODEL_SPECS,
                         WEIGHT_DIR, DATA_ROOT, resolve_weight_path, load_weight,
                         build_model, model_forward, CityscapesBlindDataset,
                         confusion_matrix, mious_from_confusion, pixel_acc_from_confusion)

# ---------------- 配置 ----------------
INPUT_SIZE = (1, 3, CROP_SIZE[0], CROP_SIZE[1])
ONNX_DIR = os.path.join(WEIGHT_DIR, "..", "train_logs", "deploy")
OUTPUT_CSV = os.path.join(WEIGHT_DIR, "..", "train_logs", "deploy_benchmark.csv")
OPSET = 17
WARMUP = 20        # 预热迭代次数
ITERS = 100        # 正式测速迭代次数

# 参与部署对比的模型（基线与核心方法）
DEPLOY_MODELS = ["ENet", "ENet+L-ASPP", "ENet-CBR",
                 "SBNet", "SBNet+L-ASPP", "SBNet-CBR"]


# ---------------- ONNX 导出 ----------------
class ExportWrapper(nn.Module):
    """包装模型前向，把 tuple 输出（CBR 系列）统一为单 tensor 输出。"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


def export_onnx(model, onnx_path):
    """导出 ONNX，成功返回 True。"""
    model.eval()
    wrapper = ExportWrapper(model)
    dummy = torch.randn(*INPUT_SIZE).to(DEVICE)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, onnx_path,
            export_params=True, opset_version=OPSET,
            input_names=["input"], output_names=["output"],
            do_constant_folding=True,
        )
    return True


# ---------------- ONNX Runtime 推理与量化 ----------------
def make_session(onnx_path, cpu_only=False):
    import onnxruntime as ort
    if cpu_only:
        # 动态 INT8 量化（ConvInteger/MatMulInteger）是 CPU 优化型算子，
        # CUDA/TensorRT provider 不支持，必须用 CPU provider 跑。
        providers = ["CPUExecutionProvider"]
    else:
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if DEVICE.type == "cuda" else ["CPUExecutionProvider"])
    return ort.InferenceSession(onnx_path, providers=providers)


def quantize_int8(onnx_path, int8_path):
    """动态量化（仅量化 Conv/MatMul 权重）。成功返回 True。

    注意：先 onnx.load 成 ModelProto 再传入 quantize_dynamic，避免其内部
    C++ shape inference 对含中文的项目路径报错（infer_shapes_path 写 -inferred.onnx 失败）。
    """
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
    model = onnx.load(onnx_path)
    quantize_dynamic(
        model, int8_path,
        # QUInt8：生成 QLinearConv 而非 ConvInteger，避免 ORT 1.19 CPU provider
        # 报 "Could not find an implementation for ConvInteger" (NOT_IMPLEMENTED)。
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=["Conv", "MatMul"],
    )
    return True


def ort_infer_once(session, inp_np):
    """ORT 单次推理，返回 logits (1, C, H, W) numpy。"""
    return session.run(None, {"input": inp_np})[0]


# ---------------- 速度测试 ----------------
def measure_fps_pt(model, iters=ITERS, warmup=WARMUP):
    model.eval()
    dummy = torch.randn(*INPUT_SIZE).to(DEVICE)
    with torch.no_grad():
        for _ in range(warmup):
            model_forward(model, dummy)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            model_forward(model, dummy)
        torch.cuda.synchronize()
        elapsed = time.time() - start
    return iters / elapsed, elapsed / iters * 1000


def measure_fps_ort(session, iters=ITERS, warmup=WARMUP):
    inp = np.random.randn(*INPUT_SIZE).astype(np.float32)
    for _ in range(warmup):
        ort_infer_once(session, inp)
    start = time.time()
    for _ in range(iters):
        ort_infer_once(session, inp)
    elapsed = time.time() - start
    return iters / elapsed, elapsed / iters * 1000


# ---------------- 精度评估（PyTorch / ORT 通用） ----------------
def evaluate_pt(model, dataset):
    model.eval()
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for img, mask in dataset:
            out = model_forward(model, img.unsqueeze(0).to(DEVICE))
            pred = torch.argmax(out, dim=1)[0].cpu()
            conf += confusion_matrix(pred, mask).cpu().numpy()
    ious, miou = mious_from_confusion(conf)
    return miou, pixel_acc_from_confusion(conf)


def evaluate_ort(session, dataset):
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for img, mask in dataset:
        inp = img.unsqueeze(0).numpy().astype(np.float32)
        out = ort_infer_once(session, inp)          # (1, 2, H, W)
        pred = out[0].argmax(axis=0)                # (H, W)
        gt = mask.numpy()
        mask_ok = (gt >= 0) & (gt < NUM_CLASSES)
        idx = NUM_CLASSES * gt[mask_ok] + pred[mask_ok]
        conf += np.bincount(idx.astype(np.int64), minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)
    ious, miou = mious_from_confusion(conf)
    return miou, pixel_acc_from_confusion(conf)


# ---------------- 主流程 ----------------
def main():
    os.makedirs(ONNX_DIR, exist_ok=True)
    print("=" * 78)
    print("  实验1：部署量化闭环（ONNX → ORT FP32 / INT8）")
    print(f"  设备: {DEVICE}    输入: {CROP_SIZE}    城市: val 全部城市")
    print("=" * 78)

    val_dataset = CityscapesBlindDataset(DATA_ROOT, split="val")
    print(f"  验证集: {len(val_dataset)} 张\n")

    results = []
    for name in DEPLOY_MODELS:
        print(f">>> 处理 {name} ...")
        row = {"model": name}
        # ---- PyTorch FP32 ----
        model, loaded = build_model(name)
        if not loaded:
            print(f"  [跳过] 权重不存在: {resolve_weight_path(name)}")
            continue
        miou_pt, pa_pt = evaluate_pt(model, val_dataset)
        fps_pt, lat_pt = measure_fps_pt(model)
        print(f"  PyTorch FP32 : mIoU={miou_pt*100:.2f}%  FPS={fps_pt:.1f}  {lat_pt:.2f}ms")

        # ---- ONNX 导出 ----
        onnx_path = os.path.join(ONNX_DIR, f"{name}.onnx")
        try:
            export_onnx(model, onnx_path)
            print(f"  ONNX 导出    : {os.path.basename(onnx_path)}")
        except Exception as e:
            print(f"  [警告] ONNX 导出失败（跳过 ORT 对比）: {e}")
            row.update({"pt_miou": round(miou_pt * 100, 2), "pt_fps": round(fps_pt, 1)})
            results.append(row)
            del model
            torch.cuda.empty_cache()
            continue

        # ---- ORT FP32 ----
        sess_fp32 = make_session(onnx_path)
        miou_fp32, _ = evaluate_ort(sess_fp32, val_dataset)
        fps_fp32, lat_fp32 = measure_fps_ort(sess_fp32)
        print(f"  ORT FP32(GPU): mIoU={miou_fp32*100:.2f}%  FPS={fps_fp32:.1f}  {lat_fp32:.2f}ms")

        # ---- ORT FP32（CPU，用于与 INT8 同设备对比）----
        # 注意：INT8 动态量化算子只能在 CPU provider 上跑，若拿 GPU 上的 FP32 速度
        # 去比 CPU 上的 INT8 速度，会得到"量化后反而慢十几倍"的错误结论。
        # 因此额外测一组 CPU 上的 FP32 速度，作为量化加速比的唯一合法基准。
        fps_fp32_cpu = lat_fp32_cpu = None
        try:
            sess_fp32_cpu = make_session(onnx_path, cpu_only=True)
            fps_fp32_cpu, lat_fp32_cpu = measure_fps_ort(sess_fp32_cpu)
            print(f"  ORT FP32(CPU): FPS={fps_fp32_cpu:.1f}  {lat_fp32_cpu:.2f}ms")
        except Exception as e:
            print(f"  [警告] CPU FP32 测速失败: {e}")

        # ---- ORT INT8 量化 ----
        int8_path = os.path.join(ONNX_DIR, f"{name}_int8.onnx")
        miou_int8, fps_int8, lat_int8 = None, None, None
        try:
            quantize_int8(onnx_path, int8_path)
            sess_int8 = make_session(int8_path, cpu_only=True)
            miou_int8, _ = evaluate_ort(sess_int8, val_dataset)
            fps_int8, lat_int8 = measure_fps_ort(sess_int8)
            print(f"  ORT INT8(CPU): mIoU={miou_int8*100:.2f}%  FPS={fps_int8:.1f}  {lat_int8:.2f}ms")
        except Exception as e:
            print(f"  [警告] INT8 量化失败: {e}")

        row.update({
            "pt_miou": round(miou_pt * 100, 2),
            "pt_fps": round(fps_pt, 1),
            "pt_latency_ms": round(lat_pt, 2),
            "ort_fp32_miou": round(miou_fp32 * 100, 2),
            "ort_fp32_fps": round(fps_fp32, 1),
            "ort_fp32_latency_ms": round(lat_fp32, 2),
            "ort_fp32_cpu_fps": round(fps_fp32_cpu, 1) if fps_fp32_cpu is not None else "N/A",
            "ort_fp32_cpu_latency_ms": round(lat_fp32_cpu, 2) if lat_fp32_cpu is not None else "N/A",
            "ort_int8_miou": round(miou_int8 * 100, 2) if miou_int8 is not None else "N/A",
            "ort_int8_fps": round(fps_int8, 1) if fps_int8 is not None else "N/A",
            "ort_int8_latency_ms": round(lat_int8, 2) if lat_int8 is not None else "N/A",
            "int8_speedup_vs_cpu_fp32": (round(fps_int8 / fps_fp32_cpu, 2)
                                         if (fps_int8 and fps_fp32_cpu) else "N/A"),
        })
        results.append(row)
        del model
        torch.cuda.empty_cache()

    # ---- 保存 CSV ----
    fieldnames = ["model", "pt_miou", "pt_fps", "pt_latency_ms",
                  "ort_fp32_miou", "ort_fp32_fps", "ort_fp32_latency_ms",
                  "ort_fp32_cpu_fps", "ort_fp32_cpu_latency_ms",
                  "ort_int8_miou", "ort_int8_fps", "ort_int8_latency_ms",
                  "int8_speedup_vs_cpu_fp32"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n结果已保存: {OUTPUT_CSV}")

    # ---- 打印汇总表 ----
    print("\n" + "=" * 92)
    print(f"  {'模型':<14s} {'PT-FP32':<9s} {'ORT-FP32':<9s} {'ORT-INT8':<9s} "
          f"{'FP32@GPU':<9s} {'FP32@CPU':<9s} {'INT8@CPU':<9s} {'加速比':<8s}")
    print("  " + "-" * 88)
    for r in results:
        print(f"  {r['model']:<14s} {str(r.get('pt_miou','')):<9s} {str(r.get('ort_fp32_miou','')):<9s} "
              f"{str(r.get('ort_int8_miou','')):<9s} {str(r.get('pt_fps','')):<9s} "
              f"{str(r.get('ort_fp32_cpu_fps','')):<9s} {str(r.get('ort_int8_fps','')):<9s} "
              f"{str(r.get('int8_speedup_vs_cpu_fp32','')):<8s}")
    print("  mIoU 三列单位 %，FPS 三列单位 frames/s；加速比 = INT8@CPU ÷ FP32@CPU")
    print("=" * 92)


if __name__ == "__main__":
    main()