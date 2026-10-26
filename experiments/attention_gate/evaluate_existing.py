"""Step 2: evaluate saved best weights without modifying historical results.

Every output directory must be new. Smoke and full runs have separate outputs.
"""
import argparse
import collections
import csv
import ctypes
import datetime
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
WORKSPACE = PROJECT.parent
CODE = PROJECT / "code"
sys.path.insert(0, str(CODE))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
import eval_common as ec
from eval_main_table import TRAIN_CSV
from metrics_v1 import boundary_band, confusion_matrix_numpy, scores_from_confusion


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, data, exclusive=True):
    with Path(path).open("x" if exclusive else "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_csv(path, rows):
    with Path(path).open("x", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def timestamp():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds")


def available_memory():
    if os.name != "nt":
        return 0
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                    ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong),
                    ("total_page", ctypes.c_ulonglong), ("available_page", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                    ("extended", ctypes.c_ulonglong)]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    return int(status.available) if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) else 0


def protected_hashes():
    inventory = json.loads((WORKSPACE / "_organization_records/2026-10-13/inventory_after.json").read_text(encoding="utf-8"))
    expected = {row["new"]: row["sha256"] for row in inventory if row["new"].startswith(
        ("blind_road_project/code/", "blind_road_project/code_sam/", "blind_road_project/weights/", "blind_road_project/train_logs/"))}
    actual = {rel: digest(WORKSPACE / rel) for rel in expected}
    mismatch = [rel for rel in expected if expected[rel] != actual[rel]]
    if mismatch:
        raise RuntimeError("Protected files differ from organization record: " + ", ".join(mismatch))
    return actual


def reported_epoch(name):
    filename = TRAIN_CSV[name]
    with (PROJECT / "train_logs" / filename).open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    key = "val_mIoU" if "val_mIoU" in rows[0] else "val_miou"
    best = max(rows, key=lambda row: float(row[key]))
    return {"training_csv": filename, "reported_best_epoch": int(best["epoch"]),
            "reported_training_miou": float(best[key]),
            "selection": "historical training metric; epoch inferred from CSV, not embedded in pure state_dict"}


def load_strict(spec):
    model = spec["build"]()
    state = torch.load(PROJECT / "weights" / spec["weight"], map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    del state
    model.eval()
    return model


def prepare_samples(dataset, indices, ratio, cache_rgb, log):
    samples, checks = [], []
    first_by_city = {}
    for index in indices:
        first_by_city.setdefault(Path(dataset.img_paths[index]).parent.name, index)
    for count, index in enumerate(indices, 1):
        image_path = Path(dataset.img_paths[index])
        label_path = Path(dataset.label_paths[index])
        with Image.open(label_path) as image:
            labels = np.array(image)
        target = np.asarray(Image.fromarray((labels == 8).astype(np.uint8)).resize((1024, 768), Image.NEAREST)).copy()
        rgb = None
        if cache_rgb:
            with Image.open(image_path) as image:
                rgb = np.asarray(image.convert("RGB").resize((1024, 768), Image.BILINEAR)).copy()
        sample = {"index": index, "id": image_path.name, "city": image_path.parent.name,
                  "image": image_path, "label": label_path, "target": target,
                  "target_boundary": boundary_band(target == 1, ratio), "rgb": rgb}
        samples.append(sample)
        if index == first_by_city[sample["city"]]:
            reference_img, reference_mask = dataset[index]
            if rgb is not None:
                cached_img = dataset.transform_img(Image.fromarray(rgb))
                equal = torch.equal(reference_img, cached_img)
            else:
                equal = True
            mask_equal = np.array_equal(reference_mask.numpy(), target)
            checks.append({"sample": sample["id"], "cached_input_matches_dataset": bool(equal), "mask_matches_dataset": bool(mask_equal)})
            if not equal or not mask_equal:
                raise RuntimeError("Cached preprocessing does not match the established dataset")
        if count % 100 == 0:
            log(f"Prepared ground truth {count}/{len(indices)}")
    return samples, checks


def summarize(model_name, rows, params, provenance, elapsed, memory_peak):
    conf = np.array([[sum(row["conf_00"] for row in rows), sum(row["conf_01"] for row in rows)],
                     [sum(row["conf_10"] for row in rows), sum(row["conf_11"] for row in rows)]], dtype=np.int64)
    scores = scores_from_confusion(conf)
    reference_ious, reference_miou = ec.mious_from_confusion(conf)
    if not np.allclose(scores["iou"], reference_ious, rtol=0, atol=1e-12) or abs(scores["miou"] - reference_miou) > 1e-12:
        raise RuntimeError("Global metrics disagree with independent project calculation")
    boundary_values = [row["boundary_iou_author_fg"] for row in rows if row["boundary_union_fg"] > 0]
    return {"model": model_name, "images": len(rows), "params_m": params / 1e6,
            "miou_global_pct": scores["miou"] * 100, "iou_sidewalk_global_pct": scores["iou"][1] * 100,
            "boundary_iou_author_fg_pct": float(np.mean(boundary_values)) * 100 if boundary_values else None,
            "boundary_valid_images": len(boundary_values), "boundary_excluded_empty_pairs": len(rows) - len(boundary_values),
            "iou_background_global_pct": scores["iou"][0] * 100, "pixel_accuracy_global_pct": scores["pixel_acc"] * 100,
            "miou_per_image_pct": float(np.mean([row["miou_per_image"] for row in rows])) * 100,
            "historical_reported_best_epoch": provenance["reported_best_epoch"],
            "weight": provenance["weight"], "weight_sha256": provenance["weight_sha256"],
            "conf_00": int(conf[0, 0]), "conf_01": int(conf[0, 1]), "conf_10": int(conf[1, 0]), "conf_11": int(conf[1, 1]),
            "evaluation_wall_seconds_not_latency": elapsed, "cuda_peak_allocated_mib": memory_peak}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if not output.is_relative_to(HERE):
        raise RuntimeError("Output must stay inside the new attention experiment directory")
    if output.exists():
        raise RuntimeError("Refusing to overwrite an existing output directory")
    output.mkdir(parents=True)
    log_path = output / "console.log"
    def log(message):
        line = timestamp() + " " + message
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    start = time.perf_counter()
    try:
        protocol = json.loads((HERE / "protocol_v1.json").read_text(encoding="utf-8"))
        step1 = json.loads((HERE / "step_01_2026-10-14/run_002/step_status.json").read_text(encoding="utf-8"))
        if step1["status"] != "complete":
            raise RuntimeError("Step 1 must be complete")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; do not silently switch the full run to CPU")
        assert protocol["data"]["input_height"] == 768 and protocol["data"]["input_width"] == 1024
        assert protocol["data"]["foreground_original_label_id"] == 8
        torch.set_num_threads(4)
        cv2.setNumThreads(1)
        torch.manual_seed(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        log("Checking original code, weights and results")
        before = protected_hashes()
        write_json(output / "protected_files_before.json", before)
        specs = dict(ec.MODEL_SPECS)
        specs["DeepLabv3+"] = {"build": lambda: __import__("model_deeplabv3plus").get_deeplabv3plus_2class(ec.DEVICE), "weight": "best_deeplab_100epoch_bin.pth"}
        if len(specs) != 15 or "U-Net" in specs:
            raise RuntimeError("Expected exactly 15 completed models, excluding U-Net")
        dataset = ec.CityscapesBlindDataset(ec.DATA_ROOT, split="val", crop_size=(768, 1024))
        if len(dataset) != 500:
            raise RuntimeError("Expected all 500 validation images")
        if args.mode == "smoke":
            city_first = {}
            for i, p in enumerate(dataset.img_paths):
                city_first.setdefault(Path(p).parent.name, i)
            indices = list(city_first.values())
        else:
            indices = list(range(500))
        free_memory = available_memory()
        # 500 uint8 RGBs + masks + boundary bands occupy about 1.97 GB.
        # Require >=4 GiB free at startup, leaving room for the Python runtime.
        cache_rgb = free_memory >= 4 * 1024 ** 3 or args.mode == "smoke"
        metadata = {"protocol_id": protocol["protocol_id"], "protocol_sha256": digest(HERE / "protocol_v1.json"),
                    "metrics_sha256": digest(HERE / "metrics_v1.py"), "evaluator_sha256": digest(__file__),
                    "mode": args.mode, "started": timestamp(), "expected_models": list(specs),
                    "samples_per_model": len(indices), "validation_is_not_an_independent_test_set": True,
                    "weight_selection": "historical training metric; not best under newly defined metrics",
                    "boundary_width_pixels": protocol["metrics"]["boundary_width_at_core_resolution"],
                    "python": sys.version.split()[0], "torch": str(torch.__version__), "torchvision": torchvision.__version__,
                    "cuda": torch.version.cuda, "numpy": np.__version__, "opencv": cv2.__version__,
                    "gpu": torch.cuda.get_device_name(0), "cudnn_deterministic": True,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                    "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "available_host_memory_bytes": free_memory, "cache_resized_rgb": cache_rgb,
                    "measure_latency_or_fps": False, "no_training": True}
        write_json(output / "run_metadata.json", metadata)
        with (output / "evaluator_snapshot.py").open("xb") as f:
            f.write(Path(__file__).read_bytes())
        write_json(output / "protocol_snapshot.json", protocol)
        log(f"Preparing {len(indices)} samples; resized RGB cache={cache_rgb}")
        samples, preprocessing_checks = prepare_samples(dataset, indices, protocol["metrics"]["boundary_dilation_ratio"], cache_rgb, log)
        write_json(output / "preprocessing_checks.json", preprocessing_checks)
        manifest = [{"index": sample["index"], "sample_id": sample["id"], "city": sample["city"],
                     "image_relative": str(sample["image"].relative_to(PROJECT)).replace(os.sep, "/"),
                     "label_relative": str(sample["label"].relative_to(PROJECT)).replace(os.sep, "/"),
                     "image_bytes": sample["image"].stat().st_size, "image_mtime_ns": sample["image"].stat().st_mtime_ns,
                     "label_bytes": sample["label"].stat().st_size, "label_mtime_ns": sample["label"].stat().st_mtime_ns} for sample in samples]
        write_json(output / "sample_manifest.json", manifest)
        (output / "per_model").mkdir()
        summaries, city_summaries = [], []
        progress = {"mode": args.mode, "status": "running", "models_completed": 0, "total_models": 15, "predictions_completed": 0}
        for model_index, (name, spec) in enumerate(specs.items(), 1):
            model_start = time.perf_counter()
            log(f"Model {model_index}/15: {name}")
            torch.cuda.reset_peak_memory_stats()
            model = load_strict(spec)
            params = sum(p.numel() for p in model.parameters())
            provenance = reported_epoch(name)
            provenance.update({"weight": spec["weight"], "weight_sha256": digest(PROJECT / "weights" / spec["weight"]),
                               "strict_load": True, "params": params})
            rows = []
            with torch.inference_mode():
                for sample_index, sample in enumerate(samples, 1):
                    img = dataset.transform_img(Image.fromarray(sample["rgb"])) if sample["rgb"] is not None else dataset[sample["index"]][0]
                    logits = ec.model_forward(model, img.unsqueeze(0).to(ec.DEVICE))
                    if list(logits.shape) != [1, 2, 768, 1024] or not bool(torch.isfinite(logits).all()):
                        raise RuntimeError(f"Invalid logits: {name}, {sample['id']}")
                    pred = logits.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()
                    conf = confusion_matrix_numpy(pred, sample["target"])
                    image_scores = scores_from_confusion(conf)
                    predicted_boundary = boundary_band(pred == 1, protocol["metrics"]["boundary_dilation_ratio"])
                    boundary_intersection = int(np.count_nonzero(predicted_boundary & sample["target_boundary"]))
                    boundary_union = int(np.count_nonzero(predicted_boundary | sample["target_boundary"]))
                    boundary_score = boundary_intersection / boundary_union if boundary_union else None
                    row = {"model": name, "index": sample["index"], "sample_id": sample["id"], "city": sample["city"],
                           "conf_00": int(conf[0, 0]), "conf_01": int(conf[0, 1]), "conf_10": int(conf[1, 0]), "conf_11": int(conf[1, 1]),
                           "miou_per_image": image_scores["miou"], "iou_fg_per_image": image_scores["iou"][1],
                           "boundary_intersection_fg": boundary_intersection, "boundary_union_fg": boundary_union,
                           "boundary_iou_author_fg": boundary_score, "gt_fg_pixels": int(np.count_nonzero(sample["target"])),
                           "pred_fg_pixels": int(np.count_nonzero(pred))}
                    if sum(conf.ravel()) != 768 * 1024 or not 0 <= image_scores["miou"] <= 1 or (boundary_score is not None and not 0 <= boundary_score <= 1):
                        raise RuntimeError("Metric range or pixel-count check failed")
                    rows.append(row)
                    del logits, pred, img
                    if sample_index % 100 == 0:
                        log(f"  {name}: {sample_index}/{len(samples)}")
                        progress.update({"current_model": name, "current_model_images": sample_index,
                                         "predictions_completed": (model_index - 1) * len(samples) + sample_index,
                                         "elapsed_seconds": time.perf_counter() - start})
                        write_json(output / "progress.json", progress, exclusive=False)
            elapsed = time.perf_counter() - model_start
            memory_peak = torch.cuda.max_memory_allocated() / 1024 ** 2
            summary = summarize(name, rows, params, provenance, elapsed, memory_peak)
            summaries.append(summary)
            for city in sorted({row["city"] for row in rows}):
                city_rows = [row for row in rows if row["city"] == city]
                city_row = summarize(name, city_rows, params, provenance, 0, 0)
                city_row["city"] = city
                city_summaries.append(city_row)
            safe_name = name.replace("+", "_plus_").replace("-", "_")
            write_csv(output / "per_model" / f"{safe_name}_per_image.csv", rows)
            write_json(output / "per_model" / f"{safe_name}_summary.json", {"summary": summary, "provenance": provenance})
            log(f"Finished {name}: global mIoU={summary['miou_global_pct']:.4f}%, sidewalk IoU={summary['iou_sidewalk_global_pct']:.4f}%, author Boundary IoU={summary['boundary_iou_author_fg_pct']:.4f}%")
            progress.update({"models_completed": model_index, "current_model_images": len(samples),
                             "predictions_completed": model_index * len(samples), "elapsed_seconds": time.perf_counter() - start})
            write_json(output / "progress.json", progress, exclusive=False)
            del model, rows
            gc.collect()
            torch.cuda.empty_cache()
        write_csv(output / "main_table.csv", summaries)
        write_csv(output / "per_city_table.csv", city_summaries)
        write_json(output / "main_table.json", summaries)
        after = protected_hashes()
        if before != after:
            raise RuntimeError("Historical code or results changed during evaluation")
        for item in manifest:
            for prefix in ("image", "label"):
                path = PROJECT / item[prefix + "_relative"]
                if path.stat().st_size != item[prefix + "_bytes"] or path.stat().st_mtime_ns != item[prefix + "_mtime_ns"]:
                    raise RuntimeError("Input data changed during evaluation")
        integrity = {"protected_files_checked": len(before), "before_after_sha256_identical": True, "input_metadata_unchanged": True}
        write_json(output / "integrity_after.json", integrity)
        status = {"step": 2, "mode": args.mode, "status": "complete", "models_completed": len(summaries),
                  "images_per_model": len(samples), "total_predictions": len(summaries) * len(samples),
                  "wall_seconds": time.perf_counter() - start, "finished": timestamp(),
                  "historical_results_preserved": True, "training_started": False,
                  "next_step": "Step 3: common runner and implementation correctness checks before mechanism diagnostics or repeated training"}
        write_json(output / "step_status.json", status)
        progress.update({"status": "complete", "elapsed_seconds": status["wall_seconds"]})
        write_json(output / "progress.json", progress, exclusive=False)
        log(json.dumps(status, ensure_ascii=False))
    except BaseException as error:
        write_json(output / "failure.json", {"error_type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(), "at": timestamp()})
        log("FAILED: " + str(error))
        raise


if __name__ == "__main__":
    main()
