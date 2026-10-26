"""Step 1: recheck preserved files, inspect history, and validate metric rules.

Reads old experiments and writes only a fresh step_01 result directory.
No training, model inference, plots, or external uploads are performed.
"""
import ast
import concurrent.futures
import csv
import gc
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ROOT = PROJECT.parent
CODE = PROJECT / "code"
sys.path.insert(0, str(CODE))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
OUTPUT = HERE / "step_01_2026-10-14" / "run_002"

CHECKPOINTS = {
    "ENet": "enet_ckpt.pth", "ENet+CA": "enet_ca_ckpt.pth",
    "ENet+BA": "enet_ba_ckpt.pth", "ENet-DBA": "enet_dba_ckpt.pth",
    "ENet+L-ASPP": "enet_laspp_ckpt.pth", "ENet-CBR": "enet_cbr_ckpt.pth",
    "SBNet": "ckpt_100bin.pth", "SBNet+CA": "ckpt_ca_100bin.pth",
    "SBNet+L-ASPP": "bisenetv2_laspp_ckpt.pth", "SBNet-CBR": "bisenetv2_cbr_ckpt.pth",
    "Fast-SCNN": "fastscnn_ckpt.pth", "SegFormer-B0": "segformer_ckpt.pth",
    "STDC-Seg": "ckpt_stdc_seg_100bin.pth", "PP-LiteSeg-T": "ckpt_ppliteseg_100bin.pth",
    "DeepLabv3+": "ckpt_deeplab_100bin.pth",
}


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save(name, value):
    p = OUTPUT / name
    with p.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main():
    if OUTPUT.exists():
        raise RuntimeError("Step 1 output already exists; refusing to overwrite it.")
    OUTPUT.mkdir(parents=True)
    record = ROOT / "_organization_records/2026-10-13"
    inventory = json.loads((record / "inventory_after.json").read_text(encoding="utf-8"))
    missing, size_changed, hash_changed = [], [], []
    critical = []
    for row in inventory:
        path = ROOT / row["new"]
        if not path.is_file():
            missing.append(row["new"])
            continue
        if path.stat().st_size != row["bytes"]:
            size_changed.append(row["new"])
        if row["new"].startswith(("blind_road_project/code/", "blind_road_project/code_sam/", "blind_road_project/weights/", "blind_road_project/train_logs/")):
            critical.append(row)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for row, digest in zip(critical, pool.map(lambda r: sha256(ROOT / r["new"]), critical)):
            if digest != row["sha256"]:
                hash_changed.append(row["new"])
    integrity = {"original_paths_size_checked": len(inventory), "critical_files_sha256_checked": len(critical),
                 "missing": missing, "size_changed": size_changed, "hash_changed": hash_changed}
    save("integrity_recheck.json", integrity)
    print(json.dumps(integrity, ensure_ascii=False), flush=True)
    if missing or size_changed or hash_changed:
        raise RuntimeError("Files changed since organization; inspect the recheck report before proceeding.")

    import numpy as np
    import torch
    import yaml
    import cv2
    from metrics_v1 import boundary_band, boundary_iou_author, confusion_matrix_numpy, scores_from_confusion
    import eval_boundary_iou as historical_boundary
    import metrics as historical_metrics

    checkpoints = []
    for name, filename in CHECKPOINTS.items():
        path = PROJECT / "weights" / filename
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        optimizer = checkpoint.get("optim_state", checkpoint.get("optimizer_state_dict", {}))
        scheduler = checkpoint.get("scheduler_state", {})
        checkpoints.append({"model": name, "checkpoint": filename, "sha256": sha256(path),
                            "epoch": checkpoint.get("epoch"),
                            "optimizer_groups": [{k: group.get(k) for k in ("lr", "initial_lr", "weight_decay", "betas", "eps", "amsgrad")} for group in optimizer.get("param_groups", [])],
                            "scheduler": {k: scheduler.get(k) for k in ("T_max", "eta_min", "last_epoch")},
                            "has_saved_config": "config" in checkpoint,
                            "has_rng_state": any("rng" in key.lower() for key in checkpoint),
                            "recorded_rows": len(checkpoint.get("log_records", []))})
        print("Checkpoint inspected:", name, flush=True)
        del checkpoint
        gc.collect()
    save("historical_checkpoint_protocols.json", checkpoints)

    logs = []
    for path in sorted((PROJECT / "train_logs").glob("*.csv")):
        if "metrics" not in path.name and not path.name.startswith("train_log"):
            continue
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        epochs = [int(row["epoch"]) for row in rows]
        logs.append({"file": path.name, "rows": len(rows), "first_epoch": min(epochs), "last_epoch": max(epochs),
                     "missing_epochs": sorted(set(range(1, 101)) - set(epochs)), "sha256": sha256(path)})
    save("historical_training_logs.json", logs)

    target = np.zeros((80, 80), dtype=bool)
    target[10:70, 10:70] = True
    nested = np.zeros_like(target)
    nested[12:68, 12:68] = True
    other = np.zeros_like(target)
    other[0:5, 0:5] = True
    empty = np.zeros_like(target)
    new_nested = boundary_iou_author(target, nested, width_pixels=2)
    old_nested = historical_boundary.boundary_iou_official(boundary_band(target, width_pixels=2), boundary_band(nested, width_pixels=2), target, nested)
    tests = {
        "perfect_boundary_is_one": boundary_iou_author(target, target, width_pixels=2) == 1.0,
        "disjoint_boundary_is_zero": boundary_iou_author(target, other, width_pixels=2) == 0.0,
        "empty_pair_is_excluded": boundary_iou_author(empty, empty, width_pixels=2) is None,
        "empty_prediction_is_zero": boundary_iou_author(target, empty, width_pixels=2) == 0.0,
        "image_border_is_retained": bool(boundary_band(np.ones((80, 80), dtype=bool), width_pixels=2)[0].all()),
        "nested_nonoverlap_boundaries_are_zero": new_nested == 0.0,
        "confusion_matches_expected": confusion_matrix_numpy(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])).tolist() == [[1, 1], [1, 1]],
        "global_miou_matches_expected": bool(np.isclose(scores_from_confusion([[9, 1], [1, 9]])["miou"], 9 / 11)),
        "historical_formula_is_different": old_nested != new_nested,
    }
    gt = torch.stack([torch.zeros((10, 10), dtype=torch.long), torch.ones((10, 10), dtype=torch.long)])
    pred = gt.clone()
    pred[1, :5] = 0
    batch_score = float(historical_metrics.compute_mIoU(pred, gt))
    image_score = float(np.mean([historical_metrics.compute_mIoU(pred[i], gt[i]) for i in range(2)]))
    tests["batch_miou_is_not_per_image_miou"] = not np.isclose(batch_score, image_score)
    metric_report = {"tests": tests, "nested_example": {"shape": [80, 80], "boundary_width": 2,
                     "historical_labeled_official": old_nested, "author_boundary_iou": new_nested},
                     "miou_example": {"historical_batch": batch_score, "per_image_average": image_score},
                     "author_source": "https://gist.github.com/bowenc0221/71f7a02afee92646ca05efeeb14d687d",
                     "new_module_sha256": sha256(HERE / "metrics_v1.py")}
    save("metric_definition_tests.json", metric_report)
    if not all(tests.values()):
        raise RuntimeError("Metric definition tests failed")

    city_counts = {}
    for split, expected in (("train", 2975), ("val", 500)):
        images = sorted((PROJECT / "dataset/cityscapes/leftImg8bit" / split).rglob("*_leftImg8bit.png"))
        missing_labels = [str(p) for p in images if not (PROJECT / "dataset/cityscapes/gtFine" / split / p.parent.name / p.name.replace("leftImg8bit", "gtFine_labelIds")).is_file()]
        assert len(images) == expected and not missing_labels
        city_counts[split] = {"images": len(images), "missing_labels": len(missing_labels)}

    environment = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
                   "numpy": np.__version__, "opencv": cv2.__version__, "cuda_available": torch.cuda.is_available(),
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   "adamw_default_weight_decay_in_installed_version": inspect.signature(torch.optim.AdamW).parameters["weight_decay"].default}
    save("environment.json", environment)
    status = {"step": 1, "status": "complete", "stage": "protocol_and_methodology_recheck",
              "old_results_modified": False, "training_started": False, "full_validation_inference_started": False,
              "metric_tests_passed": sum(tests.values()), "dataset": city_counts,
              "findings": ["Historical ENet and ENet+CA checkpoints have different weight decay.",
                           "Historical mIoU is averaged over batches, not images.",
                           "The historical Boundary IoU labeled official is not the author's formula.",
                           "Historical checkpoint scheduler counters differ; use a single new runner for formal comparisons."],
              "next_step": "Step 2: evaluate existing saved best weights with the fixed global metrics into a new output directory."}
    save("step_status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
