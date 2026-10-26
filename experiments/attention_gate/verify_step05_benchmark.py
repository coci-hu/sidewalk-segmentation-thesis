"""Verify complete one-epoch A/B benchmarks; estimate longer runs explicitly."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
import common_runner as cr
from evaluate_existing import protected_hashes, timestamp
from run_step05_benchmark import previous_hashes


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to(cr.HERE) or (run / "benchmark_verification.json").exists():
        raise ValueError("Use an unverified benchmark run")
    status = load_json(run / "step_status.json")
    assert status["status"] == "benchmark_complete_pending_verification"
    plan = load_json(run / "plan.json")
    assert plan["source_hashes"] == cr.source_hashes()
    results, checkpoints = [], {}
    for variant in ("A", "B"):
        folder = run / f"{variant}_seed42_epoch1"
        settings = load_json(folder / "settings.json")
        progress = load_json(folder / "status.json")
        timing = load_json(folder / "benchmark_timing.json")
        assert progress["status"] == "stopped_at_epoch_boundary" and progress["completed_epoch"] == 1
        assert settings["variant"] == variant and settings["seed"] == 42 and settings["mode"] == "train"
        assert settings["epoch_limit"] == settings["protocol"]["formal_training"]["scheduler_T_max"] == 100
        assert settings["source_hashes"] == plan["source_hashes"]
        checkpoint = torch.load(folder / "checkpoints/epoch_001.pth", map_location="cpu", weights_only=False)
        assert checkpoint["completed_epoch"] == checkpoint["scheduler_state"]["last_epoch"] == 1
        assert checkpoint["settings"] == settings
        assert len(checkpoint["records"]) == 1 and checkpoint["best_epoch"] == 1
        row = checkpoint["records"][0]
        assert row["train_images"] == 2975 and row["val_images"] == 500
        assert row["train_batch_sizes"] == [4] * 743 + [3]
        assert sorted(row["train_order"]) == list(range(2975))
        assert row["lr_used"] == 1e-4
        expected_lr = 1e-6 + (1e-4 - 1e-6) * (1 + math.cos(math.pi / 100)) / 2
        assert abs(row["lr_next"] - expected_lr) < 1e-15
        group = checkpoint["optimizer_state"]["param_groups"][0]
        assert group["weight_decay"] == 0.01 and group["betas"] == (0.9, 0.999)
        assert all(value["step"].item() == 744 for value in checkpoint["optimizer_state"]["state"].values())
        assert all(value.item() == 744 for name, value in checkpoint["model_state"].items() if name.endswith("num_batches_tracked"))
        assert all(bool(torch.isfinite(value).all()) for value in checkpoint["model_state"].values() if value.is_floating_point())
        initial = cr.build_paired_model(variant, 42)
        initial_hash = cr.state_digest({k: v for k, v in initial.state_dict().items() if not k.startswith("ca.")})
        assert initial_hash == load_json(folder / "initialization.json")["backbone_state_sha256"]
        best = torch.load(folder / "best_weights/epoch_001.pth", map_location="cpu", weights_only=True)
        assert equal_tree(best, checkpoint["model_state"]) and equal_tree(best, checkpoint["best_model_state"])
        assert cr.state_digest(checkpoint["model_state"]) == progress["final_state_sha256"]
        conf = np.array(row["val_confusion"], dtype=np.int64)
        assert conf.shape == (2, 2) and conf.sum() == 500 * 768 * 1024
        c00, c01, c10, c11 = conf.ravel()
        iou_fg = c11 / (c11 + c01 + c10) if c11 + c01 + c10 else 0
        iou_bg = c00 / (c00 + c01 + c10) if c00 + c01 + c10 else 0
        assert abs((iou_fg + iou_bg) / 2 - row["val_miou_global"]) < 1e-12
        with (folder / "metrics.csv").open(encoding="utf-8-sig", newline="") as f:
            csv_rows = list(csv.DictReader(f))
        assert len(csv_rows) == 1 and int(csv_rows[0]["epoch"]) == 1
        for field, value in csv_rows[0].items():
            if field != "epoch" and value != "":
                assert abs(float(value) - row[field]) < 1e-12
        with (folder / "batch_timing.csv").open(encoding="utf-8-sig", newline="") as f:
            batch_times = list(csv.DictReader(f))
        assert len(batch_times) == 1244
        phases = {}
        for phase, batches, images in (("train", 744, 2975), ("validation", 500, 500)):
            phase_rows = [r for r in batch_times if r["phase"] == phase]
            assert len(phase_rows) == batches and sum(int(r["images"]) for r in phase_rows) == images
            assert [int(r["batch"]) for r in phase_rows] == list(range(1, batches + 1))
            wait = sum(float(r["loader_wait_seconds"]) for r in phase_rows)
            work = sum(float(r["caller_work_seconds"]) for r in phase_rows)
            assert wait >= 0 and work > 0 and wait + work <= timing[phase + "_seconds"] + 1e-5
            phases[phase + "_loader_wait_seconds"] = wait
            phases[phase + "_caller_work_seconds"] = work
        measured_epoch = timing["train_seconds"] + timing["validation_seconds"]
        powers = load_json(folder / "gpu_power_samples.json")
        valid_power = [r["gpu_power_w"] for r in powers if r.get("gpu_power_w") is not None]
        results.append({"variant": variant, "model": "ENet" if variant == "A" else "ENet+CA", "seed": 42,
            "train_seconds": timing["train_seconds"], "validation_seconds": timing["validation_seconds"],
            "train_plus_validation_seconds": measured_epoch, "whole_runner_call_seconds": timing["whole_runner_call_wall_seconds"],
            **phases, "cuda_peak_allocated_mib": timing["cuda_peak_allocated_mib"],
            "gpu_power_sample_mean_w": float(np.mean(valid_power)) if valid_power else None,
            "gpu_power_samples": len(valid_power), "val_miou_epoch1_pct": row["val_miou_global"] * 100,
            "val_sidewalk_iou_epoch1_pct": iou_fg * 100,
            "estimated_100_epochs_hours": measured_epoch * 100 / 3600,
            "estimated_remaining_99_epochs_hours": measured_epoch * 99 / 3600,
            "checkpoint": str(folder / "checkpoints/epoch_001.pth"),
            "checkpoint_sha256": cr.digest(folder / "checkpoints/epoch_001.pth")})
        checkpoints[variant] = checkpoint
        del initial, best
    a, b = checkpoints["A"], checkpoints["B"]
    assert a["records"][0]["train_order"] == b["records"][0]["train_order"]
    assert equal_tree(a["rng"], b["rng"]), "Paired RNG states diverged"
    assert a["settings"]["manifest_sha256"] == b["settings"]["manifest_sha256"]
    assert a["settings"]["protocol"] == b["settings"]["protocol"]
    assert protected_hashes() == load_json(run / "protected_files_before.json")
    assert previous_hashes() == load_json(run / "previous_steps_before.json")
    document_meta = load_json(run.parent / "document_read_001/document_read_metadata.json")
    document_path = Path(document_meta["source_file"])
    assert cr.digest(document_path) == document_meta["file_sha256"]
    pair_hours = sum(r["estimated_100_epochs_hours"] for r in results)
    estimates = {"method": "linear extrapolation from first full train+val epoch; estimate, not guarantee",
        "pair_seed42_100_epochs_hours": pair_hours, "pair_seed42_remaining_hours": pair_hours * 0.99,
        "all_three_paired_seeds_100_epochs_hours": pair_hours * 3,
        "continuous_days_for_all_six_runs": pair_hours * 3 / 24,
        "at_8_hours_per_day_days_for_all_six_runs": pair_hours * 3 / 8,
        "illustrative_runtime_range_hours_all_six_runs": [pair_hours * 3 * 0.8, pair_hours * 3 * 1.3],
        "range_is_not_statistical_CI": True, "paid_hardware_or_server_cost_cny": 0,
        "electricity_total_cost_not_measured": True, "longer_training_started": False}
    cr.json_write(run / "benchmark_summary.json", {"results": results, "estimates": estimates})
    cr.json_write(run / "benchmark_verification.json", {"status": "pass", "epochs_per_variant": 1,
        "all_images_and_batches_accounted_for": True, "optimizer_steps_each_parameter": 744,
        "scheduler_completed_steps": 1, "cosine_T_max": 100, "paired_backbone_initialization": True,
        "paired_train_order_identical": True, "paired_final_all_rng_states_identical": True,
        "checkpoints_and_best_weights_consistent": True, "global_validation_counts_reconstructed": True,
        "timing_batch_counts_checked": True, "original_and_previous_step_hashes_identical": True,
        "thought_document_sha256_unchanged": True, "common_runner_sources_unchanged": True})
    status.update({"status": "complete", "benchmark_verified": True, "verification_completed": timestamp(),
        "step5_full_repeated_training_complete": False, "longer_training_started": False,
        "next_step": "Continue fixed-seed42 A/B epochs2..100 from these NEW checkpoints; other seeds and candidate C/D/E not started"})
    cr.json_write(run / "step_status.json", status, exclusive=False)
    cr.json_write(run / "progress.json", status, exclusive=False)
    print(json.dumps({"results": results, "estimates": estimates}, ensure_ascii=False))


if __name__ == "__main__":
    main()
