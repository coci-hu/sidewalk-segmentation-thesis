"""Read-only verification of epoch 2 while the independent job continues."""
import argparse
import json
import math
from pathlib import Path
import torch
import common_runner as core
import common_runner_v2 as fast
from check_step03 import equal_tree
from evaluate_existing import protected_hashes, timestamp
from run_step05_benchmark import previous_hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    job = args.job.resolve()
    assert job.is_relative_to(core.HERE)
    plan = json.loads((job / "job_plan.json").read_text(encoding="utf-8"))
    folder = Path(plan["variant_folders"]["A"])
    checkpoint_path = folder / "checkpoints/epoch_002.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    old_path = core.HERE / "step_05_2026-10-14/benchmark_001/A_seed42_epoch1/checkpoints/epoch_001.pth"
    old = torch.load(old_path, map_location="cpu", weights_only=False)
    assert checkpoint["settings"] == fast.expected_settings("A", 42)
    assert checkpoint["completed_epoch"] == checkpoint["scheduler_state"]["last_epoch"] == 2
    assert equal_tree(checkpoint["records"][:1], old["records"])
    assert [r["epoch"] for r in checkpoint["records"]] == [1, 2]
    row = checkpoint["records"][-1]
    assert row["train_images"] == 2975 and row["val_images"] == 500
    assert row["train_batch_sizes"] == [4] * 743 + [3]
    assert sorted(row["train_order"]) == list(range(2975))
    assert sum(map(sum, row["val_confusion"])) == 500 * 768 * 1024
    lr = 1e-6 + (1e-4 - 1e-6) * (1 + math.cos(math.pi * 2 / 100)) / 2
    assert abs(row["lr_next"] - lr) < 1e-14
    assert all(state["step"].item() == 1488 for state in checkpoint["optimizer_state"]["state"].values())
    assert all(value.item() == 1488 for name, value in checkpoint["model_state"].items() if name.endswith("num_batches_tracked"))
    assert all(bool(torch.isfinite(value).all()) for value in checkpoint["model_state"].values() if value.is_floating_point())
    baseline = json.loads((job / "integrity_before.json").read_text(encoding="utf-8"))
    assert protected_hashes() == baseline["original"]
    assert previous_hashes() == baseline["previous"]
    assert all(core.digest(core.HERE / name) == digest for name, digest in baseline["benchmark"].items())
    assert fast.sources() == plan["source_hashes"]
    timing = next(r for r in json.loads((folder / "epoch_timing.json").read_text(encoding="utf-8")) if r["epoch"] == 2)
    assert timing["train_seconds"] > 0 and timing["validation_seconds"] > 0
    output = core.HERE / "step_05_2026-10-14/loader_optimization_001/first_optimized_epoch_verification.json"
    core.json_write(output, {"status": "pass", "checked": timestamp(), "variant": "A", "completed_epoch": 2,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": core.digest(checkpoint_path),
        "epoch1_history_preserved": True, "optimizer_steps_each_parameter": 1488,
        "settings_sources_and_old_results_unchanged": True, "full_training_and_validation_counts_correct": True,
        "finite_parameters": True, "timing": timing, "full_100_epoch_study_complete": False})
    print(json.dumps({"status": "pass", "timing": timing, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
