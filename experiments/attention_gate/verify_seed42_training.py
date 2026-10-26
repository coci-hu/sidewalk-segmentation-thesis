"""Verify a completed paired seed, retaining all histories and old artifacts."""
import argparse
import csv
import json
import math
from pathlib import Path
import torch
import common_runner as core
import common_runner_v2 as fast
from check_step03 import equal_tree
from evaluate_existing import protected_hashes, timestamp
from run_step05_benchmark import previous_hashes


def verify_variant(folder, variant, expected_epochs, seed=42):
    state = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    completed = state["completed_epoch"]
    assert completed == expected_epochs
    checkpoint_path = folder / f"checkpoints/epoch_{completed:03d}.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["settings"] == fast.expected_settings(variant, seed)
    assert checkpoint["completed_epoch"] == checkpoint["scheduler_state"]["last_epoch"] == completed
    records = checkpoint["records"]
    assert [row["epoch"] for row in records] == list(range(1, completed + 1))
    for row in records:
        assert row["train_images"] == 2975 and row["val_images"] == 500
        assert row["train_batch_sizes"] == [4] * 743 + [3]
        assert sorted(row["train_order"]) == list(range(2975))
        confusion = row["val_confusion"]
        c00, c01, c10, c11 = confusion[0] + confusion[1]
        assert c00 + c01 + c10 + c11 == 500 * 768 * 1024
        fg = c11 / (c11 + c01 + c10) if c11 + c01 + c10 else 0
        bg = c00 / (c00 + c01 + c10) if c00 + c01 + c10 else 0
        assert abs((fg + bg) / 2 - row["val_miou_global"]) < 1e-12
        lr = 1e-6 + (1e-4 - 1e-6) * (1 + math.cos(math.pi * row["epoch"] / 100)) / 2
        assert abs(row["lr_next"] - lr) < 1e-14
    assert all(value["step"].item() == completed * 744 for value in checkpoint["optimizer_state"]["state"].values())
    assert all(value.item() == completed * 744 for name, value in checkpoint["model_state"].items() if name.endswith("num_batches_tracked"))
    assert all(bool(torch.isfinite(value).all()) for value in checkpoint["model_state"].values() if value.is_floating_point())
    best_row = max(records, key=lambda row: row["val_miou_global"])
    assert best_row["epoch"] == checkpoint["best_epoch"]
    pure_best = torch.load(folder / f"best_weights/epoch_{best_row['epoch']:03d}.pth", map_location="cpu", weights_only=True)
    assert equal_tree(pure_best, checkpoint["best_model_state"])
    initial = core.build_paired_model(variant, seed)
    initial.load_state_dict(pure_best, strict=True)
    return checkpoint, {"variant": variant, "seed": seed, "epochs": completed, "best_epoch": best_row["epoch"],
        "best_global_miou_pct": best_row["val_miou_global"] * 100,
        "best_sidewalk_iou_pct": best_row["val_iou_sidewalk_global"] * 100,
        "best_boundary_iou_pct": best_row["val_boundary_iou_author_fg_mean"] * 100,
        "best_weights": str(folder / f"best_weights/epoch_{best_row['epoch']:03d}.pth"),
        "final_checkpoint": str(checkpoint_path), "final_checkpoint_sha256": core.digest(checkpoint_path)}


def finish(job, write_markdown=True):
    plan = json.loads((job / "job_plan.json").read_text(encoding="utf-8"))
    checkpoints, results = {}, []
    for variant in ("A", "B"):
        folder = Path(plan.get("variant_folders", {}).get(variant, job / f"{variant}_seed42_epochs002_100")).resolve()
        assert folder.is_relative_to(core.HERE)
        checkpoint, result = verify_variant(folder, variant, 100)
        checkpoints[variant] = checkpoint
        results.append(result)
    assert [r["train_order"] for r in checkpoints["A"]["records"]] == [r["train_order"] for r in checkpoints["B"]["records"]]
    assert equal_tree(checkpoints["A"]["rng"], checkpoints["B"]["rng"])
    baseline = json.loads((job / "integrity_before.json").read_text(encoding="utf-8"))
    assert protected_hashes() == baseline["original"]
    assert previous_hashes() == baseline["previous"]
    assert all(core.digest(core.HERE / name) == value for name, value in baseline["benchmark"].items())
    assert fast.sources() == plan["source_hashes"]
    delta = {key: results[1][key] - results[0][key] for key in
        ("best_global_miou_pct", "best_sidewalk_iou_pct", "best_boundary_iou_pct")}
    core.json_write(job / "seed42_results.json", {"results": results, "CA_minus_ENet_percentage_points": delta,
        "interpretation": "one paired seed only; seeds43/44 not yet run; not proof of universal benefit or failure"})
    core.json_write(job / "final_verification.json", {"status": "pass", "epochs_per_variant": 100,
        "all_sample_orders_and_final_rng_paired": True, "each_parameter_optimizer_steps": 74400,
        "BN_training_batches": 74400, "best_weights_strict_load": True,
        "old_artifacts_unchanged": True, "sources_unchanged": True, "three_seed_study_complete": False})
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for variant, label in (("A", "ENet"), ("B", "ENet+CA")):
            records = checkpoints[variant]["records"]
            for ax, field, title in zip(axes, ("val_iou_sidewalk_global", "val_boundary_iou_author_fg_mean"),
                    ("Validation sidewalk IoU", "Validation foreground Boundary IoU")):
                ax.plot([r["epoch"] for r in records], [r[field] * 100 for r in records], label=label)
                ax.set(xlabel="Completed epoch", ylabel="Percent", title=title)
                ax.grid(alpha=0.2)
                ax.legend()
        fig.suptitle("Seed 42 paired control; other seeds pending")
        fig.tight_layout()
        fig.savefig(job / "seed42_learning_curves.png", dpi=170)
        plt.close(fig)
    except Exception as error:
        core.json_write(job / "plot_error.json", {"error": str(error)})
    rows = "\n".join(f"| {'ENet' if r['variant']=='A' else 'ENet+CA'} | {r['best_epoch']} | {r['best_global_miou_pct']:.3f} | {r['best_sidewalk_iou_pct']:.3f} | {r['best_boundary_iou_pct']:.3f} |" for r in results)
    if write_markdown:
        with (job / "REPORT.md").open("x", encoding="utf-8") as f:
            f.write(f"# 种子42统一对照完成\n\nA/B 两组各完成100轮，检查通过。只有一个配对种子，43、44尚未运行，不能代替三种子结论。\n\n| 模型 | 最优轮次 | 全局mIoU % | 人行道IoU % | 前景Boundary IoU % |\n|---|---:|---:|---:|---:|\n{rows}\n\n按全局验证mIoU选择各自最优轮，平局保留较早轮次。首轮来自已核验的旧读取执行，后续统一采用多进程读取；[迁移和加载来源](job_plan.json)保留完整来源，旧文件均未覆盖。\n\n[数值结果](seed42_results.json) · [核验记录](final_verification.json)\n\n下一步先审阅本种子结果，再按既定协议完成43、44；新模块仍未启动。\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, type=Path)
    args = parser.parse_args()
    finish(args.job.resolve())
