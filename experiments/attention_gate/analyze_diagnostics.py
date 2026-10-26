"""Reconstruct Step 4 tables from counts, verify saved images, plot evidence."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path

import common_runner as cr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diagnose_attention import csv_write, safe_name
from metrics_v1 import boundary_band, confusion_matrix_numpy


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def optional(value):
    return float(value) if value not in (None, "") else None


def reconstruct(rows):
    c00, c01, c10, c11 = [sum(int(r[key]) for r in rows) for key in ("conf_00", "conf_01", "conf_10", "conf_11")]
    iou_fg = c11 / (c11 + c01 + c10) if c11 + c01 + c10 else 0.0
    iou_bg = c00 / (c00 + c01 + c10) if c00 + c01 + c10 else 0.0
    boundaries = [int(r["boundary_intersection"]) / int(r["boundary_union"]) for r in rows if int(r["boundary_union"]) > 0]
    return {"images": len(rows), "global_miou_pct": (iou_fg + iou_bg) * 50,
        "global_sidewalk_iou_pct": iou_fg * 100,
        "author_boundary_mean_pct": float(np.mean(boundaries)) * 100 if boundaries else None,
        "boundary_valid_images": len(boundaries), "boundary_excluded_images": len(rows) - len(boundaries),
        "conf_00": c00, "conf_01": c01, "conf_10": c10, "conf_11": c11}


def plots(run, stats, deltas, protocol):
    names = protocol["models"]
    grouped = {name: [row for row in stats if row["model"] == name] for name in names}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for ax, field, title in zip(axes, ("gate_mean", "feature_l2_retention"),
                              ("Per-image mean CA gate", "Feature L2 retention after CA")):
        series = [[float(row[field]) for row in grouped[name]] for name in names]
        ax.boxplot(series, labels=names, showmeans=True)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Historical CA weights: fixed train-set diagnostics")
    fig.text(0.5, 0.01, "54 fixed train images; gates below 1 attenuate features by definition, not proof of failure", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(run / "gate_and_feature_distribution.png", dpi=180)
    plt.close(fig)

    conditions = [row["id"] for row in protocol["interventions"] if row["id"] != "original"]
    lookup = {(row["model"], row["condition"]): row for row in deltas}
    fig, axes = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True)
    for ax, key, title in zip(axes, ("delta_global_sidewalk_iou_pp", "delta_boundary_paired_mean_pp"),
                             ("Global sidewalk IoU change (percentage points)", "Paired per-image boundary IoU change (percentage points)")):
        values = np.array([[lookup[(name, cond)][key] for cond in conditions] for name in names], dtype=float)
        limit = max(float(np.nanmax(np.abs(values))), 0.1)
        ax.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
        for i in range(len(names)):
            for j in range(len(conditions)):
                value = values[i, j]
                ax.text(j, i, f"{value:+.2f}" if np.isfinite(value) else "NA", ha="center", va="center",
                        color="white" if abs(value) > limit * 0.55 else "black", fontsize=10)
        ax.set_yticks(range(len(names)), names)
        ax.set_title(title)
    axes[-1].set_xticks(range(len(conditions)), [c.replace("_", "\n", 1) for c in conditions], fontsize=9)
    fig.suptitle("Inference intervention minus ORIGINAL, within each historical checkpoint")
    fig.text(0.5, 0.015, "54 train images | downstream weights and BN fixed | not separately trained ablations", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(run / "intervention_deltas.png", dpi=180)
    plt.close(fig)

    (run / "figures").mkdir()
    for path in sorted((run / "visual_data").glob("*.npz")):
        data = np.load(path)
        fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
        panels = [("Input", "rgb"), ("Target sidewalk", "target"), ("Mean(channel gate), 0..1", "gate_mean_channels"),
                  ("Original CA", "prediction_original"), ("Bypass CA", "prediction_bypass"),
                  ("Energy-matched uniform", "prediction_energy_uniform"),
                  ("Energy-matched shuffle 42", "prediction_energy_shuffle_42"), ("Mixed alpha=0.1, inference only", "prediction_mixed_0_1")]
        for ax, (title, key) in zip(axes.ravel(), panels):
            if key == "gate_mean_channels":
                im = ax.imshow(data[key], cmap="viridis", vmin=0, vmax=1)
                fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
            elif key == "rgb":
                ax.imshow(data[key])
            else:
                ax.imshow(data["rgb"])
                overlay = np.zeros((*data[key].shape, 4))
                overlay[data[key] == 1] = [0, 0.9, 1, 0.65]
                ax.imshow(overlay)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.suptitle(path.stem.replace("_plus_", "+") + " | fixed manifest sample; no success-based selection")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(run / "figures" / (path.stem + ".png"), dpi=150)
        plt.close(fig)
        data.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to(cr.HERE) or (run / "analysis_verification.json").exists():
        raise ValueError("Use an unanalyzed run inside attention_gate")
    status = json.loads((run / "step_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "inference_complete_pending_analysis" and not status["training_started"]
    protocol = json.loads((run / "protocol_snapshot.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "active_manifest.json").read_text(encoding="utf-8"))
    fixed = json.loads((run / "fixed_train_manifest.json").read_text(encoding="utf-8"))
    assert all(row["split"] == "train" for row in fixed)
    assert len(fixed) == 54 and len({row["id"] for row in fixed}) == 54
    assert set(collections.Counter(row["city"] for row in fixed).values()) == {3}
    metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    snapshot = run / "diagnostic_source_exact.py"
    if not snapshot.exists():
        snapshot = run / "diagnostic_snapshot.py"
    assert cr.digest(snapshot) == metadata["diagnostic_sha256"]
    model_checks = json.loads((run / "model_checks.json").read_text(encoding="utf-8"))
    assert all(row["parameters_and_buffers_unchanged"] and row["strict_load"] for row in model_checks)
    assert all(row["max_matched_channel_energy_relative_error"] <= 2e-5 for row in model_checks if row["max_matched_channel_energy_relative_error"] is not None)
    groups, all_stats = {}, []
    ids = {row["id"] for row in manifest}
    count, gt_counts, lookup_rows = 0, {}, {}
    for path in sorted((run / "per_model").glob("*_interventions.csv")):
        for row in read_csv(path):
            key = row["model"], row["condition"]
            groups.setdefault(key, []).append(row)
            conf = [int(row[k]) for k in ("conf_00", "conf_01", "conf_10", "conf_11")]
            assert sum(conf) == 768 * 1024
            fg = conf[2] + conf[3]
            assert fg == int(row["gt_fg_pixels"])
            assert row["sample_id"] in ids
            if row["sample_id"] in gt_counts:
                assert gt_counts[row["sample_id"]] == fg
            gt_counts[row["sample_id"]] = fg
            union = conf[3] + conf[1] + conf[2]
            fg_iou = conf[3] / union if union else 0.0
            assert abs(fg_iou - float(row["iou_fg_image"])) < 1e-12
            bi, bu = int(row["boundary_intersection"]), int(row["boundary_union"])
            assert 0 <= bi <= bu and bu >= 0
            if bu:
                assert abs(bi / bu - float(row["boundary_iou_image"])) < 1e-12
            else:
                assert row["boundary_iou_image"] == ""
            if row["condition"].startswith("energy_"):
                assert float(row["channel_energy_max_relative_error"]) <= 2e-5
            lookup_rows[(row["model"], row["condition"], row["sample_id"])] = row
            count += 1
    assert count == status["predictions"]
    expected = {(name, cond["id"]) for name in protocol["models"] for cond in protocol["interventions"]}
    expected |= {(name, "historical_reference") for name in protocol["reference_models"]}
    assert set(groups) == expected
    for rows in groups.values():
        assert len(rows) == len(manifest) and {row["sample_id"] for row in rows} == ids
    main_table = [{"model": name, "condition": cond, **reconstruct(rows)} for (name, cond), rows in groups.items()]
    main_lookup = {(row["model"], row["condition"]): row for row in main_table}
    deltas = []
    for name in protocol["models"]:
        original = main_lookup[name, "original"]
        for item in protocol["interventions"]:
            condition = item["id"]
            if condition == "original":
                continue
            comparison = main_lookup[name, condition]
            differences, boundary_differences, changed = [], [], []
            for sample in manifest:
                base = lookup_rows[name, "original", sample["id"]]
                row = lookup_rows[name, condition, sample["id"]]
                differences.append(float(row["iou_fg_image"]) - float(base["iou_fg_image"]))
                bv, cv = optional(base["boundary_iou_image"]), optional(row["boundary_iou_image"])
                if bv is not None and cv is not None:
                    boundary_differences.append(cv - bv)
                changed.append(float(row["changed_prediction_pixels_fraction"]))
            deltas.append({"model": name, "condition": condition,
                "delta_global_miou_pp": comparison["global_miou_pct"] - original["global_miou_pct"],
                "delta_global_sidewalk_iou_pp": comparison["global_sidewalk_iou_pct"] - original["global_sidewalk_iou_pct"],
                "delta_boundary_paired_mean_pp": float(np.mean(boundary_differences)) * 100 if boundary_differences else None,
                "boundary_paired_valid_images": len(boundary_differences),
                "boundary_original_valid_images": original["boundary_valid_images"],
                "boundary_intervention_valid_images": comparison["boundary_valid_images"],
                "fg_image_iou_delta_mean_pp": float(np.mean(differences)) * 100,
                "fg_image_iou_delta_std_pp": float(np.std(differences, ddof=1)) * 100,
                "fg_images_improved": sum(d > 1e-12 for d in differences), "fg_images_worsened": sum(d < -1e-12 for d in differences),
                "fg_images_unchanged": sum(abs(d) <= 1e-12 for d in differences),
                "changed_prediction_pixels_mean_pct": float(np.mean(changed)) * 100})
    for path in sorted((run / "per_model").glob("*_gate_statistics.csv")):
        rows = read_csv(path)
        assert len(rows) == len(manifest) and {r["sample_id"] for r in rows} == ids
        all_stats.extend(rows)
    gate_summary = []
    fields = [key for key in all_stats[0] if key not in ("model", "sample_id", "city")]
    for name in protocol["models"]:
        rows = [r for r in all_stats if r["model"] == name]
        summary = {"model": name, "images": len(rows)}
        for field in fields:
            values = [float(r[field]) for r in rows if r[field] != ""]
            summary[field + "_mean"] = float(np.mean(values)) if values else None
            summary[field + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            summary[field + "_valid_images"] = len(values)
        gate_summary.append(summary)
    verified_predictions = 0
    for path in sorted((run / "visual_data").glob("*.npz")):
        data = np.load(path)
        name, position = path.stem.rsplit("_", 1)
        model = next(name0 for name0 in protocol["models"] if safe_name(name0) == name)
        sample = manifest[int(position)]
        target = data["target"]
        assert target.shape == (768, 1024)
        for field in data.files:
            if not field.startswith("prediction_"):
                continue
            condition = field[len("prediction_"):]
            row = lookup_rows[model, condition, sample["id"]]
            conf = confusion_matrix_numpy(data[field], target)
            assert conf.ravel().tolist() == [int(row[k]) for k in ("conf_00", "conf_01", "conf_10", "conf_11")]
            pb, gb = boundary_band(data[field] == 1), boundary_band(target == 1)
            assert int(np.count_nonzero(pb & gb)) == int(row["boundary_intersection"])
            assert int(np.count_nonzero(pb | gb)) == int(row["boundary_union"])
            verified_predictions += 1
        statistic = next(r for r in all_stats if r["model"] == model and r["sample_id"] == sample["id"])
        assert abs(float(data["gate_mean_channels"].mean()) - float(statistic["gate_mean"])) < 1e-7
        legacy = data["vertical_mean_channels"][:, None] * data["horizontal_mean_channels"][None, :]
        error = np.abs(data["gate_mean_channels"] - legacy).mean()
        assert abs(float(error) - float(statistic["legacy_heatmap_mean_absolute_error"])) < 1e-7
        data.close()
    if status["mode"] == "full":
        smoke = run.parent / "smoke_001"
        smoke_rows = []
        for path in (smoke / "per_model").glob("*_interventions.csv"):
            smoke_rows.extend(read_csv(path))
        for row in smoke_rows:
            full_row = lookup_rows[row["model"], row["condition"], row["sample_id"]]
            assert row == full_row, "Smoke/full outputs differ on shared samples"
    csv_write(run / "global_intervention_table.csv", main_table)
    csv_write(run / "paired_intervention_deltas.csv", deltas)
    csv_write(run / "gate_feature_summary.csv", gate_summary)
    cr.json_write(run / "analysis_tables.json", {"global": main_table, "paired_deltas": deltas, "gate_summary": gate_summary})
    plots(run, all_stats, deltas, protocol)
    cr.json_write(run / "analysis_verification.json", {"status": "pass", "prediction_rows": count,
        "global_rows": len(main_table), "paired_deltas": len(deltas), "gate_statistic_rows": len(all_stats),
        "saved_predictions_recomputed": verified_predictions, "smoke_full_common_rows_exact": len(smoke_rows) if status["mode"] == "full" else None,
        "global_reconstruction_from_confusion": True, "boundary_paired_valid_counts_reported": True,
        "wrapper_checks": True, "diagnostic_snapshot_hash_matches": True})
    print(json.dumps({"verification": "pass", "rows": count, "global": len(main_table), "deltas": len(deltas)}))
    print(json.dumps({"gate": [{"model": row["model"], "mean": row["gate_mean_mean"],
          "l2": row["feature_l2_retention_mean"], "near_zero_fraction": row["gate_fraction_below_005_mean"]} for row in gate_summary],
          "selected_deltas": [row for row in deltas if row["condition"] in ("bypass", "energy_uniform", "energy_shuffle_42", "energy_shuffle_43", "mixed_0_1")]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
