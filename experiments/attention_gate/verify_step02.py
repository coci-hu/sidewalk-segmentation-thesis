"""Reconstruct exported scores from per-image counts without model inference."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def same(left, right):
    return math.isclose(float(left), float(right), rel_tol=0, abs_tol=1e-10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = Path(args.run).resolve()
    if not run.is_relative_to(HERE):
        raise RuntimeError("Run must be inside the experiment directory")
    status = json.loads((run / "step_status.json").read_text(encoding="utf-8"))
    metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    integrity = json.loads((run / "integrity_after.json").read_text(encoding="utf-8"))
    assert status["mode"] == "full" and status["status"] == "complete"
    assert status["models_completed"] == 15 and status["total_predictions"] == 7500
    assert integrity["before_after_sha256_identical"] and integrity["input_metadata_unchanged"]
    assert hashlib.sha256((run / "evaluator_snapshot.py").read_bytes()).hexdigest() == metadata["evaluator_sha256"]
    summaries = json.loads((run / "main_table.json").read_text(encoding="utf-8"))
    table = read_csv(run / "main_table.csv")
    cities = read_csv(run / "per_city_table.csv")
    manifest = json.loads((run / "sample_manifest.json").read_text(encoding="utf-8"))
    ids = [item["sample_id"] for item in manifest]
    assert len(ids) == len(set(ids)) == 500
    assert len(summaries) == len(table) == 15 and len(cities) == 45
    assert {row["model"] for row in summaries} == set(metadata["expected_models"])
    city_counts = {city: sum(item["city"] == city for item in manifest) for city in {item["city"] for item in manifest}}
    assert city_counts == {"frankfurt": 267, "lindau": 59, "munster": 174}

    checks, model_rows = [], {}
    ground_truth_reference = None
    for summary in summaries:
        name = summary["model"]
        safe = name.replace("+", "_plus_").replace("-", "_")
        rows = read_csv(run / "per_model" / f"{safe}_per_image.csv")
        assert len(rows) == 500 and [row["sample_id"] for row in rows] == ids
        model_rows[name] = rows
        gt = []
        for row in rows:
            counts = [int(row[key]) for key in ("conf_00", "conf_01", "conf_10", "conf_11")]
            assert sum(counts) == 768 * 1024
            assert int(row["gt_fg_pixels"]) == counts[2] + counts[3]
            assert int(row["pred_fg_pixels"]) == counts[1] + counts[3]
            gt.append(int(row["gt_fg_pixels"]))
            inter, union = int(row["boundary_intersection_fg"]), int(row["boundary_union_fg"])
            assert 0 <= inter <= union
            if union:
                assert same(row["boundary_iou_author_fg"], inter / union)
            else:
                assert row["boundary_iou_author_fg"] == ""
        if ground_truth_reference is None:
            ground_truth_reference = gt
        else:
            assert gt == ground_truth_reference
        c00, c01, c10, c11 = [sum(int(row[key]) for row in rows) for key in ("conf_00", "conf_01", "conf_10", "conf_11")]
        iou_bg = c00 / (c00 + c01 + c10)
        iou_fg = c11 / (c11 + c01 + c10)
        boundary = [float(row["boundary_iou_author_fg"]) for row in rows if int(row["boundary_union_fg"]) > 0]
        assert same(summary["miou_global_pct"], (iou_bg + iou_fg) * 50)
        assert same(summary["iou_sidewalk_global_pct"], iou_fg * 100)
        assert same(summary["boundary_iou_author_fg_pct"], statistics.mean(boundary) * 100)
        assert summary["boundary_valid_images"] == len(boundary)
        assert summary["boundary_excluded_empty_pairs"] == 500 - len(boundary)
        csv_summary = next(row for row in table if row["model"] == name)
        for key in ("miou_global_pct", "iou_sidewalk_global_pct", "boundary_iou_author_fg_pct"):
            assert same(csv_summary[key], summary[key])
        for city in city_counts:
            selected = [row for row in rows if row["city"] == city]
            city_summary = next(row for row in cities if row["model"] == name and row["city"] == city)
            assert len(selected) == int(city_summary["images"]) == city_counts[city]
            a, b, c, d = [sum(int(row[key]) for row in selected) for key in ("conf_00", "conf_01", "conf_10", "conf_11")]
            assert same(city_summary["miou_global_pct"], (a / (a + b + c) + d / (d + b + c)) * 50)
        checks.append({"model": name, "per_image_rows": 500, "global_and_city_reconstruction": "pass"})

    cache_comparisons = []
    initial_run = run.parent / "run_001"
    for name in ("ENet", "ENet+CA"):
        safe = name.replace("+", "_plus_").replace("-", "_")
        path = initial_run / "per_model" / f"{safe}_per_image.csv"
        if not path.exists() or initial_run == run:
            continue
        old_rows = read_csv(path)
        assert len(old_rows) == 500
        keys = ("conf_00", "conf_01", "conf_10", "conf_11", "boundary_intersection_fg", "boundary_union_fg")
        differing = sum(any(a[key] != b[key] for key in keys) for a, b in zip(old_rows, model_rows[name]))
        cache_comparisons.append({"model": name, "images_compared": 500, "images_with_count_difference": differing})
        assert differing == 0, "Cached input evaluation differs from uncached evaluation"

    lookup = {row["model"]: row for row in summaries}
    pairs = [("ENet+CA", "ENet"), ("ENet+BA", "ENet"), ("ENet-DBA", "ENet"),
             ("ENet+L-ASPP", "ENet"), ("ENet-CBR", "ENet"),
             ("SBNet+CA", "SBNet"), ("SBNet+L-ASPP", "SBNet"),
             ("SBNet-CBR", "SBNet"), ("SBNet-CBR", "SBNet+L-ASPP")]
    differences = [{"model": a, "reference": b,
                    "miou_delta_pp": lookup[a]["miou_global_pct"] - lookup[b]["miou_global_pct"],
                    "sidewalk_iou_delta_pp": lookup[a]["iou_sidewalk_global_pct"] - lookup[b]["iou_sidewalk_global_pct"],
                    "boundary_iou_delta_pp": lookup[a]["boundary_iou_author_fg_pct"] - lookup[b]["boundary_iou_author_fg_pct"]} for a, b in pairs]
    with (run / "comparison_deltas.csv").open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(differences[0]))
        writer.writeheader()
        writer.writerows(differences)
    verification = {"status": "pass", "models": checks, "predictions_verified": 7500,
                    "city_counts": city_counts, "cache_comparisons": cache_comparisons,
                    "old_files_sha256_unchanged": integrity["before_after_sha256_identical"],
                    "evaluator_snapshot_sha256_matches": True}
    with (run / "analysis_verification.json").open("x", encoding="utf-8") as f:
        json.dump(verification, f, ensure_ascii=False, indent=2)
        f.write("\n")
    with (run / "comparison_deltas.json").open("x", encoding="utf-8") as f:
        json.dump(differences, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({"status": "pass", "predictions_verified": 7500, "cache_comparisons": cache_comparisons, "differences": differences}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
