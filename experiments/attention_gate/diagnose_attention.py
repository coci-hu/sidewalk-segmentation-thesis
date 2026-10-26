"""Step 4: fixed TRAIN samples, historical weights, inference-only interventions."""
import argparse
import csv
import gc
import hashlib
import json
import time
import traceback
from pathlib import Path

import common_runner as cr
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
from evaluate_existing import load_strict, protected_hashes, timestamp
import eval_common as ec
from metrics_v1 import boundary_band, confusion_matrix_numpy, scores_from_confusion

HERE = cr.HERE
PROJECT = cr.PROJECT


def csv_write(path, rows):
    with Path(path).open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def snapshot_previous():
    result = {}
    for relative in ("step_02_2026-10-14/run_002", "step_03_2026-10-14/run_003"):
        for path in (HERE / relative).rglob("*"):
            if path.is_file():
                result[str(path.relative_to(HERE))] = cr.digest(path)
    return result


def safe_ratio(numerator, denominator):
    return torch.where(denominator > 1e-12, numerator / denominator.clamp_min(1e-12), torch.zeros_like(denominator))


def norm_channels(value):
    return torch.linalg.vector_norm(value, dim=(2, 3), keepdim=True)


def extract_gates(module, x):
    height, width = x.shape[-2:]
    horizontal = x.mean(2, keepdim=True)
    vertical = x.mean(3, keepdim=True).permute(0, 1, 3, 2)
    joint = module.act(module.bn1(module.conv1(torch.cat((horizontal, vertical), dim=3))))
    horizontal, vertical = torch.split(joint, [width, height], dim=3)
    horizontal = module.conv_h(horizontal).sigmoid()
    vertical = module.conv_w(vertical.permute(0, 1, 3, 2)).sigmoid()
    return horizontal, vertical


class InterventionCA(nn.Module):
    def __init__(self, module, tolerance):
        super().__init__()
        self.original_module = module
        self.tolerance = tolerance
        self.mode = "original"
        self.sample_id = None
        self.feature = None
        self.calls = 0
        self.max_energy_error = 0.0

    def begin(self, sample_id):
        self.sample_id = sample_id
        self.feature = None
        self.calls = 0

    def shuffled_gate(self, horizontal, vertical, seed):
        stable = int(hashlib.sha256(self.sample_id.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed + stable)
        columns = torch.as_tensor(rng.permutation(horizontal.shape[3]), device=horizontal.device)
        rows = torch.as_tensor(rng.permutation(vertical.shape[2]), device=vertical.device)
        return vertical.index_select(2, rows) * horizontal.index_select(3, columns)

    def forward(self, x):
        self.calls += 1
        horizontal, vertical = extract_gates(self.original_module, x)
        gate = vertical * horizontal
        original = self.original_module(x)
        if not torch.equal(original, x * vertical * horizontal):
            raise RuntimeError("Gate extraction does not reproduce the current CA")
        if self.feature is None:
            self.feature = x.detach().clone()
            self.original_output = original.detach().clone()
            self.gate = gate.detach().clone()
            self.horizontal = horizontal.detach().clone()
            self.vertical = vertical.detach().clone()
        elif not torch.equal(x, self.feature):
            raise RuntimeError("Encoder features changed across interventions")
        mode = self.mode
        if mode == "original":
            result = original
        elif mode == "bypass":
            result = x
        elif mode == "global_mean":
            result = x * gate.mean()
        elif mode == "channel_mean":
            result = x * gate.mean(dim=(2, 3), keepdim=True)
        elif mode == "energy_uniform":
            result = x * safe_ratio(norm_channels(original), norm_channels(x))
        elif mode.startswith(("shuffle_", "energy_shuffle_")):
            seed = int(mode.rsplit("_", 1)[1])
            shuffled = self.shuffled_gate(horizontal, vertical, seed)
            if not torch.allclose(gate.mean(dim=(2, 3)), shuffled.mean(dim=(2, 3)), atol=2e-7, rtol=2e-6):
                raise RuntimeError("Shuffling changed gate channel means")
            result = x * shuffled
            if mode.startswith("energy_"):
                result = result * safe_ratio(norm_channels(original), norm_channels(result))
        elif mode == "mixed_0_1":
            result = x * (0.9 + 0.1 * gate)
        else:
            raise ValueError(mode)
        reference_norm = norm_channels(original)
        current_norm = norm_channels(result)
        error = safe_ratio((current_norm - reference_norm).abs(), reference_norm).max().item()
        self.max_energy_error = error
        if mode.startswith("energy_") and error > self.tolerance:
            raise RuntimeError(f"Per-channel energy mismatch {error}")
        self.output_retention = safe_ratio(torch.linalg.vector_norm(result), torch.linalg.vector_norm(x)).item()
        return result


def region_value(value, weight):
    total = weight.sum().item()
    return (value * weight).sum().item() / total if total > 0 else None


def feature_statistics(wrapper, sample, protocol):
    x, y, gate = wrapper.feature, wrapper.original_output, wrapper.gate
    flat = gate.cpu().numpy().ravel()
    quantiles = np.quantile(flat, protocol["gate_statistics"]["quantiles"])
    channel_gate_mean = gate.mean(dim=(2, 3))
    spatial = gate.mean(dim=1, keepdim=True)
    x_energy, y_energy = x.square().sum(1, keepdim=True), y.square().sum(1, keepdim=True)
    norm_x, norm_y = torch.linalg.vector_norm(x), torch.linalg.vector_norm(y)
    row = {"gate_mean": float(flat.mean()), "gate_std": float(flat.std()),
           "gate_min": float(flat.min()), "gate_max": float(flat.max()),
           "gate_fraction_below_005": float(np.mean(flat < 0.05)),
           "gate_fraction_above_095": float(np.mean(flat > 0.95)),
           "horizontal_fraction_below_005": float((wrapper.horizontal < 0.05).float().mean().item()),
           "horizontal_fraction_above_095": float((wrapper.horizontal > 0.95).float().mean().item()),
           "vertical_fraction_below_005": float((wrapper.vertical < 0.05).float().mean().item()),
           "vertical_fraction_above_095": float((wrapper.vertical > 0.95).float().mean().item()),
           "gate_spatial_std_per_channel_mean": gate.std(dim=(2, 3), unbiased=False).mean().item(),
           "gate_channel_mean_std": channel_gate_mean.std(unbiased=False).item(),
           "horizontal_spatial_std_mean": wrapper.horizontal.std(3, unbiased=False).mean().item(),
           "vertical_spatial_std_mean": wrapper.vertical.std(2, unbiased=False).mean().item(),
           "feature_l2_retention": safe_ratio(norm_y, norm_x).item(),
           "feature_cosine_before_after": safe_ratio((x * y).sum(), norm_x * norm_y).item(),
           "feature_channels": x.shape[1], "feature_height": x.shape[2], "feature_width": x.shape[3]}
    for q, value in zip(protocol["gate_statistics"]["quantiles"], quantiles):
        row[f"gate_q{int(q * 100):02d}"] = float(value)
    for name, mask in (("foreground", sample["target"] == 1), ("background", sample["target"] == 0),
                       ("boundary", sample["boundary"])):
        weight = F.interpolate(torch.as_tensor(mask.astype(np.float32), device=x.device)[None, None],
                               size=x.shape[-2:], mode="area")
        row[f"gate_mean_{name}"] = region_value(spatial, weight)
        numerator, denominator = (y_energy * weight).sum(), (x_energy * weight).sum()
        row[f"l2_retention_{name}"] = safe_ratio(numerator, denominator).sqrt().item() if denominator.item() > 0 else None
    legacy_display = wrapper.vertical.mean(1, keepdim=True) * wrapper.horizontal.mean(1, keepdim=True)
    row["legacy_heatmap_mean_absolute_error"] = (spatial - legacy_display).abs().mean().item()
    row["legacy_heatmap_max_absolute_error"] = (spatial - legacy_display).abs().max().item()
    return row


def select_manifest(dataset, protocol):
    cities = {}
    for index, (image, label) in enumerate(dataset.pairs):
        cities.setdefault(image.parent.name, []).append(index)
    if len(cities) != protocol["selection"]["expected_cities"]:
        raise RuntimeError("Unexpected training city count")
    rng = np.random.default_rng(protocol["selection"]["seed"])
    rows = []
    for city, indices in sorted(cities.items()):
        for index in sorted(rng.choice(indices, protocol["selection"]["per_city"], replace=False).tolist()):
            image, label = dataset.pairs[index]
            rows.append({"index": index, "id": image.name, "city": city, "split": "train",
                "image": str(image.relative_to(PROJECT)), "label": str(label.relative_to(PROJECT)),
                "image_sha256": cr.digest(image), "label_sha256": cr.digest(label)})
    if len(rows) != protocol["selection"]["expected_images"]:
        raise RuntimeError("Unexpected fixed sample count")
    return rows


def prepare_samples(dataset, manifest, ratio):
    samples = []
    for row in manifest:
        index = row["index"]
        image_path, label_path = dataset.pairs[index]
        with Image.open(image_path) as image:
            rgb = np.array(image.convert("RGB").resize((1024, 768), Image.BILINEAR))
        with Image.open(label_path) as label:
            binary = (np.array(label) == 8).astype(np.uint8)
        target = np.array(Image.fromarray(binary).resize((1024, 768), Image.NEAREST))
        tensor = dataset.transform(Image.fromarray(rgb))
        old_tensor, old_mask = dataset[index]
        if not torch.equal(tensor, old_tensor) or not np.array_equal(target, old_mask.numpy()):
            raise RuntimeError("Cached input differs from the fixed preprocessing")
        samples.append({**row, "rgb": rgb, "target": target, "boundary": boundary_band(target == 1, ratio)})
    return samples


def metric_row(name, condition, sample, logits, original_prediction=None):
    if list(logits.shape) != [1, 2, 768, 1024] or not bool(torch.isfinite(logits).all()):
        raise RuntimeError("Invalid output")
    pred = logits.argmax(1)[0].to(torch.uint8).cpu().numpy()
    confusion = confusion_matrix_numpy(pred, sample["target"])
    predicted_boundary = boundary_band(pred == 1)
    bi = int(np.count_nonzero(predicted_boundary & sample["boundary"]))
    bu = int(np.count_nonzero(predicted_boundary | sample["boundary"]))
    scores = scores_from_confusion(confusion)
    return {"model": name, "condition": condition, "index": sample["index"], "sample_id": sample["id"], "city": sample["city"],
        "conf_00": int(confusion[0, 0]), "conf_01": int(confusion[0, 1]), "conf_10": int(confusion[1, 0]), "conf_11": int(confusion[1, 1]),
        "iou_fg_image": scores["iou"][1], "miou_image": scores["miou"],
        "boundary_intersection": bi, "boundary_union": bu, "boundary_iou_image": bi / bu if bu else None,
        "gt_fg_pixels": int(sample["target"].sum()), "pred_fg_pixels": int(pred.sum()),
        "changed_prediction_pixels_fraction": float(np.mean(pred != original_prediction)) if original_prediction is not None else 0.0}, pred


def safe_name(name):
    return name.replace("+", "_plus_").replace("-", "_")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(HERE) or output.exists() or output == HERE:
        raise ValueError("Use a new attention_gate output subdirectory")
    output.mkdir(parents=True)
    started = time.perf_counter()
    def log(message):
        line = timestamp() + " " + message
        print(line, flush=True)
        with (output / "console.log").open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    try:
        previous_status = json.loads((HERE / "step_03_2026-10-14/run_003/step_status.json").read_text(encoding="utf-8"))
        if previous_status["status"] != "complete":
            raise RuntimeError("Step 3 incomplete")
        cr.configure(42)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        protocol = json.loads((HERE / "diagnostic_protocol_v1.json").read_text(encoding="utf-8"))
        before, previous = protected_hashes(), snapshot_previous()
        cr.json_write(output / "protected_files_before.json", before)
        cr.json_write(output / "previous_steps_before.json", previous)
        cr.json_write(output / "protocol_snapshot.json", protocol)
        with (output / "diagnostic_snapshot.py").open("xb") as f:
            f.write(Path(__file__).read_bytes())
        cr.json_write(output / "run_metadata.json", {"mode": args.mode, "started": timestamp(), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0), "training": False, "split": "train",
            "diagnostic_sha256": cr.digest(__file__), "protocol_sha256": cr.digest(HERE / "diagnostic_protocol_v1.json"),
            "source_hashes": {str(path.relative_to(PROJECT)): cr.digest(path) for path in
                [HERE / "metrics_v1.py", HERE / "common_runner.py", cr.CODE / "eval_common.py", cr.CODE / "components/ca_attention.py"]}})
        dataset = cr.SidewalkDataset(cr.load_protocol(), "train")
        fixed = select_manifest(dataset, protocol)
        cr.json_write(output / "fixed_train_manifest.json", fixed)
        active = fixed[:3] if args.mode == "smoke" else fixed
        cr.json_write(output / "active_manifest.json", active)
        samples = prepare_samples(dataset, active, 0.02)
        log(f"Fixed train sample manifest: {len(fixed)} images, active={len(active)}; no output-based selection")
        output.joinpath("per_model").mkdir()
        output.joinpath("visual_data").mkdir()
        names = protocol["reference_models"] + protocol["models"]
        checks = []
        predictions = 0
        for name in names:
            model = load_strict(ec.MODEL_SPECS[name])
            source_state_hash = cr.state_digest(model.state_dict())
            module = getattr(model, "ca", None)
            wrapper = InterventionCA(module, protocol["controls"]["energy_matching_max_channel_relative_error_tolerance"]).eval() if module is not None else None
            rows, stats = [], []
            conditions = [row["id"] for row in protocol["interventions"]] if wrapper else ["historical_reference"]
            log(f"Model {name}, {len(samples)} samples, {len(conditions)} conditions")
            energy_max = 0.0
            with torch.inference_mode():
                for position, sample in enumerate(samples):
                    tensor = dataset.transform(Image.fromarray(sample["rgb"]))[None].cuda()
                    reference_logits = ec.model_forward(model, tensor)
                    if wrapper:
                        wrapper.begin(sample["id"])
                        model.ca = wrapper
                    if any(item.training for item in model.modules()):
                        raise RuntimeError("A module is unexpectedly in training mode")
                    original_prediction = None
                    visual = {"rgb": sample["rgb"], "target": sample["target"]}
                    for condition in conditions:
                        if wrapper:
                            wrapper.mode = condition
                            logits = ec.model_forward(model, tensor)
                            if condition == "original":
                                if not torch.equal(logits, reference_logits):
                                    raise RuntimeError("Diagnostic wrapper changes original logits")
                                info = feature_statistics(wrapper, sample, protocol)
                                stats.append({"model": name, "sample_id": sample["id"], "city": sample["city"], **info})
                                visual["gate_mean_channels"] = wrapper.gate.mean(1)[0].cpu().numpy()
                                visual["horizontal_mean_channels"] = wrapper.horizontal.mean(1)[0, 0].cpu().numpy()
                                visual["vertical_mean_channels"] = wrapper.vertical.mean(1)[0, :, 0].cpu().numpy()
                            if condition.startswith("energy_"):
                                energy_max = max(energy_max, wrapper.max_energy_error)
                        else:
                            logits = reference_logits
                        row, pred = metric_row(name, condition, sample, logits, original_prediction)
                        if original_prediction is None:
                            original_prediction = pred.copy()
                        row["feature_l2_retention_under_intervention"] = wrapper.output_retention if wrapper else None
                        row["channel_energy_max_relative_error"] = wrapper.max_energy_error if wrapper else None
                        rows.append(row)
                        predictions += 1
                        if condition in ("original", "bypass", "energy_uniform", "energy_shuffle_42", "mixed_0_1"):
                            visual["prediction_" + condition] = pred
                        del logits
                    if wrapper:
                        if wrapper.calls != len(conditions):
                            raise RuntimeError("Unexpected CA call count")
                        model.ca = module
                        if position in (0, 18, 36) or args.mode == "smoke" and position == 0:
                            np.savez_compressed(output / "visual_data" / f"{safe_name(name)}_{position:02d}.npz", **visual)
                    del tensor, reference_logits, visual
                    if (position + 1) % 9 == 0 or position + 1 == len(samples):
                        log(f"  {name}: {position + 1}/{len(samples)}, predictions={predictions}")
                        cr.json_write(output / "progress.json", {"status": "running", "model": name,
                            "images_done": position + 1, "predictions": predictions, "elapsed_seconds": time.perf_counter() - started}, exclusive=False)
            if cr.state_digest(model.state_dict()) != source_state_hash:
                raise RuntimeError("Model parameters or buffers mutated")
            csv_write(output / "per_model" / f"{safe_name(name)}_interventions.csv", rows)
            if stats:
                csv_write(output / "per_model" / f"{safe_name(name)}_gate_statistics.csv", stats)
            checks.append({"model": name, "strict_load": True, "weight": ec.MODEL_SPECS[name]["weight"],
                "weight_sha256": cr.digest(PROJECT / "weights" / ec.MODEL_SPECS[name]["weight"]),
                "parameters_and_buffers_unchanged": True, "original_logits_bitwise_equal": True if wrapper else None,
                "features_identical_across_interventions": True if wrapper else None,
                "max_matched_channel_energy_relative_error": energy_max if wrapper else None,
                "images": len(samples), "conditions": len(conditions)})
            del model, wrapper, module, rows, stats
            gc.collect()
            torch.cuda.empty_cache()
        cr.json_write(output / "model_checks.json", checks)
        after = protected_hashes()
        previous_after = snapshot_previous()
        if before != after or previous != previous_after:
            raise RuntimeError("Historical files or previous step outputs changed")
        for row in fixed:
            if cr.digest(PROJECT / row["image"]) != row["image_sha256"] or cr.digest(PROJECT / row["label"]) != row["label_sha256"]:
                raise RuntimeError("Selected source data changed")
        cr.json_write(output / "integrity_after.json", {"original_files": len(before), "original_sha256_identical": True,
            "previous_step_files": len(previous), "previous_step_sha256_identical": True,
            "fixed_images_and_labels": len(fixed) * 2, "data_sha256_identical": True})
        status = {"step": 4, "status": "inference_complete_pending_analysis", "mode": args.mode,
            "images": len(samples), "models": len(names), "CA_models": 4, "conditions_per_CA_model": len(protocol["interventions"]),
            "predictions": predictions, "training_started": False, "wall_seconds": time.perf_counter() - started, "finished": timestamp()}
        cr.json_write(output / "step_status.json", status)
        cr.json_write(output / "progress.json", status, exclusive=False)
        log("Inference finished; independent reconstruction and report remain")
    except Exception:
        cr.json_write(output / "failure.json", {"traceback": traceback.format_exc(), "at": timestamp()}, exclusive=False)
        raise


if __name__ == "__main__":
    main()
