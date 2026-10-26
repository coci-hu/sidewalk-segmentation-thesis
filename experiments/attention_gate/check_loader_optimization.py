"""Compare workers 1/2/4 without changing scientific training conditions."""
import argparse
import copy
import ctypes
import gc
import json
import time
import traceback
from pathlib import Path
import common_runner as cr
import numpy as np
import torch
from torch.utils.data import Subset
from check_step03 import equal_tree
from evaluate_existing import protected_hashes, available_memory, timestamp
from run_step05_benchmark import previous_hashes


def tensor_hash(tensor):
    import hashlib
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def load_checkpoint(variant):
    return torch.load(cr.HERE / f"step_05_2026-10-14/benchmark_001/{variant}_seed42_epoch1/checkpoints/epoch_001.pth",
                      map_location="cpu", weights_only=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(cr.HERE) or output.exists():
        raise ValueError("Use a new experiment directory")
    output.mkdir(parents=True)
    before, previous = protected_hashes(), previous_hashes()
    benchmark = cr.HERE / "step_05_2026-10-14/benchmark_001"
    prior_benchmark = {str(p.relative_to(cr.HERE)): cr.digest(p) for p in benchmark.rglob("*") if p.is_file()}
    cr.json_write(output / "protected_before.json", {"original": before, "previous": previous, "benchmark": prior_benchmark})
    protocol = cr.load_protocol()
    train, val = cr.SidewalkDataset(protocol, "train"), cr.SidewalkDataset(protocol, "val")
    cities = {}
    for index, (image, _) in enumerate(train.pairs):
        cities.setdefault(image.parent.name, []).append(index)
    sample_ids = [index for values in cities.values() for index in (values[0], values[-1])]
    performance_ids = np.random.default_rng(20261014).choice(len(train), 192, replace=False).tolist()
    val_first = {}
    for i, (image, _) in enumerate(val.pairs):
        val_first.setdefault(image.parent.name, i)
    manifests = {"input_train_indices": sample_ids, "input_val_indices": list(val_first.values()),
                 "performance_train_indices": performance_ids, "toy_train_indices": [0, 1487, 2974, 1, 1488, 2973, 2]}
    cr.json_write(output / "sample_plan.json", manifests)
    profiles, input_reference, toy_reference = [], None, {}
    free_start = available_memory()
    for workers in (1, 2, 4):
        if workers == 4 and available_memory() < 2 * 2 ** 30:
            profiles.append({"workers": 4, "skipped": "less than 2 GiB available RAM"})
            continue
        p = copy.deepcopy(protocol)
        p["formal_training"]["num_workers"] = workers
        cr.configure(42)
        gens = cr.generators(42)
        loader, validation, sampler = cr.make_loaders(Subset(train, sample_ids), Subset(val, list(val_first.values())), p, gens)
        signatures = {"train": [], "val": []}
        for split, batches in (("train", loader), ("val", validation)):
            for images, masks in batches:
                for image, mask in zip(images, masks):
                    signatures[split].append({"image_sha256": tensor_hash(image), "mask_sha256": tensor_hash(mask)})
        signatures["train_order"] = list(sampler.last_order)
        if input_reference is None:
            input_reference = signatures
        else:
            assert signatures == input_reference, "Input values or ordering changed"
        cr.json_write(output / f"workers_{workers}_input_signatures.json", signatures)
        print(f"workers={workers}: 36 train + 3 val tensor signatures match", flush=True)
        gens = cr.generators(42)
        loader, _, sampler = cr.make_loaders(Subset(train, performance_ids), Subset(val, [0]), p, gens)
        start = time.perf_counter()
        arrivals, counts = [], []
        for images, masks in loader:
            arrivals.append(time.perf_counter() - start)
            counts.append(len(images))
        total = time.perf_counter() - start
        # Ignore first 8 batches for startup; measure delivery of remaining 160 images.
        steady_seconds = arrivals[-1] - arrivals[7]
        steady_images = sum(counts[8:])
        profile = {"workers": workers, "input_and_order_equal": True, "images": sum(counts),
            "loader_total_seconds": total, "steady_images": steady_images, "steady_seconds": steady_seconds,
            "steady_images_per_second": steady_images / steady_seconds, "available_ram_bytes_after": available_memory()}
        print(f"workers={workers}: data throughput={profile['steady_images_per_second']:.2f} images/s", flush=True)
        for variant in ("A", "B"):
            checkpoint = load_checkpoint(variant)
            cr.configure(42)
            model = cr.build_paired_model(variant, 42).cuda()
            optimizer, scheduler = cr.optimizer_scheduler(model, p)
            model.load_state_dict(checkpoint["model_state"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            gens = cr.generators(42)
            small_train = Subset(train, manifests["toy_train_indices"])
            small_val = Subset(val, list(val_first.values()))
            loader, validation, sampler = cr.make_loaders(small_train, small_val, p, gens)
            cr.restore_rng(checkpoint["rng"], gens)
            # Two epochs cover restart of nonpersistent workers and the final batch of 3.
            records = []
            for epoch in range(2):
                row = cr.train_epoch(model, loader, optimizer, p, torch.device("cuda"))
                row["order"] = list(sampler.last_order)
                row.update(cr.validate(model, validation, p, torch.device("cuda")))
                scheduler.step()
                records.append(row)
            state = {"model_state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "optimizer_state": copy.deepcopy(optimizer.state_dict()), "scheduler_state": scheduler.state_dict(),
                "rng": cr.capture_rng(gens), "records": records}
            # Canonical CPU optimizer state, so comparison does not retain GPU tensors.
            for entry in state["optimizer_state"]["state"].values():
                for key, value in entry.items():
                    if isinstance(value, torch.Tensor):
                        entry[key] = value.cpu().clone()
            if workers == 1:
                toy_reference[variant] = state
            else:
                assert equal_tree(state, toy_reference[variant]), f"Training/RNG differs for {variant}, workers={workers}"
            print(f"workers={workers}, {variant}: two toy epochs exact match", flush=True)
            profile[variant + "_two_toy_epochs_model_optimizer_scheduler_rng_metrics_exact"] = True
            del model, optimizer, scheduler, checkpoint, state, loader, validation
            gc.collect()
            torch.cuda.empty_cache()
        profiles.append(profile)
        cr.json_write(output / "profiles.json", profiles, exclusive=False)
    usable = [p for p in profiles if not p.get("skipped")]
    best = max(usable, key=lambda p: p["steady_images_per_second"])
    # Use the smaller process count when its throughput is within 5% of the best.
    selected = min((p for p in usable if p["steady_images_per_second"] >= best["steady_images_per_second"] * 0.95), key=lambda p: p["workers"])
    assert selected["workers"] > 1, "No beneficial data-loader optimization found"
    assert before == protected_hashes() and previous == previous_hashes()
    assert all(cr.digest(cr.HERE / name) == value for name, value in prior_benchmark.items())
    result = {"status": "pass", "selected_workers": selected["workers"], "profiles": profiles,
        "core_source_hashes": cr.source_hashes(), "test_source_sha256": cr.digest(__file__),
        "free_ram_bytes_at_start": free_start, "input_train_images": 36, "input_val_images": 3,
        "same_dataset_class_transform_labels_and_sampler": True, "model_optimizer_scheduler_rng_and_metrics_exact": True,
        "scientific_conditions_unchanged": True, "old_results_and_checkpoints_unchanged": True,
        "completed": timestamp()}
    cr.json_write(output / "verification.json", result)
    print("LOADER OPTIMIZATION VERIFIED: workers=" + str(selected["workers"]), flush=True)


if __name__ == "__main__":
    function = ctypes.windll.kernel32.SetThreadExecutionState
    function.argtypes, function.restype = [ctypes.c_uint], ctypes.c_uint
    held = bool(function(0x80000001))
    try:
        main()
    finally:
        if held:
            function(0x80000000)
