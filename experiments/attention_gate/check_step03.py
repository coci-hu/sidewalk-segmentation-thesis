"""Step 3 checks; tiny smoke runs only, no full-dataset training."""
import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import common_runner as runner
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from evaluate_existing import protected_hashes
from eval_common import CityscapesBlindDataset
from train_enet import BlindRoadDataset, DiceLoss

HERE = runner.HERE


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if not output.is_relative_to(HERE) or output.exists() or output == HERE:
        raise ValueError("Use a new directory inside attention_gate")
    output.mkdir(parents=True)
    checks = []
    started = time.perf_counter()

    def record(name, details):
        checks.append({"check": name, "status": "pass", "details": details})
        runner.json_write(output / "checks.json", checks, exclusive=False)
        print("PASS " + name, flush=True)

    try:
        before = protected_hashes()
        runner.json_write(output / "protected_files_before.json", before)
        step2 = HERE / "step_02_2026-10-14/run_002"
        evaluated_before = {str(path.relative_to(HERE)): runner.digest(path) for path in step2.rglob("*") if path.is_file()}
        runner.json_write(output / "step02_files_before.json", evaluated_before)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        runner.configure(42)
        protocol = runner.load_protocol()
        record("protected_sources_and_results_before", {"original_files": len(before), "step02_files": len(evaluated_before)})

        initialization = []
        for seed in (42, 43, 44):
            rng = torch.cuda.get_rng_state_all()
            a, b = runner.build_paired_model("A", seed), runner.build_paired_model("B", seed)
            assert equal_tree(torch.cuda.get_rng_state_all(), rng), "Constructor changed dropout CUDA RNG"
            sa, sb = a.state_dict(), b.state_dict()
            assert all(torch.equal(value, sb[key]) for key, value in sa.items())
            assert all(repr(module) == repr(getattr(b, name)) for name, module in a.named_children())
            c = runner.build_paired_model("B", seed)
            assert equal_tree(sb, c.state_dict())
            initialization.append({"seed": seed, "common_state_tensors": len(sa),
                "backbone_sha256": runner.state_digest(sa), "ca_sha256": runner.state_digest({k: v for k, v in sb.items() if k.startswith("ca.")})})
        runner.json_write(output / "paired_initialization.json", initialization)
        record("paired_initialization_and_architecture_three_seeds", initialization)
        del a, b, c, sa, sb
        gc.collect()

        a, b = runner.build_paired_model("A", 42).cuda(), runner.build_paired_model("B", 42).cuda()
        original_ca = b.ca
        b.ca = nn.Identity()
        x = torch.randn(2, 3, 96, 160, device="cuda")
        bypass = {}
        for training in (False, True):
            a.train(training)
            b.train(training)
            rng = runner.capture_rng(runner.generators(42))
            with torch.no_grad():
                ya = a(x)
                runner.restore_rng(rng, runner.generators(42))
                yb = b(x)
            assert torch.equal(ya, yb)
            bypass["train" if training else "eval"] = {"bitwise_equal": True, "max_absolute_error": (ya - yb).abs().max().item()}
        b.ca = original_ca
        record("CA_bypass_matches_baseline_in_eval_and_train", bypass)
        del a, b, x, ya, yb, original_ca
        gc.collect()
        torch.cuda.empty_cache()

        from components.ca_attention import CoordAtt
        module = CoordAtt(channels=128).cuda().eval()
        x = torch.randn(2, 128, 7, 11, device="cuda", requires_grad=True)
        horizontal = x.mean(2, keepdim=True)
        vertical = x.mean(3, keepdim=True).permute(0, 1, 3, 2)
        joint = module.act(module.bn1(module.conv1(torch.cat((horizontal, vertical), 3))))
        horizontal, vertical = torch.split(joint, [11, 7], 3)
        ah = module.conv_h(horizontal).sigmoid()
        aw = module.conv_w(vertical.permute(0, 1, 3, 2)).sigmoid()
        manual = x * aw * ah
        actual = module(x)
        assert torch.equal(manual, actual) and actual.shape == x.shape
        assert bool(((ah > 0) & (ah < 1)).all()) and bool(((aw > 0) & (aw < 1)).all())
        actual.square().mean().backward()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and p.grad.abs().sum().item() > 0 for p in module.parameters())
        record("CA_rectangular_axes_formula_and_gradients", {"input_shape": list(x.shape), "horizontal_shape": list(ah.shape),
               "vertical_shape": list(aw.shape), "formula_exact_match": True, "all_parameter_gradients_finite_nonzero": True})
        del module, x, horizontal, vertical, joint, ah, aw, manual, actual
        gc.collect()
        torch.cuda.empty_cache()

        loss_checks = []
        for device in ("cpu", "cuda"):
            logits = torch.randn(2, 2, 48, 80, device=device, requires_grad=True)
            target = torch.randint(0, 2, (2, 48, 80), device=device)
            # Reference-only comparison: historical CUDA CE cannot run in strict mode.
            torch.use_deterministic_algorithms(False)
            legacy = 0.4 * F.cross_entropy(logits, target) + 0.6 * DiceLoss()(logits, target)
            legacy_grad = torch.autograd.grad(legacy, logits)[0]
            torch.use_deterministic_algorithms(True)
            new = runner.combined_loss(logits, target, protocol)
            new_grad = torch.autograd.grad(new, logits)[0]
            assert torch.allclose(legacy, new, atol=2e-7, rtol=2e-6)
            assert torch.allclose(legacy_grad, new_grad, atol=1e-9, rtol=1e-5)
            loss_checks.append({"device": device, "loss_absolute_error": (legacy - new).abs().item(),
                "gradient_max_absolute_error": (legacy_grad - new_grad).abs().max().item(),
                "new_forward_and_backward_strict_deterministic": True, "dice_smooth": 1e-6})
        record("loss_and_gradients_match_original_CE_and_Dice", loss_checks)
        del logits, target, legacy, new, legacy_grad, new_grad

        data_train = runner.SidewalkDataset(protocol, "train")
        data_val = runner.SidewalkDataset(protocol, "val")
        old_train = BlindRoadDataset(str(runner.PROJECT / protocol["data"]["root_relative_to_project"]), "train", (768, 1024))
        old_val = CityscapesBlindDataset(str(runner.PROJECT / protocol["data"]["root_relative_to_project"]))
        assert [(str(x), str(y)) for x, y in data_train.pairs] == list(zip(old_train.img_paths, old_train.label_paths))
        assert [(str(x), str(y)) for x, y in data_val.pairs] == list(zip(old_val.img_paths, old_val.label_paths))
        matched_samples = []
        for dataset, legacy_dataset, indices, split in ((data_train, old_train, [0, 1487, 2974], "train"),
                                                       (data_val, old_val, [0, 267, 326], "val")):
            for index in indices:
                images, masks = dataset[index]
                old_images, old_masks = legacy_dataset[index]
                assert torch.equal(images, old_images) and torch.equal(masks, old_masks)
                matched_samples.append({"split": split, "index": index, "image": dataset.pairs[index][0].name,
                    "input_equal": True, "mask_equal": True, "mask_classes": torch.unique(masks).tolist()})
        record("all_pairs_and_six_actual_preprocessing_comparisons", {"train_pairs": len(data_train), "val_pairs": len(data_val), "samples": matched_samples})

        orders = {}
        for variant in ("A", "B"):
            gen = runner.generators(42)["sample_order"]
            sampler = runner.OrderedSampler(data_train, gen)
            orders[variant] = [list(sampler) for _ in range(2)]
        assert orders["A"] == orders["B"]
        assert all(sorted(order) == list(range(2975)) for order in orders["A"])
        assert orders["A"][0] != orders["A"][1]
        assert orders["A"][0] != list(runner.OrderedSampler(data_train, runner.generators(43)["sample_order"]))
        runner.json_write(output / "two_epoch_full_sample_order.json", orders["A"])
        record("full_dataset_order_paired_and_no_dropped_samples", {"epochs_checked": 2, "images_per_epoch": 2975,
               "batches_per_epoch": 744, "last_batch_size": 3, "same_seed_paired": True, "different_seed_order_changes": True})

        batch = [data_train[i] for i in range(4)]
        images = torch.stack([x for x, _ in batch]).cuda()
        masks = torch.stack([y for _, y in batch]).cuda()
        gradient_rows = []
        for variant in ("A", "B"):
            runner.configure(42)
            model = runner.build_paired_model(variant, 42).cuda().train()
            optimizer, scheduler = runner.optimizer_scheduler(model, protocol)
            group = optimizer.param_groups[0]
            assert group["weight_decay"] == 0.01 and group["lr"] == 1e-4 and group["betas"] == (0.9, 0.999)
            assert group["eps"] == 1e-8 and not group["amsgrad"] and not group["foreach"]
            assert scheduler.T_max == 100 and scheduler.eta_min == 1e-6
            before_classifier = model.classifier.weight.detach().clone()
            torch.cuda.reset_peak_memory_stats()
            logits = model(images)
            assert list(logits.shape) == [4, 2, 768, 1024] and bool(torch.isfinite(logits).all())
            loss = runner.combined_loss(logits, masks, protocol)
            loss.backward()
            norms = {}
            for name, parameter in model.named_parameters():
                assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), name
                norms[name] = parameter.grad.norm().item()
            assert sum(norms.values()) > 0
            if variant == "B":
                assert all(norms[name] > 0 for name in norms if name.startswith("ca."))
            optimizer.step()
            assert not torch.equal(before_classifier, model.classifier.weight)
            gradient_rows.append({"variant": variant, "batch_size": 4, "loss": loss.item(),
                "parameters_m": sum(p.numel() for p in model.parameters()) / 1e6,
                "parameter_tensors_with_finite_gradients": len(norms), "zero_gradient_tensors": [k for k, v in norms.items() if v == 0],
                "CA_gradient_norms": {k: v for k, v in norms.items() if k.startswith("ca.")},
                "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20})
            del model, optimizer, scheduler, group, logits, loss, before_classifier
            gc.collect()
            torch.cuda.empty_cache()
        record("real_768x1024_batch4_backward_and_optimizer", gradient_rows)
        del images, masks, batch, data_train, data_val, old_train, old_val
        gc.collect()
        torch.cuda.empty_cache()

        def launch(name, arguments, cwd=None, expect_success=True):
            environment = os.environ.copy()
            environment["PYTHONIOENCODING"] = "utf-8"
            environment["PYTHONUTF8"] = "1"
            with (output / f"{name}_console.log").open("x", encoding="utf-8") as log:
                result = subprocess.run([sys.executable, str(HERE / "common_runner.py")] + arguments,
                                        cwd=str(cwd or runner.PROJECT.parent), env=environment,
                                        stdout=log, stderr=subprocess.STDOUT)
            text = (output / f"{name}_console.log").read_text(encoding="utf-8")
            if expect_success and result.returncode:
                raise RuntimeError(f"{name} failed: " + text[-4000:])
            if not expect_success and not result.returncode:
                raise RuntimeError(f"{name} unexpectedly succeeded")
            return text

        plan_output = output / "formal_plan_only"
        launch("plan", ["--variant", "A", "--output", str(plan_output)])
        plan = json.loads((plan_output / "status.json").read_text(encoding="utf-8"))
        assert plan["status"] == "plan_only" and not plan["training_started"]
        assert plan["train_images"] == 2975 and plan["val_images"] == 500 and not (plan_output / "checkpoints").exists()
        message = launch("overwrite_rejected", ["--variant", "A", "--output", str(plan_output)], expect_success=False)
        assert "Refusing to overwrite" in message
        for i, cwd in enumerate((runner.PROJECT.parent, runner.CODE, HERE)):
            launch(f"help_cwd_{i}", ["--help"], cwd=cwd)
        record("default_plan_overwrite_guard_and_three_working_directories", plan)

        resumed = []
        full_checkpoints = {}
        for variant in ("A", "B"):
            full = output / f"{variant}_continuous"
            branch = output / f"{variant}_resume_epoch1"
            launch(f"{variant}_continuous", ["--mode", "smoke", "--variant", variant, "--output", str(full)])
            print("Completed tiny continuous run " + variant, flush=True)
            launch(f"{variant}_resume", ["--mode", "smoke", "--variant", variant, "--output", str(branch),
                                       "--resume", str(full / "checkpoints/epoch_001.pth")])
            original = torch.load(full / "checkpoints/epoch_002.pth", map_location="cpu", weights_only=False)
            restored = torch.load(branch / "checkpoints/epoch_002.pth", map_location="cpu", weights_only=False)
            assert equal_tree(original, restored), "Resumed checkpoint differs from uninterrupted run"
            assert original["completed_epoch"] == original["scheduler_state"]["last_epoch"] == 2
            assert all(row["train_batch_sizes"] == [4, 3] and row["train_images"] == 7 and row["val_images"] == 3 for row in original["records"])
            expected_lr = 1e-6 + (1e-4 - 1e-6) * (1 + np.cos(np.pi * 2 / 100)) / 2
            assert abs(original["optimizer_state"]["param_groups"][0]["lr"] - expected_lr) < 1e-15
            full_checkpoints[variant] = original
            status = json.loads((full / "status.json").read_text(encoding="utf-8"))
            resumed.append({"variant": variant, "train_images": 7, "validation_images": 3, "epochs": 2,
                "resume_new_process": True, "entire_checkpoint_bitwise_equal": True,
                "includes": ["model", "optimizer", "scheduler", "Python/NumPy/CPU/CUDA RNG", "data generators", "best weights", "metrics and sample order"],
                "wall_seconds_continuous": status["wall_seconds"], "peak_allocated_mib": status["cuda_peak_allocated_mib"]})
            record(f"{variant}_two_tiny_epochs_and_fresh_process_resume", resumed[-1])
            del restored
        assert [r["train_order"] for r in full_checkpoints["A"]["records"]] == [r["train_order"] for r in full_checkpoints["B"]["records"]]
        record("actual_A_B_worker1_loader_orders_match", {"epochs": 2, "batch_sizes_each_epoch": [4, 3]})
        message = launch("wrong_variant_resume", ["--mode", "smoke", "--variant", "B", "--output", str(output / "rejected_wrong_variant"),
            "--resume", str(output / "A_continuous/checkpoints/epoch_001.pth")], expect_success=False)
        assert "Resume code, protocol, model, seed or data fingerprint mismatch" in message
        record("incompatible_checkpoint_rejected_before_training", {"wrong_variant": True})

        after = protected_hashes()
        evaluated_after = {str(path.relative_to(HERE)): runner.digest(path) for path in step2.rglob("*") if path.is_file()}
        assert before == after and evaluated_before == evaluated_after
        runner.json_write(output / "integrity_after.json", {"protected_files_checked": len(before), "original_sha256_identical": True,
            "step02_files_checked": len(evaluated_before), "step02_sha256_identical": True})
        record("original_and_step02_artifacts_preserved", {"original_files": len(before), "step02_files": len(evaluated_before)})
        runner.json_write(output / "step_status.json", {"step": 3, "status": "complete", "checks_passed": len(checks),
            "formal_training_started": False, "implemented_variants": ["A: ENet", "B: ENet + current ReLU CA"],
            "smoke_only": True, "wall_seconds": time.perf_counter() - started,
            "next_step": "Step 4: fixed-sample attention diagnostics; decide formal training scope afterwards"})
        print("STEP 3 COMPLETE", flush=True)
    except Exception:
        runner.json_write(output / "failure.json", {"checks_passed_before_failure": len(checks), "traceback": traceback.format_exc()}, exclusive=False)
        raise


if __name__ == "__main__":
    main()
