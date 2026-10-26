"""Create traceable NEW execution copies after loader equivalence passed."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import torch
import common_runner as core
import common_runner_v2 as fast
from check_step03 import equal_tree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(core.HERE) or output.exists():
        raise ValueError("Use a new migration output")
    proof = json.loads(fast.PROOF_FILE.read_text(encoding="utf-8"))
    fast.protocol()  # Validate proof and absence of scientific changes before writes.
    output.mkdir(parents=True)
    reports = []
    for variant in ("A", "B"):
        old = core.HERE / f"step_05_2026-10-14/benchmark_001/{variant}_seed42_epoch1/checkpoints/epoch_001.pth"
        old_hash = core.digest(old)
        checkpoint = torch.load(old, map_location="cpu", weights_only=False)
        new_settings = fast.expected_settings(variant, 42)
        settings = copy.deepcopy(new_settings)
        settings["protocol"] = core.load_protocol()
        settings["source_hashes"] = core.source_hashes()
        if checkpoint["settings"] != settings or checkpoint["completed_epoch"] != 1:
            raise RuntimeError("Original checkpoint is not exactly the verified v1 one-epoch run")
        if checkpoint["format"] != "attention_gate_epoch_v1":
            raise RuntimeError("Unexpected checkpoint schema")
        old_settings = checkpoint["settings"]
        state_hash = core.state_digest(checkpoint["model_state"])
        checkpoint["settings"] = new_settings
        checkpoint["loader_execution_migration"] = {"parent_checkpoint": str(old), "parent_sha256": old_hash,
            "parent_settings": old_settings, "new_protocol_id": new_settings["protocol"]["protocol_id"],
            "equivalence_proof": str(fast.PROOF_FILE), "equivalence_proof_sha256": core.digest(fast.PROOF_FILE),
            "epochs_completed_under_parent_execution": 1, "next_epoch": 2,
            "change": "num_workers only; scientific operations, tensor states, history, and RNG preserved"}
        new = output / f"{variant}_seed42_epoch1_loader_v2.pth"
        with new.open("xb") as f:
            torch.save(checkpoint, f)
        loaded = torch.load(new, map_location="cpu", weights_only=False)
        reference = torch.load(old, map_location="cpu", weights_only=False)
        for field in ("model_state", "optimizer_state", "scheduler_state", "rng", "records", "best_model_state", "best_epoch", "best_miou", "completed_epoch"):
            if not equal_tree(reference[field], loaded[field]):
                raise RuntimeError("Migration changed training state: " + field)
        if core.digest(old) != old_hash or core.state_digest(loaded["model_state"]) != state_hash:
            raise RuntimeError("Migration state preservation failed")
        reports.append({"variant": variant, "source": str(old), "source_sha256": old_hash,
            "copy": str(new), "copy_sha256": core.digest(new), "all_training_state_and_history_exact": True,
            "original_unchanged": True, "new_workers": proof["selected_workers"]})
    core.json_write(output / "migration_verification.json", {"status": "pass", "checkpoints": reports,
        "core_source_hashes": core.source_hashes(), "fast_source_hashes": fast.sources(),
        "proof_sha256": core.digest(fast.PROOF_FILE)})
    print("Both checkpoint execution copies verified; originals preserved")


if __name__ == "__main__":
    main()
