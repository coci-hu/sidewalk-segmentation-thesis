"""Verify v2 adapter resume and strict rejection before formal continuation."""
import argparse
import json
import os
import subprocess
import sys
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
        raise ValueError("Use a new QA directory")
    output.mkdir(parents=True)
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"], environment["PYTHONUTF8"] = "utf-8", "1"
    def execute(name, options, success=True):
        with (output / f"{name}.log").open("x", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, "-u", str(core.HERE / "common_runner_v2.py")] + options,
                env=environment, cwd=str(core.PROJECT.parent), stdout=log, stderr=subprocess.STDOUT)
        text = (output / f"{name}.log").read_text(encoding="utf-8")
        if success and result.returncode:
            raise RuntimeError(text[-4000:])
        if not success and not result.returncode:
            raise RuntimeError("Expected rejection did not occur")
        return text
    continuous, resumed = output / "B_continuous", output / "B_resumed"
    execute("continuous", ["--mode", "smoke", "--variant", "B", "--output", str(continuous)])
    print("V2 two tiny epochs completed", flush=True)
    execute("resume", ["--mode", "smoke", "--variant", "B", "--output", str(resumed),
        "--resume", str(continuous / "checkpoints/epoch_001.pth")])
    full = torch.load(continuous / "checkpoints/epoch_002.pth", map_location="cpu", weights_only=False)
    branch = torch.load(resumed / "checkpoints/epoch_002.pth", map_location="cpu", weights_only=False)
    assert equal_tree(full, branch), "Adapter resume changed model/optimizer/RNG/metrics"
    print("V2 new-process resume checkpoint exact match", flush=True)
    old = core.HERE / "step_05_2026-10-14/benchmark_001/B_seed42_epoch1/checkpoints/epoch_001.pth"
    message = execute("old_checkpoint_rejected", ["--mode", "train", "--variant", "B", "--output", str(output / "rejected_old"),
        "--resume", str(old)], success=False)
    assert "Resume code, protocol, model, seed or data fingerprint mismatch" in message
    assert not (output / "rejected_old/checkpoints").exists()
    core.json_write(output / "verification.json", {"status": "pass", "adapter_new_process_resume_exact": True,
        "old_checkpoint_not_silently_accepted": True, "fast_source_hashes": fast.sources(),
        "equivalence_proof": str(fast.PROOF_FILE), "full_training_started": False})
    print("LOADER V2 INTEGRATION PASSED", flush=True)


if __name__ == "__main__":
    with fast.keep_awake():
        main()
