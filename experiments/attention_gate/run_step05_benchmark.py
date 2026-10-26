"""Sequential A/B full ONE epoch each; no automatic longer training."""
import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
import common_runner as cr
from evaluate_existing import protected_hashes, timestamp


def previous_hashes():
    values = {}
    for directory in ("step_02_2026-10-14/run_002", "step_03_2026-10-14/run_003", "step_04_2026-10-14"):
        for path in (cr.HERE / directory).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                values[str(path.relative_to(cr.HERE))] = cr.digest(path)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(cr.HERE) or output.exists():
        raise ValueError("Use a new output directory")
    output.mkdir(parents=True)
    started = time.perf_counter()
    try:
        step4 = json.loads((cr.HERE / "step_04_2026-10-14/run_001/step_status.json").read_text(encoding="utf-8"))
        assert step4["status"] == "complete" and step4["analysis_verified"]
        protected, previous, sources = protected_hashes(), previous_hashes(), cr.source_hashes()
        cr.json_write(output / "protected_files_before.json", protected)
        cr.json_write(output / "previous_steps_before.json", previous)
        cr.json_write(output / "plan.json", {"scope": "exactly one full epoch each, A then B, seed42",
            "train_images": 2975, "validation_images": 500, "T_max": 100, "stop_after_epoch": 1,
            "source_hashes": sources, "automatic_continuation": False, "started": timestamp()})
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["PYTHONUTF8"] = "1"
        for variant in ("A", "B"):
            cr.json_write(output / "progress.json", {"status": "running", "current_variant": variant,
                "epoch_limit_this_task": 1, "elapsed_seconds": time.perf_counter() - started}, exclusive=False)
            print("Starting full epoch " + variant, flush=True)
            with (output / f"{variant}_console.log").open("x", encoding="utf-8") as log:
                result = subprocess.run([sys.executable, "-u", str(cr.HERE / "benchmark_one_epoch.py"),
                    "--variant", variant, "--output", str(output / f"{variant}_seed42_epoch1")],
                    cwd=str(cr.PROJECT.parent), env=environment, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Variant {variant} failed; see {variant}_console.log")
            print("Finished full epoch " + variant, flush=True)
        assert protected_hashes() == protected and previous_hashes() == previous and cr.source_hashes() == sources
        cr.json_write(output / "integrity_after.json", {"original_files": len(protected), "original_sha256_identical": True,
            "previous_step_files": len(previous), "previous_step_sha256_identical": True, "common_runner_sources_unchanged": True})
        status = {"step": 5, "substage": "one_full_epoch_benchmark", "status": "benchmark_complete_pending_verification",
            "variants": ["A", "B"], "seed": 42, "epochs_per_variant": 1,
            "automatic_continuation": False, "wall_seconds": time.perf_counter() - started, "finished": timestamp()}
        cr.json_write(output / "step_status.json", status)
        cr.json_write(output / "progress.json", status, exclusive=False)
        print("BOTH ONE-EPOCH BENCHMARKS COMPLETE", flush=True)
    except Exception:
        cr.json_write(output / "failure.json", {"traceback": traceback.format_exc(), "at": timestamp()}, exclusive=False)
        raise


if __name__ == "__main__":
    # Cover setup, inter-process gaps and integrity verification as well as
    # the child training process. Releasing the child-only request previously
    # allowed an already idle machine to sleep before parent verification.
    guard = False
    if os.name == "nt":
        function = ctypes.windll.kernel32.SetThreadExecutionState
        function.argtypes = [ctypes.c_uint]
        function.restype = ctypes.c_uint
        guard = bool(function(0x80000001))
    try:
        main()
    finally:
        if guard:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
