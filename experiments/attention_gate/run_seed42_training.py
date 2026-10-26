"""Authorized scope: finish A/B seed42 to epoch100, then verify and stop."""
import argparse
import json
import msvcrt
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
import common_runner as core
import common_runner_v2 as fast
from evaluate_existing import protected_hashes, timestamp
from run_step05_benchmark import previous_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--migration", required=True, type=Path)
    parser.add_argument("--resume-a", type=Path)
    parser.add_argument("--resume-b", type=Path)
    args = parser.parse_args()
    output, migration = args.output.resolve(), args.migration.resolve()
    if not output.is_relative_to(core.HERE) or output.exists() or not migration.is_relative_to(core.HERE):
        raise ValueError("Use a new job directory and local verified migration")
    proof = json.loads((migration / "migration_verification.json").read_text(encoding="utf-8"))
    if proof["status"] != "pass" or proof["fast_source_hashes"] != fast.sources():
        raise RuntimeError("Migration settings or sources no longer match")
    quality = json.loads((migration.parent / "loader_v2_integration_001/verification.json").read_text(encoding="utf-8"))
    if quality["status"] != "pass" or quality["fast_source_hashes"] != fast.sources():
        raise RuntimeError("Execution adapter integration has not passed")
    for row in proof["checkpoints"]:
        if core.digest(row["copy"]) != row["copy_sha256"] or core.digest(row["source"]) != row["source_sha256"]:
            raise RuntimeError("Checkpoint provenance changed")
    resumes, folders, carried = {}, {}, {}
    for variant, supplied in (("A", args.resume_a), ("B", args.resume_b)):
        checkpoint_path = (supplied or migration / f"{variant}_seed42_epoch1_loader_v2.pth").resolve()
        if not checkpoint_path.is_relative_to(core.HERE):
            raise ValueError("Resume checkpoint outside experiment")
        checkpoint = core.torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["settings"] != fast.expected_settings(variant, 42) or not 1 <= checkpoint["completed_epoch"] <= 100:
            raise RuntimeError("Incompatible resume checkpoint")
        resumes[variant] = checkpoint_path
        carried[variant] = checkpoint["completed_epoch"] == 100
        folders[variant] = checkpoint_path.parent.parent if carried[variant] else output / f"{variant}_seed42_epochs002_100"
        del checkpoint
    output.mkdir(parents=True)
    started = time.perf_counter()
    old_benchmark = core.HERE / "step_05_2026-10-14/benchmark_001"
    baseline = {"original": protected_hashes(), "previous": previous_hashes(),
        "benchmark": {str(p.relative_to(core.HERE)): core.digest(p) for p in old_benchmark.rglob("*") if p.is_file()}}
    core.json_write(output / "integrity_before.json", baseline)
    plan = {"scope": "A/B seed42 only; epochs2..100; stop before seeds43/44 or new modules",
        "pid": os.getpid(), "started": timestamp(), "source_hashes": fast.sources(),
        "num_workers": fast.protocol()["formal_training"]["num_workers"],
        "migration": str(migration), "migration_sha256": core.digest(migration / "migration_verification.json"),
        "checkpoint_sources": proof["checkpoints"], "job_source_sha256": core.digest(__file__),
        "resume_checkpoints": {v: {"path": str(p), "sha256": core.digest(p)} for v, p in resumes.items()},
        "variant_folders": {v: str(p) for v, p in folders.items()},
        "checkpoint_each_completed_epoch": True, "automatic_other_seeds": False}
    core.json_write(output / "job_plan.json", plan)
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"], environment["PYTHONUTF8"] = "utf-8", "1"
    try:
        for variant in ("A", "B"):
            if carried[variant]:
                print(timestamp() + " Reusing completed " + variant + " checkpoint; no repeat training", flush=True)
                continue
            if (output / "STOP_AFTER_EPOCH").exists():
                fast.atomic_json(output / "job_status.json", {"status": "stopped_at_epoch_boundary", "next_variant": variant})
                return
            fast.atomic_json(output / "job_status.json", {"status": "running", "variant": variant, "seed": 42,
                "target_epochs": 100, "pid": os.getpid(), "started": plan["started"], "elapsed_seconds": time.perf_counter() - started})
            print(timestamp() + " Starting " + variant + " seed42 epochs2..100", flush=True)
            with (output / f"{variant}_console.log").open("x", encoding="utf-8") as log:
                process = subprocess.Popen([sys.executable, "-u", str(core.HERE / "common_runner_v2.py"),
                    "--mode", "train", "--variant", variant, "--seed", "42",
                    "--output", str(folders[variant]),
                    "--resume", str(resumes[variant])],
                    cwd=str(core.PROJECT.parent), env=environment, stdout=log, stderr=subprocess.STDOUT)
                child_progress = output / f"{variant}_seed42_epochs002_100/training_progress.json"
                while process.poll() is None:
                    live = {}
                    if child_progress.exists():
                        try:
                            live = json.loads(child_progress.read_text(encoding="utf-8"))
                        except (OSError, ValueError):
                            pass
                    fast.atomic_json(output / "job_status.json", {"status": "running", "variant": variant,
                        "seed": 42, "target_epochs": 100, "pid": os.getpid(), "child_pid": process.pid,
                        "elapsed_seconds": time.perf_counter() - started, "training": live})
                    time.sleep(15)
                exit_code = process.wait()
            if exit_code == 75 and (output / "STOP_AFTER_EPOCH").exists():
                fast.atomic_json(output / "job_status.json", {"status": "stopped_at_epoch_boundary", "variant": variant,
                    "reason": "requested stop; latest complete checkpoint preserved"})
                return
            if exit_code:
                raise RuntimeError(f"{variant} failed, see {variant}_console.log; completed epochs preserved")
            print(timestamp() + " Completed " + variant, flush=True)
        fast.atomic_json(output / "job_status.json", {"status": "verifying", "seed": 42, "pid": os.getpid()})
        from verify_seed42_training import finish
        results = finish(output)
        fast.atomic_json(output / "job_status.json", {"status": "complete", "seed": 42, "epochs_per_variant": 100,
            "verified": True, "completed": timestamp(), "wall_seconds": time.perf_counter() - started,
            "three_seed_study_complete": False, "next_step": "review seed42 results before seeds43/44; no new modules started",
            "report": str(output / "REPORT.md"), "results": results})
        print(timestamp() + " SEED42 PAIR COMPLETE AND VERIFIED", flush=True)
    except BaseException:
        error = {"status": "failed", "at": timestamp(), "traceback": traceback.format_exc(),
            "completed_epochs_preserved": True, "no_automatic_restart": True}
        fast.atomic_json(output / "job_status.json", error)
        raise


if __name__ == "__main__":
    lock_path = core.HERE / "seed42_training.lock"
    with lock_path.open("a+b") as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b"1")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            with fast.keep_awake():
                main()
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
