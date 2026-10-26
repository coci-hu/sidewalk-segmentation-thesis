"""Time ONE full A or B epoch without changing common_runner or training logic.

Pass-through loader timing consumes no RNG and changes no sample/order/updates.
T_max remains 100. Epoch 1 checkpoint can resume unchanged common_runner.
"""
import argparse
import csv
import ctypes
import json
import os
import subprocess
import threading
import time
import traceback
from pathlib import Path
import common_runner as cr
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("A", "B"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(cr.HERE) or output.exists():
        raise ValueError("Use a NEW attention_gate output subdirectory")
    phase_times, batch_times, power_rows = {}, [], []
    finished = threading.Event()
    start = time.perf_counter()
    original_train, original_validate = cr.train_epoch, cr.validate

    def progress(phase, completed, total):
        state = {"variant": args.variant, "seed": 42, "epoch": 1, "phase": phase,
                 "batches_completed": completed, "batches_total": total,
                 "elapsed_seconds": time.perf_counter() - start}
        cr.json_write(output / "benchmark_progress.json", state, exclusive=False)
        print(f"{args.variant} {phase} {completed}/{total} elapsed={state['elapsed_seconds']:.1f}s", flush=True)

    class TimedLoader:
        def __init__(self, loader, phase):
            self.loader, self.phase = loader, phase

        def __len__(self):
            return len(self.loader)

        def __iter__(self):
            completed = 0
            wait_start = time.perf_counter()
            for batch in self.loader:
                fetched = time.perf_counter()
                wait = fetched - wait_start
                yield batch
                ended = time.perf_counter()
                completed += 1
                batch_times.append({"phase": self.phase, "batch": completed, "images": len(batch[0]),
                    "loader_wait_seconds": wait, "caller_work_seconds": ended - fetched})
                if completed == 1 or completed % 50 == 0 or completed == len(self.loader):
                    progress(self.phase, completed, len(self.loader))
                wait_start = time.perf_counter()

    def measured_train(model, loader, optimizer, protocol, device):
        torch.cuda.synchronize()
        began = time.perf_counter()
        result = original_train(model, TimedLoader(loader, "train"), optimizer, protocol, device)
        torch.cuda.synchronize()
        phase_times["train_seconds"] = time.perf_counter() - began
        return result

    def measured_validate(model, loader, protocol, device):
        torch.cuda.synchronize()
        began = time.perf_counter()
        result = original_validate(model, TimedLoader(loader, "validation"), protocol, device)
        torch.cuda.synchronize()
        phase_times["validation_seconds"] = time.perf_counter() - began
        return result

    def monitor_gpu():
        while not finished.is_set():
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                values = result.stdout.strip().splitlines()[0].split(",")
                row = {"elapsed_seconds": time.perf_counter() - start}
                for key, value in zip(("gpu_power_w", "gpu_temperature_c", "gpu_utilization_pct", "gpu_memory_used_mib"), values):
                    try:
                        row[key] = float(value.strip())
                    except ValueError:
                        row[key] = None
                power_rows.append(row)
            except Exception as error:
                power_rows.append({"elapsed_seconds": time.perf_counter() - start, "monitor_error": str(error)})
            finished.wait(30)

    sleep_guard = False
    if os.name == "nt":
        # Request system wakefulness only while this thread lives; no global setting edits.
        function = ctypes.windll.kernel32.SetThreadExecutionState
        function.argtypes = [ctypes.c_uint]
        function.restype = ctypes.c_uint
        sleep_guard = bool(function(0x80000001))
    thread = threading.Thread(target=monitor_gpu, daemon=True)
    cr.train_epoch, cr.validate = measured_train, measured_validate
    try:
        thread.start()
        checkpoint = cr.run(args.variant, 42, "train", output, stop_after_epoch=1)
        settings = json.loads((output / "settings.json").read_text(encoding="utf-8"))
        assert settings["epoch_limit"] == settings["protocol"]["formal_training"]["scheduler_T_max"] == 100
        status = json.loads((output / "status.json").read_text(encoding="utf-8"))
        assert status["completed_epoch"] == 1 and status["status"] == "stopped_at_epoch_boundary"
        finished.set()
        thread.join(timeout=12)
        with (output / "batch_timing.csv").open("x", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(batch_times[0]))
            writer.writeheader()
            writer.writerows(batch_times)
        cr.json_write(output / "gpu_power_samples.json", power_rows)
        cr.json_write(output / "benchmark_timing.json", {"variant": args.variant, "seed": 42,
            "completed_epochs": 1, "train_images": 2975, "validation_images": 500,
            "train_batches": sum(row["phase"] == "train" for row in batch_times),
            "validation_batches": sum(row["phase"] == "validation" for row in batch_times),
            **phase_times, "whole_runner_call_wall_seconds": time.perf_counter() - start,
            "cuda_peak_allocated_mib": status["cuda_peak_allocated_mib"], "checkpoint": str(checkpoint),
            "benchmark_source_sha256": cr.digest(__file__), "common_runner_unchanged": True,
            "monitor_interval_seconds": 30, "power_scope": "GPU sensor, not whole-system electricity",
            "sleep_prevention_during_process": sleep_guard, "training_continued_after_epoch1": False})
        print(f"{args.variant} BENCHMARK COMPLETE, train={phase_times['train_seconds']:.1f}s, validation={phase_times['validation_seconds']:.1f}s", flush=True)
    except Exception:
        if output.exists():
            cr.json_write(output / "benchmark_failure.json", {"traceback": traceback.format_exc()}, exclusive=False)
        raise
    finally:
        finished.set()
        thread.join(timeout=12)
        cr.train_epoch, cr.validate = original_train, original_validate
        if os.name == "nt" and sleep_guard:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
