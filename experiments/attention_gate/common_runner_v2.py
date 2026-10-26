"""Versioned loader execution adapter; scientific operations remain in v1.

Only protocol loader process count changes. v1 and all original outputs stay
unchanged. Resume is still checked by v1's exact settings comparison.
"""
import argparse
import contextlib
import ctypes
import json
import os
import time
from pathlib import Path
import common_runner as core
import torch

BASE_SOURCE_HASHES = core.source_hashes
BASE_LOAD_PROTOCOL = core.load_protocol
BASE_TRAIN = core.train_epoch
BASE_VALIDATE = core.validate
PROTOCOL_FILE = core.HERE / "protocol_v1_2.json"
PROOF_FILE = core.HERE / "step_05_2026-10-14/loader_optimization_001/verification.json"


def protocol():
    data = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    proof = json.loads(PROOF_FILE.read_text(encoding="utf-8"))
    if proof["status"] != "pass" or proof["core_source_hashes"] != BASE_SOURCE_HASHES():
        raise RuntimeError("Loader equivalence evidence is missing or stale")
    if proof["test_source_sha256"] != core.digest(core.HERE / "check_loader_optimization.py"):
        raise RuntimeError("Loader equivalence test source changed")
    original = BASE_LOAD_PROTOCOL()
    expected = json.loads(json.dumps(original))
    expected["formal_training"]["num_workers"] = proof["selected_workers"]
    for key in ("data", "metrics", "formal_training", "candidate_matrix", "mixed_gate_candidate", "resources", "output_policy"):
        if data[key] != expected[key]:
            raise RuntimeError("Unexpected change beyond loader count: " + key)
    return data


def sources():
    result = BASE_SOURCE_HASHES()
    for path in (Path(__file__), PROTOCOL_FILE, PROOF_FILE):
        result[str(path.relative_to(core.PROJECT))] = core.digest(path)
    return result


def expected_settings(variant, seed, mode="train"):
    import hashlib
    settings_protocol = protocol()
    _, _, manifest = core.prepare(mode, settings_protocol)
    return {"variant": variant, "seed": seed, "mode": mode, "protocol": settings_protocol,
        "source_hashes": sources(),
        "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest(),
        "runtime": {"python": __import__("sys").version, "torch": torch.__version__,
            "torchvision": core.torchvision.__version__, "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "epoch_limit": 2 if mode == "smoke" else settings_protocol["formal_training"]["epochs"]}


def atomic_json(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(core.HERE):
        raise ValueError("Progress outside workspace")
    temporary = path.with_suffix(path.suffix + ".tmp")
    core.json_write(temporary, value, exclusive=False)
    os.replace(temporary, path)


def run(variant, seed, mode, output, resume=None, stop_after_epoch=None):
    output = Path(output).resolve()
    proof = protocol()
    old_values = core.load_protocol, core.source_hashes, core.train_epoch, core.validate
    completed = 0
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        completed = checkpoint["completed_epoch"]
        del checkpoint
    epoch = completed
    times = []
    wall_start = time.perf_counter()
    pinned_sources = sources()

    class ProgressLoader:
        def __init__(self, loader, phase):
            self.loader, self.phase = loader, phase
        def __len__(self):
            return len(self.loader)
        def __iter__(self):
            count = 0
            for batch in self.loader:
                yield batch
                count += 1
                if count == 1 or count % 50 == 0 or count == len(self.loader):
                    atomic_json(output / "training_progress.json", {"status": "running", "variant": variant,
                        "seed": seed, "epoch": epoch, "target_epochs": 100, "phase": self.phase,
                        "batches_completed": count, "batches_total": len(self.loader),
                        "num_workers": proof["formal_training"]["num_workers"], "elapsed_seconds": time.perf_counter() - wall_start})
                    print(f"{variant} epoch={epoch} {self.phase} {count}/{len(self.loader)}", flush=True)

    def train(model, loader, optimizer, p, device):
        nonlocal epoch
        if sources() != pinned_sources:
            raise RuntimeError("Execution sources changed during this run")
        if (output.parent / "STOP_AFTER_EPOCH").exists():
            raise StopAtEpochBoundary()
        epoch += 1
        torch.cuda.synchronize()
        began = time.perf_counter()
        result = BASE_TRAIN(model, ProgressLoader(loader, "train"), optimizer, p, device)
        torch.cuda.synchronize()
        times.append({"epoch": epoch, "train_seconds": time.perf_counter() - began})
        return result

    def validate(model, loader, p, device):
        torch.cuda.synchronize()
        began = time.perf_counter()
        result = BASE_VALIDATE(model, ProgressLoader(loader, "validation"), p, device)
        torch.cuda.synchronize()
        times[-1]["validation_seconds"] = time.perf_counter() - began
        atomic_json(output / "epoch_timing.json", times)
        return result

    core.load_protocol, core.source_hashes = protocol, sources
    core.train_epoch, core.validate = train, validate
    try:
        path = core.run(variant, seed, mode, output, resume, stop_after_epoch)
        state = json.loads((output / "status.json").read_text(encoding="utf-8"))
        atomic_json(output / "training_progress.json", state)
        return path
    except StopAtEpochBoundary:
        # The last complete epoch has already been saved by the unmodified loop.
        state_path = output / "status.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {
            "completed_epoch": completed, "last_checkpoint": str(resume)}
        state.update({"status": "stopped_at_epoch_boundary", "variant": variant,
            "last_completed_epoch": epoch, "reason": "STOP_AFTER_EPOCH requested"})
        atomic_json(state_path, state)
        atomic_json(output / "training_progress.json", state)
        raise
    finally:
        core.load_protocol, core.source_hashes, core.train_epoch, core.validate = old_values


class StopAtEpochBoundary(Exception):
    pass


@contextlib.contextmanager
def keep_awake():
    function = ctypes.windll.kernel32.SetThreadExecutionState
    function.argtypes, function.restype = [ctypes.c_uint], ctypes.c_uint
    held = bool(function(0x80000001))
    if not held:
        raise RuntimeError("Cannot acquire temporary system wake request")
    try:
        yield
    finally:
        function(0x80000000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("A", "B"), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--mode", choices=("plan", "smoke", "train"), default="plan")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-epoch", type=int)
    args = parser.parse_args()
    with keep_awake():
        try:
            run(args.variant, args.seed, args.mode, args.output, args.resume, args.stop_after_epoch)
        except StopAtEpochBoundary:
            raise SystemExit(75)


if __name__ == "__main__":
    main()
