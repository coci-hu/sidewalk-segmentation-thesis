"""Matched A/B training, separate from all historical runners and outputs.

Default mode only writes a plan. Smoke mode is two epochs on 7/3 images.
Formal mode requires --mode train and is never invoked by the Step 3 check.
Resume is supported at completed epoch boundaries, into a NEW directory.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
CODE = PROJECT / "code"
sys.path.insert(0, str(CODE))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from torchvision import transforms
from model_enet import ENet
from model_enet_ca import ENetCA
from metrics_v1 import boundary_iou_author, confusion_matrix_numpy, scores_from_confusion


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def json_write(path, data, exclusive=True):
    with Path(path).open("x" if exclusive else "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def load_protocol():
    return json.loads((HERE / "protocol_v1_1.json").read_text(encoding="utf-8"))


def configure(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def source_hashes():
    paths = [HERE / "common_runner.py", HERE / "metrics_v1.py", HERE / "protocol_v1_1.json",
             CODE / "model_enet.py", CODE / "model_enet_ca.py", CODE / "components/ca_attention.py"]
    return {str(p.relative_to(PROJECT)): digest(p) for p in paths}


def state_digest(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        result.update(name.encode("utf-8"))
        result.update(str(value.dtype).encode("ascii"))
        result.update(str(tuple(value.shape)).encode("ascii"))
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def build_paired_model(variant, seed):
    """Do not rely on identical constructor seeds: CA shifts decoder RNG draws."""
    if variant not in ("A", "B"):
        raise ValueError("Only baseline A and historical ReLU CA B are implemented")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        backbone = ENet(num_classes=2)
        if variant == "A":
            return backbone
        torch.random.default_generator.manual_seed(seed + 10000)
        model = ENetCA(num_classes=2)
    baseline = backbone.state_dict()
    state = model.state_dict()
    if set(state) - set(baseline) != {key for key in state if key.startswith("ca.")}:
        raise RuntimeError("Unexpected architecture difference outside CA")
    for key, value in baseline.items():
        if key not in state or value.shape != state[key].shape:
            raise RuntimeError("Backbone state mismatch: " + key)
        state[key] = value.clone()
    model.load_state_dict(state, strict=True)
    return model


class SidewalkDataset(Dataset):
    def __init__(self, protocol, split):
        data = protocol["data"]
        root = PROJECT / data["root_relative_to_project"]
        self.height, self.width = data["input_height"], data["input_width"]
        self.foreground = data["foreground_original_label_id"]
        image_root = root / "leftImg8bit" / split
        label_root = root / "gtFine" / split
        self.pairs = []
        for city in sorted(p for p in image_root.iterdir() if p.is_dir()):
            for image in sorted(city.glob("*_leftImg8bit.png")):
                label = label_root / city.name / image.name.replace("leftImg8bit", "gtFine_labelIds")
                if not label.is_file():
                    raise FileNotFoundError(label)
                self.pairs.append((image, label))
        expected = data["train_images" if split == "train" else "validation_images"]
        if len(self.pairs) != expected:
            raise RuntimeError(f"Unexpected {split} count {len(self.pairs)}, expected {expected}")
        self.transform = transforms.Compose([
            transforms.Resize((self.height, self.width)), transforms.ToTensor(),
            transforms.Normalize(data["normalization_mean"], data["normalization_std"])])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, label_path = self.pairs[index]
        with Image.open(image_path) as source:
            image = self.transform(source.convert("RGB"))
        with Image.open(label_path) as source:
            binary = (np.array(source) == self.foreground).astype(np.uint8)
        mask = np.array(Image.fromarray(binary).resize((self.width, self.height), Image.NEAREST))
        return image, torch.from_numpy(mask.astype(np.int64))

    def manifest(self):
        return [{"image": str(image.relative_to(PROJECT)), "label": str(label.relative_to(PROJECT)),
                 "image_bytes": image.stat().st_size, "image_mtime_ns": image.stat().st_mtime_ns,
                 "label_bytes": label.stat().st_size, "label_mtime_ns": label.stat().st_mtime_ns}
                for image, label in self.pairs]


class OrderedSampler(Sampler):
    def __init__(self, dataset, generator):
        self.size, self.generator = len(dataset), generator
        self.last_order = []

    def __iter__(self):
        self.last_order = torch.randperm(self.size, generator=self.generator).tolist()
        return iter(self.last_order)

    def __len__(self):
        return self.size


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def generators(seed):
    return {name: torch.Generator().manual_seed(seed + offset) for name, offset in
            (("sample_order", 20000), ("train_workers", 30000), ("val_workers", 40000))}


def make_loaders(train, val, protocol, gens):
    settings = protocol["formal_training"]
    sampler = OrderedSampler(train, gens["sample_order"])
    shared = dict(num_workers=settings["num_workers"], pin_memory=settings["pin_memory"],
                  persistent_workers=False, worker_init_fn=seed_worker)
    train_loader = DataLoader(train, batch_size=settings["physical_batch_size"], sampler=sampler,
                              drop_last=False, generator=gens["train_workers"], **shared)
    val_loader = DataLoader(val, batch_size=1, shuffle=False, drop_last=False,
                            generator=gens["val_workers"], **shared)
    return train_loader, val_loader, sampler


def combined_loss(logits, target, protocol):
    settings = protocol["formal_training"]["loss"]
    probabilities = F.softmax(logits, dim=1)
    truth = F.one_hot(target, num_classes=2).permute(0, 3, 1, 2).float()
    intersection = (probabilities * truth).sum(dim=(2, 3))
    union = probabilities.sum(dim=(2, 3)) + truth.sum(dim=(2, 3))
    smooth = settings["dice_smooth"]
    dice = 1 - ((2 * intersection + smooth) / (union + smooth)).mean()
    # Equivalent unweighted CE; local CUDA nll_loss2d mean rejects strict determinism.
    ce = -F.log_softmax(logits, dim=1).gather(1, target.unsqueeze(1)).mean()
    return settings["cross_entropy_weight"] * ce + settings["dice_weight"] * dice


def optimizer_scheduler(model, protocol):
    s = protocol["formal_training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=s["learning_rate"],
        weight_decay=s["weight_decay"], betas=tuple(s["betas"]), eps=s["epsilon"],
        amsgrad=s["amsgrad"], foreach=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=s["scheduler_T_max"], eta_min=s["scheduler_eta_min"])
    return optimizer, scheduler


def capture_rng(gens):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "generators": {key: value.get_state() for key, value in gens.items()}}


def restore_rng(state, gens):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    for key, value in gens.items():
        value.set_state(state["generators"][key])


def train_epoch(model, loader, optimizer, protocol, device):
    model.train()
    total, count, batch_sizes = 0.0, 0, []
    for images, target in loader:
        images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = combined_loss(model(images), target, protocol)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
        size = images.shape[0]
        total += loss.item() * size
        count += size
        batch_sizes.append(size)
    return {"train_loss": total / count, "train_images": count, "train_batch_sizes": batch_sizes}


@torch.inference_mode()
def validate(model, loader, protocol, device):
    model.eval()
    confusion = np.zeros((2, 2), dtype=np.int64)
    total, count, boundary_values = 0.0, 0, []
    for images, target in loader:
        images, target_gpu = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
        logits = model(images)
        loss = combined_loss(logits, target_gpu, protocol)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite validation loss")
        prediction = logits.argmax(1).cpu().numpy()
        truth = target.numpy()
        confusion += confusion_matrix_numpy(prediction, truth)
        for pred, gt in zip(prediction, truth):
            value = boundary_iou_author(gt == 1, pred == 1, protocol["metrics"]["boundary_dilation_ratio"])
            if value is not None:
                boundary_values.append(value)
        size = len(images)
        total += loss.item() * size
        count += size
    scores = scores_from_confusion(confusion)
    return {"val_loss": total / count, "val_images": count, "val_miou_global": scores["miou"],
            "val_iou_sidewalk_global": scores["iou"][1], "val_pixel_accuracy_global": scores["pixel_acc"],
            "val_boundary_iou_author_fg_mean": float(np.mean(boundary_values)) if boundary_values else None,
            "boundary_valid_images": len(boundary_values), "val_confusion": confusion.tolist()}


def prepare(mode, protocol):
    train, val = SidewalkDataset(protocol, "train"), SidewalkDataset(protocol, "val")
    train_ids, val_ids = list(range(len(train))), list(range(len(val)))
    if mode == "smoke":
        train_ids = list(range(7))
        first = {}
        for i, (image, _) in enumerate(val.pairs):
            first.setdefault(image.parent.name, i)
        val_ids = list(first.values())
    manifest = {"train": train.manifest(), "val": val.manifest(),
                "selected_train_indices": train_ids, "selected_val_indices": val_ids}
    return Subset(train, train_ids), Subset(val, val_ids), manifest


def run(variant, seed, mode, output, resume=None, stop_after_epoch=None):
    if mode not in ("plan", "smoke", "train") or variant not in ("A", "B"):
        raise ValueError("Unknown run mode or variant")
    output = Path(output).resolve()
    if not output.is_relative_to(HERE) or output == HERE:
        raise ValueError("Outputs must be a new subdirectory of attention_gate")
    if output.exists():
        raise FileExistsError("Refusing to overwrite: " + str(output))
    output.mkdir(parents=True)
    protocol = load_protocol()
    if seed not in protocol["formal_training"]["seeds"]:
        raise ValueError("Seed is outside the fixed protocol")
    train, val, manifest = prepare("train" if mode == "plan" else mode, protocol)
    fingerprint = source_hashes()
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()
    settings = {"variant": variant, "seed": seed, "mode": mode, "protocol": protocol,
                "source_hashes": fingerprint, "manifest_sha256": manifest_hash,
                "runtime": {"python": sys.version, "torch": torch.__version__,
                            "torchvision": torchvision.__version__, "cuda_runtime": torch.version.cuda,
                            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
                "epoch_limit": 2 if mode == "smoke" else protocol["formal_training"]["epochs"]}
    json_write(output / "settings.json", settings)
    json_write(output / "sample_manifest.json", manifest)
    json_write(output / "environment.json", {"python": sys.version, "torch": torch.__version__,
        "torchvision": torchvision.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "deterministic_algorithms": True, "cudnn_benchmark": False,
        "cudnn_allow_tf32": True, "matmul_allow_tf32": False,
        "optimizer_foreach": False, "resume_scope": "completed epochs, same code/data/device environment"})
    if mode == "plan":
        json_write(output / "status.json", {"status": "plan_only", "training_started": False,
                   "train_images": len(train), "val_images": len(val),
                   "steps_per_epoch": (len(train) + 3) // 4, "last_batch_size": len(train) % 4 or 4})
        return None
    if not torch.cuda.is_available():
        raise RuntimeError("This fixed protocol requires the local CUDA GPU")
    device = torch.device("cuda:0")
    configure(seed)
    gens = generators(seed)
    model = build_paired_model(variant, seed).to(device)
    initial_backbone_hash = state_digest({k: v for k, v in model.state_dict().items() if not k.startswith("ca.")})
    optimizer, scheduler = optimizer_scheduler(model, protocol)
    train_loader, val_loader, sampler = make_loaders(train, val, protocol, gens)
    start_epoch, best_miou, best_epoch, best_state, records = 1, -float("inf"), None, None, []
    if resume:
        resume = Path(resume).resolve()
        if not resume.is_relative_to(HERE):
            raise ValueError("Only checkpoints from the new experiment are accepted")
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != "attention_gate_epoch_v1" or checkpoint["settings"] != settings:
            raise RuntimeError("Resume code, protocol, model, seed or data fingerprint mismatch")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["completed_epoch"] + 1
        best_miou, best_epoch = checkpoint["best_miou"], checkpoint["best_epoch"]
        best_state, records = checkpoint["best_model_state"], checkpoint["records"]
        restore_rng(checkpoint["rng"], gens)
        json_write(output / "resume_provenance.json", {"checkpoint": str(resume), "sha256": digest(resume),
                   "starting_epoch": start_epoch, "carried_best_epoch": best_epoch})
    output.joinpath("checkpoints").mkdir()
    output.joinpath("best_weights").mkdir()
    if best_state is not None:
        with (output / "best_weights" / f"epoch_{best_epoch:03d}.pth").open("xb") as f:
            torch.save(best_state, f)
    json_write(output / "initialization.json", {"backbone_state_sha256": initial_backbone_hash,
               "attention_initialization_seed": seed + 10000 if variant == "B" else None,
               "parameters": sum(p.numel() for p in model.parameters()), "pretrained": False})
    fields = ["epoch", "lr_used", "lr_next", "train_loss", "val_loss", "val_miou_global",
              "val_iou_sidewalk_global", "val_boundary_iou_author_fg_mean", "val_pixel_accuracy_global"]
    with (output / "metrics.csv").open("x", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    limit = settings["epoch_limit"] if stop_after_epoch is None else min(settings["epoch_limit"], stop_after_epoch)
    if limit < start_epoch:
        raise ValueError("Stop epoch precedes resume starting epoch")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, limit + 1):
        print(f"{variant} seed={seed} {mode} epoch={epoch}/{settings['epoch_limit']}", flush=True)
        lr_used = optimizer.param_groups[0]["lr"]
        row = {"epoch": epoch, "lr_used": lr_used}
        row.update(train_epoch(model, train_loader, optimizer, protocol, device))
        row["train_order"] = list(sampler.last_order)
        row.update(validate(model, val_loader, protocol, device))
        scheduler.step()
        row["lr_next"] = optimizer.param_groups[0]["lr"]
        if row["val_miou_global"] > best_miou:
            best_miou, best_epoch = row["val_miou_global"], epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            with (output / "best_weights" / f"epoch_{epoch:03d}.pth").open("xb") as f:
                torch.save(best_state, f)
        records.append(row)
        checkpoint = {"format": "attention_gate_epoch_v1", "settings": settings,
            "completed_epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "rng": capture_rng(gens), "best_miou": best_miou, "best_epoch": best_epoch,
            "best_model_state": best_state, "records": records}
        path = output / "checkpoints" / f"epoch_{epoch:03d}.pth"
        with path.open("xb") as f:
            torch.save(checkpoint, f)
        with (output / "metrics.csv").open("a", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=fields, extrasaction="ignore").writerow(row)
        json_write(output / "status.json", {"status": "running", "completed_epoch": epoch,
                   "last_checkpoint": str(path)}, exclusive=False)
        print(f"  loss={row['train_loss']:.6f}, val global mIoU={row['val_miou_global']:.6f}", flush=True)
    _, _, after_manifest = prepare(mode, protocol)
    if after_manifest != manifest:
        raise RuntimeError("Dataset identity changed during training")
    status = {"status": "complete" if limit == settings["epoch_limit"] else "stopped_at_epoch_boundary",
              "mode": mode, "formal_training": mode == "train", "completed_epoch": limit,
              "best_epoch": best_epoch, "best_miou": best_miou, "last_checkpoint": str(path),
              "final_state_sha256": state_digest(model.state_dict()), "wall_seconds": time.perf_counter() - started,
              "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
              "dataset_identity_unchanged": True}
    json_write(output / "status.json", status, exclusive=False)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "smoke", "train"), default="plan")
    parser.add_argument("--variant", choices=("A", "B"), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-epoch", type=int)
    args = parser.parse_args()
    if args.mode == "plan" and (args.resume or args.stop_after_epoch):
        parser.error("Plan mode does not resume or train")
    try:
        run(args.variant, args.seed, args.mode, args.output, args.resume, args.stop_after_epoch)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
