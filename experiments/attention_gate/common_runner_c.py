"""Versioned C control: same v1_2 scientific loop, only CA ReLU -> h-swish.

C has an explicit checkpoint format/settings/source identity. A/B checkpoints
are rejected before output creation. All frozen A/B source files stay intact.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import common_runner as core
import common_runner_v2 as fast
import safe_progress_io as io
from ca_hswish_control import build_c

FORMAT = 'attention_gate_C_epoch_v1'
REFERENCE = core.HERE / 'step_06_2026-10-14/preparation_001/author_reference'


def sources():
    result = fast.sources()
    for path in (Path(__file__), core.HERE / 'ca_hswish_control.py',
                 REFERENCE / 'coordatt.py', REFERENCE / 'source.json'):
        result[str(path.relative_to(core.PROJECT))] = core.digest(path)
    return result


def settings(seed, mode, manifest=None):
    base = fast.expected_settings('B', seed, mode)
    base.update(variant='C', source_hashes=sources(),
        control_id='CA_hswish_legacy_axes_v1',
        control_change='only ca.act ReLU -> parameter-free paper h-swish',
        operational_source_hashes={'safe_progress_io.py': core.digest(core.HERE / 'safe_progress_io.py')})
    if manifest is not None:
        assert base['manifest_sha256'] == hashlib.sha256(json.dumps(manifest, sort_keys=True).encode('utf-8')).hexdigest()
    return base


class StopAtEpochBoundary(Exception):
    pass


def run(seed, mode, output, resume=None, stop_after_epoch=None):
    output = Path(output).resolve()
    if mode not in ('train', 'smoke', 'plan') or seed not in (42, 43, 44):
        raise ValueError('Unknown C run mode/seed')
    if output == core.HERE or not output.is_relative_to(core.HERE) or output.exists():
        raise ValueError('Use a NEW local C output directory')
    protocol = fast.protocol()
    train, val, manifest = core.prepare('train' if mode == 'plan' else mode, protocol)
    expected = settings(seed, mode, manifest)
    checkpoint = None
    if resume:
        resume = Path(resume).resolve()
        if not resume.is_relative_to(core.HERE):
            raise ValueError('Checkpoint outside attention_gate')
        checkpoint = core.torch.load(resume, map_location='cpu', weights_only=False)
        if checkpoint.get('format') != FORMAT or checkpoint.get('settings') != expected:
            raise RuntimeError('C resume rejects A/B, different seed/data/protocol/runtime or sources')
    output.mkdir(parents=True)
    core.json_write(output / 'settings.json', expected)
    core.json_write(output / 'sample_manifest.json', manifest)
    core.json_write(output / 'environment.json', dict(expected['runtime'], deterministic_algorithms=True,
        cudnn_benchmark=False, amp=False, variant='C'))
    if mode == 'plan':
        core.json_write(output / 'status.json', {'status': 'plan_only', 'variant': 'C', 'training_started': False})
        return None
    if not core.torch.cuda.is_available():
        raise RuntimeError('C fixed protocol requires local CUDA')
    core.configure(seed)
    gens = core.generators(seed)
    model = build_c(seed).cuda()
    initial_hash = core.state_digest({k: v for k, v in model.state_dict().items() if not k.startswith('ca.')})
    optimizer, scheduler = core.optimizer_scheduler(model, protocol)
    train_loader, val_loader, sampler = core.make_loaders(train, val, protocol, gens)
    start, best, best_epoch, best_state, records = 1, -float('inf'), None, None, []
    if checkpoint:
        model.load_state_dict(checkpoint['model_state'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start = checkpoint['completed_epoch'] + 1
        best, best_epoch = checkpoint['best_miou'], checkpoint['best_epoch']
        best_state, records = checkpoint['best_model_state'], checkpoint['records']
        core.restore_rng(checkpoint['rng'], gens)
        core.json_write(output / 'resume_provenance.json', {'checkpoint': str(resume),
            'sha256': core.digest(resume), 'starting_epoch': start, 'variant': 'C'})
    (output / 'checkpoints').mkdir()
    (output / 'best_weights').mkdir()
    if best_state is not None:
        with (output / f'best_weights/epoch_{best_epoch:03d}.pth').open('xb') as f:
            core.torch.save(best_state, f)
    core.json_write(output / 'initialization.json', {'backbone_state_sha256': initial_hash,
        'attention_initialization_seed': seed + 10000, 'variant': 'C',
        'parameters': sum(p.numel() for p in model.parameters()), 'pretrained': False,
        'constructed_from_paired_initial_B_only': True, 'activation': 'paper h-swish'})
    fields = ['epoch', 'lr_used', 'lr_next', 'train_loss', 'val_loss', 'val_miou_global',
        'val_iou_sidewalk_global', 'val_boundary_iou_author_fg_mean', 'val_pixel_accuracy_global']
    with (output / 'metrics.csv').open('x', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    limit = expected['epoch_limit'] if stop_after_epoch is None else min(expected['epoch_limit'], stop_after_epoch)
    if limit < start:
        raise ValueError('Stop epoch precedes resume')
    began, times = time.perf_counter(), []
    pinned = sources()
    core.torch.cuda.reset_peak_memory_stats()

    class ProgressLoader:
        def __init__(self, loader, phase, epoch):
            self.loader, self.phase, self.epoch = loader, phase, epoch
        def __len__(self):
            return len(self.loader)
        def __iter__(self):
            for count, batch in enumerate(self.loader, 1):
                yield batch
                if count == 1 or count % 50 == 0 or count == len(self.loader):
                    io.atomic_json(output / 'training_progress.json', {'status': 'running', 'variant': 'C',
                        'seed': seed, 'epoch': self.epoch, 'target_epochs': expected['epoch_limit'],
                        'phase': self.phase, 'batches_completed': count, 'batches_total': len(self.loader),
                        'elapsed_seconds': time.perf_counter() - began})
                    print(f'C epoch={self.epoch} {self.phase} {count}/{len(self.loader)}', flush=True)

    try:
        for epoch in range(start, limit + 1):
            if (output.parent / 'STOP_AFTER_EPOCH').exists():
                raise StopAtEpochBoundary()
            assert sources() == pinned, 'C scientific source changed during run'
            assert expected['operational_source_hashes']['safe_progress_io.py'] == core.digest(core.HERE / 'safe_progress_io.py')
            print(f'C seed={seed} {mode} epoch={epoch}/{expected["epoch_limit"]}', flush=True)
            row = {'epoch': epoch, 'lr_used': optimizer.param_groups[0]['lr']}
            core.torch.cuda.synchronize()
            t = time.perf_counter()
            row.update(fast.BASE_TRAIN(model, ProgressLoader(train_loader, 'train', epoch), optimizer, protocol, core.torch.device('cuda:0')))
            row['train_order'] = list(sampler.last_order)
            core.torch.cuda.synchronize()
            train_seconds = time.perf_counter() - t
            t = time.perf_counter()
            row.update(fast.BASE_VALIDATE(model, ProgressLoader(val_loader, 'validation', epoch), protocol, core.torch.device('cuda:0')))
            core.torch.cuda.synchronize()
            times.append({'epoch': epoch, 'train_seconds': train_seconds, 'validation_seconds': time.perf_counter() - t})
            io.atomic_json(output / 'epoch_timing.json', times)
            scheduler.step()
            row['lr_next'] = optimizer.param_groups[0]['lr']
            if row['val_miou_global'] > best:
                best, best_epoch = row['val_miou_global'], epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                with (output / f'best_weights/epoch_{epoch:03d}.pth').open('xb') as f:
                    core.torch.save(best_state, f)
            records.append(row)
            state = {'format': FORMAT, 'settings': expected, 'completed_epoch': epoch,
                'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(), 'rng': core.capture_rng(gens),
                'best_miou': best, 'best_epoch': best_epoch, 'best_model_state': best_state, 'records': records}
            path = output / f'checkpoints/epoch_{epoch:03d}.pth'
            with path.open('xb') as f:
                core.torch.save(state, f)
            with (output / 'metrics.csv').open('a', newline='', encoding='utf-8-sig') as f:
                csv.DictWriter(f, fieldnames=fields, extrasaction='ignore').writerow(row)
            io.atomic_json(output / 'status.json', {'status': 'running', 'variant': 'C',
                'completed_epoch': epoch, 'last_checkpoint': str(path)})
            print(f'  loss={row["train_loss"]:.6f}, global mIoU={row["val_miou_global"]:.6f}', flush=True)
    except StopAtEpochBoundary:
        saved = json.loads((output / 'status.json').read_text()) if (output / 'status.json').exists() else {'completed_epoch': start - 1, 'last_checkpoint': str(resume)}
        io.atomic_json(output / 'status.json', dict(saved, status='stopped_at_epoch_boundary', variant='C'))
        raise
    _, _, after = core.prepare(mode, protocol)
    assert after == manifest
    io.atomic_json(output / 'status.json', {'status': 'complete' if limit == expected['epoch_limit'] else 'stopped_at_epoch_boundary',
        'variant': 'C', 'seed': seed, 'completed_epoch': limit, 'best_epoch': best_epoch,
        'last_checkpoint': str(path), 'wall_seconds': time.perf_counter() - began,
        'cuda_peak_allocated_mib': core.torch.cuda.max_memory_allocated() / 2**20,
        'dataset_identity_unchanged': True})
    core.json_write(output / 'io_health.json', io.STATS)
    return path


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('plan', 'smoke', 'train'), default='plan')
    p.add_argument('--seed', type=int, choices=(42, 43, 44), default=42)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--stop-after-epoch', type=int)
    a = p.parse_args()
    with fast.keep_awake():
        try:
            run(a.seed, a.mode, a.output, a.resume, a.stop_after_epoch)
        except StopAtEpochBoundary:
            raise SystemExit(75)
