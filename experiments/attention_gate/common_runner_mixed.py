"""Versioned D/E experiment; frozen A/B/C code and artifacts are read only.

D uses fixed alpha=.1, E learns sigmoid(beta) from exactly the same value.
Formal training requires a separate GPU/resume acceptance proof. CPU tests
cannot unlock it. Core data, loss, evaluation and selection remain unchanged.
"""
import argparse
import csv
import json
import msvcrt
import time
from pathlib import Path

import common_runner as core
import common_runner_v2 as fast
import common_runner_c as reference
import mixed_ca_candidate as candidate
import safe_progress_io as io

FORMAT = 'attention_gate_mixed_epoch_v1'


def sources():
    result = reference.sources()
    for path in (Path(__file__), Path(candidate.__file__)):
        result[str(path.relative_to(core.PROJECT))] = core.digest(path)
    return result


def settings(variant, seed, mode, manifest=None):
    if variant not in ('D', 'E') or seed not in (42, 43, 44):
        raise ValueError('Unknown candidate or seed')
    base = reference.settings(seed, mode, manifest)
    base.update(variant=variant, source_hashes=sources(),
        control_id='CA_hswish_scalar_mixture_v1',
        control_change='fixed alpha=.1' if variant == 'D' else 'learned sigmoid(beta), initial alpha=.1',
        candidate_extension={'formula': '(1-alpha)*x + alpha*CA_hswish(x)',
            'alpha_initial': .1, 'alpha_learned': variant == 'E',
            'beta_weight_decay': 0.0 if variant == 'E' else None,
            'shared_parameter_weight_decay': .01, 'beta_lr_and_schedule': 'same as shared parameters',
            'reference': 'fresh C initialization, never trained C weights'})
    return base


def optimizer_scheduler(model, protocol):
    s = protocol['formal_training']
    optimizer = core.torch.optim.AdamW(candidate.parameter_groups(model, s['weight_decay']),
        lr=s['learning_rate'], betas=tuple(s['betas']), eps=s['epsilon'],
        amsgrad=s['amsgrad'], foreach=False)
    scheduler = core.torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        T_max=s['scheduler_T_max'], eta_min=s['scheduler_eta_min'])
    return optimizer, scheduler


def validate_resume(checkpoint, expected):
    if checkpoint.get('format') != FORMAT or checkpoint.get('settings') != expected:
        raise RuntimeError('Resume identity mismatch: variant/seed/data/protocol/runtime/source')
    epoch = checkpoint['completed_epoch']
    if checkpoint['scheduler_state']['last_epoch'] != epoch:
        raise RuntimeError('Scheduler position differs from saved epoch')
    if [r['epoch'] for r in checkpoint['records']] != list(range(1, epoch + 1)):
        raise RuntimeError('Incomplete checkpoint history')
    expected_groups = 2 if expected['variant'] == 'E' else 1
    groups = checkpoint['optimizer_state']['param_groups']
    if len(groups) != expected_groups or groups[0]['weight_decay'] != .01:
        raise RuntimeError('Optimizer groups differ from predeclared experiment')
    if expected_groups == 2 and (groups[1]['weight_decay'] != 0 or len(groups[1]['params']) != 1):
        raise RuntimeError('E beta must be the sole parameter in the zero-decay group')
    if not 1 <= checkpoint['best_epoch'] <= epoch:
        raise RuntimeError('Invalid best epoch')
    updates = sum(len(r['train_batch_sizes']) for r in checkpoint['records'])
    if any(int(state['step']) != updates for state in checkpoint['optimizer_state']['state'].values()):
        raise RuntimeError('Optimizer step differs from completed updates')


def checkpoint_state(expected, epoch, model, optimizer, scheduler, gens, best, best_epoch, best_state, records):
    return dict(format=FORMAT, settings=expected, completed_epoch=epoch,
        model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(), rng=core.capture_rng(gens),
        best_miou=best, best_epoch=best_epoch, best_model_state=best_state, records=records)


def require_acceptance(path):
    if path is None:
        raise RuntimeError('Formal D/E training requires GPU and fresh-process resume acceptance')
    proof = json.loads(Path(path).read_text(encoding='utf-8'))
    if proof.get('status') != 'pass' or proof.get('source_hashes') != sources():
        raise RuntimeError('D/E acceptance missing or stale')
    for name in ('gpu_fullsize_D_E', 'fresh_process_full_state_resume_D_E', 'sensitivity_and_diagnosis_reviewed'):
        if proof.get(name) is not True:
            raise RuntimeError('D/E acceptance incomplete: ' + name)


def run(variant, seed, mode, output, resume=None, stop_after_epoch=None, acceptance_proof=None):
    output = Path(output).resolve()
    if mode not in ('plan', 'smoke', 'train'):
        raise ValueError('Unknown run mode')
    if output == core.HERE or not output.is_relative_to(core.HERE) or output.exists():
        raise ValueError('Use a NEW output directory inside attention_gate')
    if mode == 'train':
        require_acceptance(acceptance_proof)
    protocol = fast.protocol()
    train, val, manifest = core.prepare('train' if mode == 'plan' else mode, protocol)
    expected = settings(variant, seed, mode, manifest)
    checkpoint = None
    if resume is not None:
        resume = Path(resume).resolve()
        if not resume.is_relative_to(core.HERE):
            raise ValueError('Checkpoint outside experiment root')
        checkpoint = core.torch.load(resume, map_location='cpu', weights_only=False)
        validate_resume(checkpoint, expected)
    if mode != 'plan' and not core.torch.cuda.is_available():
        raise RuntimeError('Fixed protocol training requires CUDA')
    output.mkdir(parents=True)
    core.json_write(output / 'settings.json', expected)
    core.json_write(output / 'sample_manifest.json', manifest)
    if mode == 'plan':
        core.json_write(output / 'status.json', dict(status='plan_only', training_started=False, variant=variant))
        return None
    core.json_write(output / 'environment.json', dict(expected['runtime'], deterministic_algorithms=True,
        cudnn_benchmark=False, amp=False, variant=variant))
    core.configure(seed)
    gens = core.generators(seed)
    model = candidate.build_candidate(variant, seed).cuda()
    initial_hash = core.state_digest({k: v for k, v in model.state_dict().items() if not k.startswith('ca.')})
    optimizer, scheduler = optimizer_scheduler(model, protocol)
    train_loader, val_loader, sampler = core.make_loaders(train, val, protocol, gens)
    start, best, best_epoch, best_state, records = 1, -float('inf'), None, None, []
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start = checkpoint['completed_epoch'] + 1
        best, best_epoch = checkpoint['best_miou'], checkpoint['best_epoch']
        best_state, records = checkpoint['best_model_state'], checkpoint['records']
        core.restore_rng(checkpoint['rng'], gens)
        core.json_write(output / 'resume_provenance.json', dict(checkpoint=str(resume),
            sha256=core.digest(resume), starting_epoch=start, variant=variant))
    (output / 'checkpoints').mkdir()
    (output / 'best_weights').mkdir()
    if best_state is not None:
        with (output / f'best_weights/epoch_{best_epoch:03d}.pth').open('xb') as f:
            core.torch.save(best_state, f)
    core.json_write(output / 'initialization.json', dict(backbone_state_sha256=initial_hash,
        attention_initialization_seed=seed + 10000, variant=variant, pretrained=False,
        parameters=sum(p.numel() for p in model.parameters()), alpha=float(model.ca.alpha().detach()),
        fresh_initialization_before_resume=checkpoint is None))
    fields = ['epoch', 'lr_used', 'lr_next', 'train_loss', 'val_loss', 'val_miou_global',
        'val_iou_sidewalk_global', 'val_boundary_iou_author_fg_mean', 'val_pixel_accuracy_global',
        'alpha_before', 'alpha_after']
    with (output / 'metrics.csv').open('x', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    limit = expected['epoch_limit'] if stop_after_epoch is None else min(expected['epoch_limit'], stop_after_epoch)
    if limit < start:
        raise ValueError('Stop epoch precedes resume')
    began, times, pinned = time.perf_counter(), [], sources()
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
                    io.atomic_json(output / 'training_progress.json', dict(status='running', variant=variant,
                        seed=seed, epoch=self.epoch, target_epochs=expected['epoch_limit'], phase=self.phase,
                        batches_completed=count, batches_total=len(self.loader), elapsed_seconds=time.perf_counter()-began))

    try:
        for epoch in range(start, limit + 1):
            if (output.parent / 'STOP_AFTER_EPOCH').exists():
                raise reference.StopAtEpochBoundary()
            if sources() != pinned or expected['operational_source_hashes']['safe_progress_io.py'] != core.digest(core.HERE / 'safe_progress_io.py'):
                raise RuntimeError('D/E sources changed during run')
            row = dict(epoch=epoch, lr_used=optimizer.param_groups[0]['lr'], alpha_before=float(model.ca.alpha().detach()))
            core.torch.cuda.synchronize()
            t = time.perf_counter()
            row.update(fast.BASE_TRAIN(model, ProgressLoader(train_loader, 'train', epoch), optimizer, protocol, core.torch.device('cuda:0')))
            row['train_order'] = list(sampler.last_order)
            row['alpha_after'] = float(model.ca.alpha().detach())
            core.torch.cuda.synchronize()
            training_seconds = time.perf_counter() - t
            t = time.perf_counter()
            row.update(fast.BASE_VALIDATE(model, ProgressLoader(val_loader, 'validation', epoch), protocol, core.torch.device('cuda:0')))
            core.torch.cuda.synchronize()
            times.append(dict(epoch=epoch, train_seconds=training_seconds, validation_seconds=time.perf_counter()-t))
            io.atomic_json(output / 'epoch_timing.json', times)
            scheduler.step()
            row['lr_next'] = optimizer.param_groups[0]['lr']
            if row['val_miou_global'] > best:
                best, best_epoch = row['val_miou_global'], epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                with (output / f'best_weights/epoch_{epoch:03d}.pth').open('xb') as f:
                    core.torch.save(best_state, f)
            records.append(row)
            state = checkpoint_state(expected, epoch, model, optimizer, scheduler, gens, best, best_epoch, best_state, records)
            path = output / f'checkpoints/epoch_{epoch:03d}.pth'
            with path.open('xb') as f:
                core.torch.save(state, f)
            with (output / 'metrics.csv').open('a', newline='', encoding='utf-8-sig') as f:
                csv.DictWriter(f, fieldnames=fields, extrasaction='ignore').writerow(row)
            io.atomic_json(output / 'status.json', dict(status='running', variant=variant, completed_epoch=epoch, last_checkpoint=str(path)))
            print(f'{variant} seed={seed} epoch={epoch} mIoU={best:.6f} alpha={row["alpha_after"]:.8f}', flush=True)
    except reference.StopAtEpochBoundary:
        saved = json.loads((output / 'status.json').read_text()) if (output / 'status.json').exists() else dict(completed_epoch=start-1, last_checkpoint=str(resume))
        io.atomic_json(output / 'status.json', dict(saved, status='stopped_at_epoch_boundary', variant=variant))
        raise
    _, _, after = core.prepare(mode, protocol)
    if after != manifest:
        raise RuntimeError('Data identity changed during D/E run')
    if sources() != pinned:
        raise RuntimeError('Sources changed before final verification')
    io.atomic_json(output / 'status.json', dict(status='complete' if limit == expected['epoch_limit'] else 'stopped_at_epoch_boundary',
        variant=variant, seed=seed, completed_epoch=limit, best_epoch=best_epoch, last_checkpoint=str(path),
        wall_seconds=time.perf_counter()-began, cuda_peak_allocated_mib=core.torch.cuda.max_memory_allocated()/2**20,
        dataset_identity_unchanged=True))
    core.json_write(output / 'io_health.json', io.STATS)
    return path


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=('D', 'E'), required=True)
    p.add_argument('--seed', type=int, choices=(42, 43, 44), default=42)
    p.add_argument('--mode', choices=('plan', 'smoke', 'train'), default='plan')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--stop-after-epoch', type=int)
    p.add_argument('--acceptance-proof', type=Path)
    a = p.parse_args()
    if a.mode == 'plan':
        run(a.variant, a.seed, a.mode, a.output, a.resume, a.stop_after_epoch)
    else:
        with (core.HERE / 'seed42_training.lock').open('a+b') as lock:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                with fast.keep_awake():
                    run(a.variant, a.seed, a.mode, a.output, a.resume, a.stop_after_epoch, a.acceptance_proof)
            except reference.StopAtEpochBoundary:
                raise SystemExit(75)
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
