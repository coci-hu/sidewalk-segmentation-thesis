"""Actual D/E network, optimizer and serialization checks; no CUDA allocation.

Reduced synthetic images exercise recovery, not segmentation accuracy. Full
768x1024 CPU forward is separate. Neither replaces full-size GPU acceptance.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
import common_runner as core
import common_runner_mixed as mixed
import common_runner_v2 as fast
from check_step03 import equal_tree
from mixed_ca_candidate import build_candidate


def test_settings(variant):
    result = mixed.settings(variant, 42, 'smoke')
    result['acceptance_test_scope'] = 'CPU synthetic 128x256 checkpoint mechanics, not formal training'
    return result


def worker(variant, output, through, resume=None):
    torch.set_num_threads(2)
    core.configure(42)
    protocol = fast.protocol()
    expected, gens = test_settings(variant), core.generators(42)
    model = build_candidate(variant, 42)
    optimizer, scheduler = mixed.optimizer_scheduler(model, protocol)
    data_rng = torch.Generator().manual_seed(71024)
    images = torch.randn(7, 3, 128, 256, generator=data_rng)
    labels = torch.randint(0, 2, (7, 128, 256), generator=data_rng)
    start, records, best, best_epoch, best_state = 1, [], -float('inf'), None, None
    if resume:
        cp = torch.load(resume, map_location='cpu', weights_only=False)
        mixed.validate_resume(cp, expected)
        model.load_state_dict(cp['model_state'], strict=True)
        optimizer.load_state_dict(cp['optimizer_state'])
        scheduler.load_state_dict(cp['scheduler_state'])
        start, records = cp['completed_epoch'] + 1, cp['records']
        best, best_epoch, best_state = cp['best_miou'], cp['best_epoch'], cp['best_model_state']
        core.restore_rng(cp['rng'], gens)
    output.mkdir()
    for epoch in range(start, through + 1):
        order = torch.randperm(7, generator=gens['sample_order']).tolist()
        batches = [(images[order[i:i+4]], labels[order[i:i+4]]) for i in (0, 4)]
        row = dict(epoch=epoch, train_order=order, lr_used=optimizer.param_groups[0]['lr'],
            alpha_before=float(model.ca.alpha().detach()))
        row.update(fast.BASE_TRAIN(model, batches, optimizer, protocol, torch.device('cpu')))
        row['alpha_after'] = float(model.ca.alpha().detach())
        row.update(fast.BASE_VALIDATE(model, [(images[:1], labels[:1])], protocol, torch.device('cpu')))
        scheduler.step()
        row['lr_next'] = optimizer.param_groups[0]['lr']
        if row['val_miou_global'] > best:
            best, best_epoch = row['val_miou_global'], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        records.append(row)
        cp = mixed.checkpoint_state(expected, epoch, model, optimizer, scheduler, gens, best, best_epoch, best_state, records)
        with (output / f'epoch_{epoch:03d}.pth').open('xb') as f:
            torch.save(cp, f)


def check(output):
    if output.exists() or output == core.HERE or not output.is_relative_to(core.HERE):
        raise ValueError('Use a new local output directory')
    output.mkdir(parents=True)
    before, checks, began = mixed.sources(), [], time.perf_counter()
    torch.set_num_threads(2)
    for variant in ('D', 'E'):
        for name, through, resume in (('continuous', 2, None), ('first', 1, None),
                ('resumed', 2, output / f'{variant}_first/epoch_001.pth')):
            command = [sys.executable, __file__, '--worker', '--variant', variant,
                '--output', str(output / f'{variant}_{name}'), '--through', str(through)]
            if resume:
                command += ['--resume', str(resume)]
            with (output / f'{variant}_{name}.log').open('x', encoding='utf-8') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=240,
                    env=dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8'))
            if result.returncode:
                raise RuntimeError('CPU child failed: ' + str(output / f'{variant}_{name}.log'))
        uninterrupted = torch.load(output / f'{variant}_continuous/epoch_002.pth', weights_only=False)
        resumed = torch.load(output / f'{variant}_resumed/epoch_002.pth', weights_only=False)
        assert equal_tree(uninterrupted, resumed), 'Full state changed after fresh-process resume'
        assert uninterrupted['records'][0]['train_batch_sizes'] == [4, 3]
        checks.append(variant + '_fresh_process_exact_model_optimizer_scheduler_rng_best_and_history_with_tail')
        if variant == 'E':
            assert uninterrupted['records'][-1]['alpha_after'] != uninterrupted['records'][0]['alpha_before']
            assert uninterrupted['optimizer_state']['param_groups'][1]['weight_decay'] == 0
            checks.append('E_beta_updates_and_zero_decay_survives_resume')
        for field, value in (('variant', 'E' if variant == 'D' else 'D'), ('seed', 43),
                ('manifest_sha256', 'different'), ('source_hashes', {})):
            bad = copy.deepcopy(uninterrupted['settings'])
            bad[field] = value
            try:
                mixed.validate_resume(uninterrupted, bad)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Resume accepted mismatched ' + field)
        bad_cp = dict(uninterrupted, format='attention_gate_C_epoch_v1')
        try:
            mixed.validate_resume(bad_cp, uninterrupted['settings'])
        except RuntimeError:
            pass
        else:
            raise AssertionError('C checkpoint accepted')
        checks.append(variant + '_wrong_variant_seed_data_source_and_C_checkpoint_rejected')
        print('PASS ' + variant + ' fresh-process CPU resume and rejection checks', flush=True)
    core.configure(42)
    x = torch.randn(1, 3, 768, 1024)
    with torch.inference_mode():
        d = build_candidate('D', 42).eval()(x)
        e = build_candidate('E', 42).eval()(x)
    assert d.shape == (1, 2, 768, 1024) and torch.isfinite(d).all() and torch.equal(d, e)
    checks.append('full_768x1024_CPU_network_forward_finite_D_E_initial_logits_exact')
    gate = output / 'cpu_only_acceptance_rejection.json'
    core.json_write(gate, dict(status='pass', source_hashes=before,
        gpu_fullsize_D_E=False, fresh_process_full_state_resume_D_E=False, sensitivity_and_diagnosis_reviewed=False))
    try:
        mixed.require_acceptance(gate)
    except RuntimeError:
        pass
    else:
        raise AssertionError('CPU-only proof unlocked formal GPU training')
    checks.append('CPU_proof_cannot_unlock_formal_training')
    assert before == mixed.sources()
    core.json_write(output / 'verification.json', dict(status='pass', checks=checks,
        scope='CPU tests only, synthetic training and full-resolution forward',
        source_hashes=before, check_sha256=core.digest(__file__), cuda_allocated=False,
        formal_training_ready=False, formal_training_started=False, wall_seconds=time.perf_counter()-began,
        pending=['GPU full-size D/E smoke', 'GPU independent-process full-state resume',
            'C44 completion and nine-model sensitivity/diagnosis review', 'paired D42/E42 supervisor acceptance']))
    print(json.dumps({'status': 'pass', 'checks': len(checks), 'formal_training_ready': False}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--variant', choices=('D', 'E'))
    p.add_argument('--through', type=int)
    p.add_argument('--resume', type=Path)
    a = p.parse_args()
    if a.worker:
        worker(a.variant, a.output.resolve(), a.through, a.resume)
    else:
        check(a.output.resolve())
