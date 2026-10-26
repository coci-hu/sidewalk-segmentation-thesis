"""Full-resolution CUDA checks and fresh-process C checkpoint resume test.

Smoke uses the production code path on 7 train/3 val images, two epochs. It is
not a formal accuracy result; full-data continuity is checked by the supervisor.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import common_runner as core
import common_runner_v2 as fast
import common_runner_c as c
from ca_hswish_control import build_c, PaperHardSwish
from check_step03 import equal_tree


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ab-job', type=Path, required=True)
    p.add_argument('--reuse-smoke-runs', type=Path)
    a = p.parse_args()
    output = a.output.resolve()
    assert output.is_relative_to(core.HERE) and not output.exists()
    output.mkdir(parents=True)
    core.torch.set_num_threads(2)
    pinned = c.sources()
    checks = []
    def passed(name, detail):
        checks.append({'check': name, 'status': 'pass', 'detail': detail})
        print('PASS ' + name, flush=True)
    results = json.loads((a.ab_job / 'three_seed_results.json').read_text(encoding='utf-8'))
    assert json.loads((a.ab_job / 'final_verification.json').read_text(encoding='utf-8'))['status'] == 'pass'
    for row in results['results']:
        assert core.digest(row['final_checkpoint']) == row['final_checkpoint_sha256']
    for seed in (42, 43, 44):
        b, model = core.build_paired_model('B', seed), build_c(seed)
        assert equal_tree(b.state_dict(), model.state_dict())
        assert isinstance(model.ca.act, PaperHardSwish)
    passed('completed_AB_identity_and_three_paired_initial_states', 'Six final checkpoint hashes match; C uses same initial tensors as B for42/43/44')
    core.configure(42)
    image = core.torch.randn(4, 3, 768, 1024, device='cuda')
    target = core.torch.zeros(4, 768, 1024, dtype=core.torch.long, device='cuda')
    rng = core.torch.cuda.get_rng_state_all()
    reference = None
    for variant in ('B', 'C_relu_restored'):
        model = core.build_paired_model('B', 42) if variant == 'B' else build_c(42)
        if variant != 'B':
            model.ca.act = core.torch.nn.ReLU(inplace=True)
        model = model.cuda().train()
        core.torch.cuda.set_rng_state_all(rng)
        loss = core.combined_loss(model(image), target, fast.protocol())
        loss.backward()
        state = {'loss': loss.item(), 'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            'gradients': {k: p.grad.detach().cpu().clone() for k, p in model.named_parameters()}}
        if reference is None:
            reference = state
        else:
            assert equal_tree(reference, state)
        del model, loss
    del image, target, reference, state
    gc.collect()
    core.torch.cuda.empty_cache()
    passed('CUDA_full_resolution_ReLU_restoration', '768x1024 batch4: restored C reproduces B loss, BN/state and all parameter gradients bitwise')
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    def child(name, extra):
        folder = output / name
        command = [sys.executable, '-u', str(core.HERE / 'common_runner_c.py'), '--mode', 'smoke',
            '--seed', '42', '--output', str(folder)] + extra
        with (output / f'{name}.log').open('x', encoding='utf-8') as log:
            subprocess.run(command, env=env, cwd=str(core.PROJECT.parent), stdout=log,
                stderr=subprocess.STDOUT, check=True)
        return folder
    if a.reuse_smoke_runs:
        source = a.reuse_smoke_runs.resolve()
        assert source.is_relative_to(core.HERE)
        continuous, first, resumed = (source / name for name in ('continuous_2epochs', 'first_epoch', 'fresh_process_epoch2'))
        for folder, epoch in ((continuous, 2), (first, 1), (resumed, 2)):
            assert json.loads((folder / 'status.json').read_text(encoding='utf-8'))['completed_epoch'] == epoch
            cp = core.torch.load(folder / f'checkpoints/epoch_{epoch:03d}.pth', map_location='cpu', weights_only=False)
            assert cp['settings'] == c.settings(42, 'smoke')
            del cp
    else:
        continuous = child('continuous_2epochs', [])
        first = child('first_epoch', ['--stop-after-epoch', '1'])
        resumed = child('fresh_process_epoch2', ['--resume', str(first / 'checkpoints/epoch_001.pth')])
    direct = core.torch.load(continuous / 'checkpoints/epoch_002.pth', map_location='cpu', weights_only=False)
    restored = core.torch.load(resumed / 'checkpoints/epoch_002.pth', map_location='cpu', weights_only=False)
    assert equal_tree(direct, restored)
    assert direct['format'] == c.FORMAT and direct['settings']['variant'] == 'C'
    assert all(r['train_images'] == 7 and r['val_images'] == 3 and r['train_batch_sizes'] == [4, 3] for r in direct['records'])
    assert all(v['step'].item() == 4 for v in direct['optimizer_state']['state'].values())
    ca_states = [v for k, v in direct['model_state'].items() if k.startswith('ca.') and v.is_floating_point()]
    assert all(bool(core.torch.isfinite(v).all()) for v in ca_states)
    initial = build_c(42).state_dict()
    assert any(not core.torch.equal(v, initial[k]) for k, v in direct['model_state'].items() if k.startswith('ca.') and v.is_floating_point())
    peak = json.loads((continuous / 'status.json').read_text(encoding='utf-8'))['cuda_peak_allocated_mib']
    passed('C_fullsize_updates_and_fresh_process_resume_exact', {'full_checkpoint_bitwise_equal': True,
        'smoke_train_images': 7, 'smoke_val_images': 3, 'production_loop': True,
        'epochs': 2, 'optimizer_updates': 4, 'CA_parameters_updated': True, 'peak_allocated_MiB': peak})
    for row in results['results'][:2]:
        rejected = False
        destination = output / f'rejected_{row["variant"]}'
        try:
            c.run(42, 'train', destination, row['final_checkpoint'])
        except RuntimeError as error:
            assert 'rejects A/B' in str(error)
            rejected = True
        assert rejected and not destination.exists()
    rejected = False
    try:
        c.run(43, 'smoke', output / 'rejected_seed', first / 'checkpoints/epoch_001.pth')
    except RuntimeError:
        rejected = True
    assert rejected and not (output / 'rejected_seed').exists()
    passed('A_B_and_wrong_seed_checkpoint_rejection', 'No output created for rejected checkpoint; all C metadata stays C')
    assert c.sources() == pinned
    core.json_write(output / 'verification.json', {'status': 'pass', 'checks': checks,
        'source_hashes': pinned, 'test_source_sha256': core.digest(__file__),
        'operational_io_sha256': core.digest(core.HERE / 'safe_progress_io.py'),
        'ab_results_sha256': core.digest(a.ab_job / 'three_seed_results.json'),
        'full_dataset_formal_training_started_by_test': False,
        'smoke_runs_reused_from': str(a.reuse_smoke_runs.resolve()) if a.reuse_smoke_runs else None,
        'formal_first_epoch_and_resume_audit_required': True})


if __name__ == '__main__':
    main()
