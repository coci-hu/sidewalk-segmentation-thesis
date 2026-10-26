"""CPU audit of the first fully saved epoch after interrupted training resumes."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import argparse
import json
import math
from pathlib import Path
import torch
import common_runner as core
import common_runner_v2 as fast
from prepare_step06 import equal_tree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    job, output = args.job.resolve(), args.output.resolve()
    assert job.is_relative_to(core.HERE) and output.is_relative_to(core.HERE)
    assert not output.exists()
    torch.set_num_threads(2)
    assert not torch.cuda.is_available()
    plan = json.loads((job / 'job_plan.json').read_text(encoding='utf-8'))
    assert plan['source_hashes'] == fast.sources()
    source = Path(plan['resume_checkpoints']['A']['path'])
    assert core.digest(source) == plan['resume_checkpoints']['A']['sha256']
    old = torch.load(source, map_location='cpu', weights_only=False)
    resumed_epoch = old['completed_epoch'] + 1
    folder = Path(plan['variant_folders']['A'])
    path = folder / f'checkpoints/epoch_{resumed_epoch:03d}.pth'
    new = torch.load(path, map_location='cpu', weights_only=False)
    assert new['settings'] == old['settings']
    assert new['completed_epoch'] == new['scheduler_state']['last_epoch'] == resumed_epoch
    assert equal_tree(new['records'][:-1], old['records'])
    row = new['records'][-1]
    assert row['epoch'] == resumed_epoch
    assert row['train_images'] == 2975 and row['val_images'] == 500
    assert row['train_batch_sizes'] == [4] * 743 + [3]
    assert sorted(row['train_order']) == list(range(2975))
    assert sum(map(sum, row['val_confusion'])) == 500 * 768 * 1024
    order_gen = torch.Generator()
    order_gen.set_state(old['rng']['generators']['sample_order'])
    expected_order = torch.randperm(2975, generator=order_gen).tolist()
    assert row['train_order'] == expected_order
    assert torch.equal(order_gen.get_state(), new['rng']['generators']['sample_order'])
    expected_lr = 1e-6 + (1e-4 - 1e-6) * (1 + math.cos(math.pi * resumed_epoch / 100)) / 2
    assert abs(row['lr_next'] - expected_lr) < 1e-14
    assert all(state['step'].item() == resumed_epoch * 744 for state in new['optimizer_state']['state'].values())
    assert all(v.item() == resumed_epoch * 744 for k, v in new['model_state'].items() if k.endswith('num_batches_tracked'))
    assert all(bool(torch.isfinite(v).all()) for v in new['model_state'].values() if v.is_floating_point())
    assert any(not torch.equal(v, new['model_state'][k]) for k, v in old['model_state'].items() if v.is_floating_point())
    provenance = json.loads((folder / 'resume_provenance.json').read_text(encoding='utf-8'))
    assert provenance['sha256'] == plan['resume_checkpoints']['A']['sha256']
    assert provenance['starting_epoch'] == resumed_epoch
    assert fast.sources() == plan['source_hashes']
    core.json_write(output, {'status': 'pass', 'completed_epoch': resumed_epoch,
        'checkpoint': str(path), 'checkpoint_sha256': core.digest(path),
        'source_checkpoint': str(source), 'source_checkpoint_sha256': core.digest(source),
        'full_carried_history_exact': True, 'settings_and_frozen_sources_unchanged': True,
        'next_sample_order_and_sample_generator_state_exact': True,
        'optimizer_and_BN_steps': resumed_epoch * 744,
        'train_images': 2975, 'val_images': 500, 'finite_parameters': True,
        'formal_comparison_complete': False})
    print(json.dumps({'status': 'pass', 'completed_epoch': resumed_epoch, 'output': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
