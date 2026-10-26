"""Stage6: one exploratory C seed42 run, mandatory full-data resume at epoch1.

Verify six completed A/B runs, preserve old artifacts, require GPU smoke proof,
and stop after C100. No C43/44, new gates, paid services or deployments start.
"""
import argparse
import copy
import json
import math
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import common_runner as core
import common_runner_v2 as fast
import common_runner_c as c
import run_ab_repeats as ab
import safe_progress_io as io
from ca_hswish_control import build_c
from check_step03 import equal_tree
from evaluate_existing import protected_hashes, timestamp
from run_step05_benchmark import previous_hashes


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def integrity(job, plan):
    base = read(job / 'integrity_before.json')
    assert protected_hashes() == base['original']
    assert previous_hashes() == base['previous']
    for key in ('benchmark', 'seed42', 'interrupted_job', 'completed_ab'):
        assert all(core.digest(core.HERE / path) == sha for path, sha in base[key].items())
    assert c.sources() == plan['source_hashes']
    assert fast.sources() == plan['ab_source_hashes']
    for name, sha in plan['operational_hashes'].items():
        assert core.digest(core.HERE / name) == sha
    assert core.digest(plan['gpu_verification']) == plan['gpu_verification_sha256']


def prepare(job, ab_job, proof):
    assert job.is_relative_to(core.HERE) and not job.exists()
    assert read(ab_job / 'final_verification.json')['three_seed_study_complete']
    verification = read(proof)
    assert verification['status'] == 'pass' and verification['source_hashes'] == c.sources()
    assert verification['test_source_sha256'] == core.digest(core.HERE / 'check_step06_gpu.py')
    assert verification['operational_io_sha256'] == core.digest(core.HERE / 'safe_progress_io.py')
    assert verification['ab_results_sha256'] == core.digest(ab_job / 'three_seed_results.json')
    expected = c.settings(42, 'train')
    result = read(ab_job / 'three_seed_results.json')
    b42 = next(r for r in result['results'] if r['seed'] == 42 and r['variant'] == 'B')
    b = core.torch.load(b42['final_checkpoint'], map_location='cpu', weights_only=False)
    for key in ('protocol', 'manifest_sha256', 'runtime', 'epoch_limit', 'mode'):
        assert expected[key] == b['settings'][key]
    initial = build_c(42)
    before = read(ab_job / 'integrity_before.json')
    before['completed_ab'] = ab.snapshot(ab_job)
    job.mkdir(parents=True)
    core.json_write(job / 'integrity_before.json', before)
    plan = {'prepared': timestamp(), 'scope': 'C seed42 only, fresh initial state, full epoch1 then independent process epochs2..100',
        'ab_job': str(ab_job), 'ab_three_seed_results_sha256': core.digest(ab_job / 'three_seed_results.json'),
        'source_hashes': c.sources(), 'ab_source_hashes': fast.sources(), 'expected_settings': expected,
        'gpu_verification': str(proof), 'gpu_verification_sha256': core.digest(proof),
        'operational_hashes': {name: core.digest(core.HERE / name) for name in
            ('run_c_exploration.py', 'safe_progress_io.py', 'check_step06_gpu.py')},
        'initial_backbone_sha256': core.state_digest({k: v for k, v in initial.state_dict().items() if not k.startswith('ca.')}),
        'parameters': sum(p.numel() for p in initial.parameters()),
        'stage1': str(job / 'C_seed42_epoch001'), 'stage2': str(job / 'C_seed42_epochs002_100'),
        'B_seed42_final_checkpoint': b42['final_checkpoint'],
        'initialization': 'same initial B tensors; activation only, no trained B weights loaded',
        'automatic_other_seeds_or_new_modules': False, 'no_automatic_retry': True,
        'estimated_hours_reference_only': 13.5}
    core.json_write(job / 'job_plan.json', plan)
    integrity(job, plan)
    core.json_write(job / 'preflight_verification.json', {'status': 'pass',
        'three_seed_AB_complete': True, 'same_formal_protocol_data_runtime': True,
        'C_GPU_and_fresh_process_smoke_pass': True, 'C_fresh_initialization_only': True,
        'full_data_epoch1_and_resume_check_required': True})
    io.atomic_json(job / 'job_status.json', {'status': 'prepared', 'variant': 'C', 'seed': 42,
        'ab_three_seed_study_complete': True, 'C_three_seed_study_complete': False,
        'total_runs': 1, 'completed_runs': 0})
    print(json.dumps({'status': 'prepared', 'scope': plan['scope'], 'parameters': plan['parameters']}, ensure_ascii=False), flush=True)


def audit(folder, completed, plan):
    cp_path = folder / f'checkpoints/epoch_{completed:03d}.pth'
    cp = core.torch.load(cp_path, map_location='cpu', weights_only=False)
    assert cp['format'] == c.FORMAT and cp['settings'] == plan['expected_settings']
    assert cp['settings']['variant'] == 'C'
    assert cp['completed_epoch'] == cp['scheduler_state']['last_epoch'] == completed
    assert [r['epoch'] for r in cp['records']] == list(range(1, completed + 1))
    for row in cp['records']:
        assert row['train_images'] == 2975 and row['val_images'] == 500
        assert row['train_batch_sizes'] == [4] * 743 + [3]
        assert sorted(row['train_order']) == list(range(2975))
        assert sum(map(sum, row['val_confusion'])) == 500 * 768 * 1024
        matrix = row['val_confusion']
        x00, x01, x10, x11 = matrix[0] + matrix[1]
        fg = x11 / (x11 + x01 + x10) if x11 + x01 + x10 else 0
        bg = x00 / (x00 + x01 + x10) if x00 + x01 + x10 else 0
        assert abs((fg + bg) / 2 - row['val_miou_global']) < 1e-12
        lr = 1e-6 + (1e-4 - 1e-6) * (1 + math.cos(math.pi * row['epoch'] / 100)) / 2
        assert abs(row['lr_next'] - lr) < 1e-14
    assert all(s['step'].item() == completed * 744 for s in cp['optimizer_state']['state'].values())
    assert all(v.item() == completed * 744 for k, v in cp['model_state'].items() if k.endswith('num_batches_tracked'))
    assert all(bool(core.torch.isfinite(v).all()) for v in cp['model_state'].values() if v.is_floating_point())
    assert read(folder / 'initialization.json')['backbone_state_sha256'] == plan['initial_backbone_sha256']
    b = core.torch.load(plan['B_seed42_final_checkpoint'], map_location='cpu', weights_only=False)
    assert [r['train_order'] for r in cp['records']] == [r['train_order'] for r in b['records'][:completed]]
    generator = core.generators(42)['sample_order']
    for _ in range(completed):
        core.torch.randperm(2975, generator=generator)
    assert core.torch.equal(generator.get_state(), cp['rng']['generators']['sample_order'])
    if completed == 100:
        assert equal_tree(cp['rng'], b['rng'])
    return cp


def journal(job, mode):
    command = [str(ab.DOC_PYTHON), str(core.HERE / 'update_work_record.py'), '--mode', mode, '--job', str(job)]
    return subprocess.run(command, check=False).returncode == 0


def verify_resume(job, plan):
    first = core.torch.load(Path(plan['stage1']) / 'checkpoints/epoch_001.pth', map_location='cpu', weights_only=False)
    resumed = audit(Path(plan['stage2']), 2, plan)
    assert equal_tree(resumed['records'][:1], first['records'])
    assert resumed['settings'] == first['settings']
    generator = core.torch.Generator()
    generator.set_state(first['rng']['generators']['sample_order'])
    assert core.torch.randperm(2975, generator=generator).tolist() == resumed['records'][1]['train_order']
    provenance = read(Path(plan['stage2']) / 'resume_provenance.json')
    assert provenance['starting_epoch'] == 2
    assert provenance['sha256'] == core.digest(Path(plan['stage1']) / 'checkpoints/epoch_001.pth')
    core.json_write(job / 'formal_resume_verification.json', {'status': 'pass', 'variant': 'C', 'seed': 42,
        'completed_epoch': 2, 'optimizer_and_BN_updates': 1488,
        'full_data_2975_500_each_epoch': True, 'history_and_next_order_exact': True,
        'C_identity_and_sources_unchanged': True, 'fresh_process_resume': True})


def finish(job, plan):
    folder = Path(plan['stage2'])
    cp = audit(folder, 100, plan)
    assert read(job / 'formal_resume_verification.json')['status'] == 'pass'
    best = max(cp['records'], key=lambda r: r['val_miou_global'])
    assert best['epoch'] == cp['best_epoch']
    best_path = folder / f'best_weights/epoch_{best["epoch"]:03d}.pth'
    pure = core.torch.load(best_path, map_location='cpu', weights_only=True)
    assert equal_tree(pure, cp['best_model_state'])
    build_c(42).load_state_dict(pure, strict=True)
    result = {'variant': 'C', 'seed': 42, 'epochs': 100, 'best_epoch': best['epoch'],
        'best_global_miou_pct': best['val_miou_global'] * 100,
        'best_sidewalk_iou_pct': best['val_iou_sidewalk_global'] * 100,
        'best_boundary_iou_pct': best['val_boundary_iou_author_fg_mean'] * 100,
        'best_weights': str(best_path), 'final_checkpoint': str(folder / 'checkpoints/epoch_100.pth'),
        'final_checkpoint_sha256': core.digest(folder / 'checkpoints/epoch_100.pth')}
    rows = [r for r in read(Path(plan['ab_job']) / 'three_seed_results.json')['results'] if r['seed'] == 42]
    fields = ('best_global_miou_pct', 'best_sidewalk_iou_pct', 'best_boundary_iou_pct')
    delta = {f'C_minus_{r["variant"]}_percentage_points': {k: result[k] - r[k] for k in fields} for r in rows}
    integrity(job, plan)
    core.json_write(job / 'c_seed42_results.json', {'results': rows + [result], **delta,
        'interpretation': 'C seed42 exploration only; formal stable C claim needs matching43/44; h-swish control is not original method'})
    core.json_write(job / 'final_verification.json', {'status': 'pass', 'C_seed42_complete': True,
        'epochs': 100, 'optimizer_and_BN_updates': 74400, 'all_orders_and_final_rng_paired_with_B42': True,
        'best_weights_strict_load': True, 'full_dataset_resume_verified': True,
        'old_artifacts_unchanged': True, 'ab_three_seed_study_complete': True,
        'C_three_seed_study_complete': False})
    ok = journal(job, 'step06-completion')
    io.atomic_json(job / 'job_status.json', {'status': 'complete', 'verified': True, 'variant': 'C', 'seed': 42,
        'completed_runs': 1, 'total_runs': 1, 'saved_completed_epoch': 100, 'completed': timestamp(),
        'ab_three_seed_study_complete': True, 'C_three_seed_study_complete': False,
        'work_record_updated': ok, 'work_record_update_pending': not ok,
        'next_step': 'review exploratory C42 before C repeats, gates, deployment or new models'})


def train(job):
    plan = read(job / 'job_plan.json')
    assert read(job / 'job_status.json')['status'] == 'prepared'
    assert read(job / 'preflight_verification.json')['status'] == 'pass'
    integrity(job, plan)
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    began = time.perf_counter()
    pending = []
    for stage in (1, 2):
        folder = Path(plan[f'stage{stage}'])
        if (job / 'STOP_AFTER_EPOCH').exists():
            io.atomic_json(job / 'job_status.json', {'status': 'stopped_at_epoch_boundary', 'next_stage': stage, 'variant': 'C', 'seed': 42})
            return
        command = [sys.executable, '-u', str(core.HERE / 'common_runner_c.py'), '--mode', 'train', '--seed', '42', '--output', str(folder)]
        command += ['--stop-after-epoch', '1'] if stage == 1 else ['--resume', str(Path(plan['stage1']) / 'checkpoints/epoch_001.pth')]
        with (job / f'C_seed42_stage{stage}.log').open('x', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=str(core.PROJECT.parent), env=env, stdout=log, stderr=subprocess.STDOUT)
            checked = False
            try:
                while process.poll() is None:
                    live, completed = {}, stage - 1
                    try:
                        if (folder / 'training_progress.json').exists():
                            live = read(folder / 'training_progress.json')
                        if (folder / 'status.json').exists():
                            completed = read(folder / 'status.json')['completed_epoch']
                        if stage == 2 and completed >= 2 and not checked:
                            verify_resume(job, plan)
                            checked = True
                            if not journal(job, 'step06-check'):
                                pending.append('formal_resume_check')
                    except (json.JSONDecodeError, PermissionError):
                        pass
                    io.atomic_json(job / 'job_status.json', {'status': 'running', 'variant': 'C', 'seed': 42,
                        'stage': stage, 'variant_folder': str(folder), 'saved_completed_epoch': completed,
                        'pid': os.getpid(), 'child_pid': process.pid, 'target_epochs': 100, 'training': live,
                        'completed_runs': 0, 'total_runs': 1, 'ab_three_seed_study_complete': True,
                        'C_three_seed_study_complete': False, 'elapsed_seconds': time.perf_counter() - began,
                        'work_record_update_pending': pending})
                    time.sleep(15)
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=30)
                raise
            code = process.wait()
        if code == 75 and (job / 'STOP_AFTER_EPOCH').exists():
            io.atomic_json(job / 'job_status.json', {'status': 'stopped_at_epoch_boundary', 'variant': 'C', 'seed': 42,
                'variant_folder': str(folder), 'stage': stage})
            return
        if code:
            raise RuntimeError(f'C stage{stage} failed {code}; complete checkpoints preserved')
        if stage == 1:
            cp = audit(folder, 1, plan)
            core.json_write(job / 'first_formal_epoch_verification.json', {'status': 'pass', 'variant': 'C', 'seed': 42,
                'completed_epoch': 1, 'train_images': 2975, 'val_images': 500, 'updates': 744,
                'same_order_as_B42': True, 'initial_backbone_exact': True,
                'checkpoint_sha256': core.digest(folder / 'checkpoints/epoch_001.pth')})
        elif not checked:
            verify_resume(job, plan)
    finish(job, plan)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('prepare', 'train'), required=True)
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--ab-job', type=Path)
    p.add_argument('--gpu-proof', type=Path)
    a = p.parse_args()
    job = a.job.resolve()
    assert job.is_relative_to(core.HERE) and job != core.HERE
    core.torch.set_num_threads(2)
    if a.mode == 'prepare':
        prepare(job, a.ab_job.resolve(), a.gpu_proof.resolve())
        return
    with Path(str(job) + '.stdout.log').open('x', encoding='utf-8', buffering=1) as out, Path(str(job) + '.stderr.log').open('x', encoding='utf-8', buffering=1) as err:
        sys.stdout, sys.stderr = out, err
        with (core.HERE / 'seed42_training.lock').open('a+b') as lock:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                with fast.keep_awake():
                    try:
                        train(job)
                    except BaseException:
                        io.atomic_json(job / 'job_status.json', {'status': 'failed', 'variant': 'C', 'seed': 42,
                            'at': timestamp(), 'traceback': traceback.format_exc(), 'completed_checkpoints_preserved': True,
                            'no_automatic_retry': True})
                        traceback.print_exc()
                        raise
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    main()
