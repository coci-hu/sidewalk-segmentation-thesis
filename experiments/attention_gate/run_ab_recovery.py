"""Explicitly resumed A43/B43/A44/B44 queue, with verified operational I/O.

Frozen trainer, protocol and checkpoint identity are retained. Operational source
identity and fault-injection/numerical evidence are pinned separately, explicitly.
"""
import argparse
import copy
import json
import math
import msvcrt
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import common_runner as core
import common_runner_v2 as fast
import run_ab_repeats as previous
import safe_progress_io as io
from check_step03 import equal_tree
from evaluate_existing import timestamp

BASE_INTEGRITY = previous.assert_integrity
DOC_PYTHON = previous.DOC_PYTHON


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def integrity(job, plan):
    BASE_INTEGRITY(job, plan)
    baseline = read(job / 'integrity_before.json')
    assert all(core.digest(core.HERE / name) == sha for name, sha in baseline['interrupted_job'].items())
    evidence = read(plan['io_verification'])
    assert evidence['status'] == 'pass'
    assert evidence['adapter_sha256'] == core.digest(core.HERE / 'safe_progress_io.py')
    assert evidence['frozen_source_hashes'] == fast.sources()
    assert core.digest(plan['io_verification']) == plan['io_verification_sha256']


def prepare(job, source_job, proof):
    job, source_job, proof = previous.local(job), previous.local(source_job), Path(proof).resolve()
    assert not job.exists()
    old_plan = read(source_job / 'job_plan.json')
    paused = read(source_job / 'pause_status.json')
    checkpoint_path = Path(paused['checkpoint'])
    assert core.digest(checkpoint_path) == paused['checkpoint_sha256']
    assert paused['completed_epoch'] == 93 and paused['resume_next_epoch'] == 94
    assert fast.sources() == old_plan['source_hashes']
    passed = read(proof)
    assert passed['status'] == 'pass' and passed['frozen_source_hashes'] == fast.sources()
    assert passed['adapter_sha256'] == core.digest(core.HERE / 'safe_progress_io.py')
    assert passed['test_source_sha256'] == core.digest(core.HERE / 'check_progress_io.py')
    assert passed['checkpoint_sha256'] == paused['checkpoint_sha256']
    for name, sha in old_plan['orchestration_hashes'].items():
        assert core.digest(core.HERE / name) == sha
    checkpoint, _ = previous.verify_variant(checkpoint_path.parent.parent, 'A', 93, seed=43)
    assert checkpoint['settings'] == old_plan['expected_settings']['A_seed43']
    for seed, variant in previous.EXPECTED_ORDER:
        assert fast.expected_settings(variant, seed) == old_plan['expected_settings'][f'{variant}_seed{seed}']
    assert shutil.disk_usage(core.HERE).free > 10 * 2**30
    baseline = read(source_job / 'integrity_before.json')
    baseline['interrupted_job'] = previous.snapshot(source_job)
    job.mkdir(parents=True)
    core.json_write(job / 'integrity_before.json', baseline)
    plan = copy.deepcopy(old_plan)
    plan.update({'prepared': timestamp(), 'previous_interrupted_job': str(source_job),
        'resume_authorized_by_user': True, 'io_verification': str(proof),
        'io_verification_sha256': core.digest(proof),
        'scope': 'Resume A43 from completed93; then B43/A44/B44; stop before new modules',
        'scientific_sources_unchanged': True, 'operational_adapter': 'safe_progress_io_v1',
        'operational_adapter_role': 'bounded retry; skip only disposable progress/timing; critical failures remain fatal',
        'resume_checkpoints': {'A': {'path': str(checkpoint_path), 'sha256': core.digest(checkpoint_path)}},
        'first_epoch_source': str(checkpoint_path.parent.parent / 'first_epoch_verification.json'),
        'stop_file': str(job / 'STOP_AFTER_EPOCH'),
        'estimated_training_hours': old_plan['estimated_training_hours'] * 307 / 400,
        'remaining_epochs_at_start': 307,
        'orchestration_hashes': {name: core.digest(core.HERE / name) for name in
            ('run_ab_repeats.py', 'verify_seed42_training.py', 'run_ab_recovery.py',
             'safe_progress_io.py', 'run_training_io_safe.py', 'verify_resumed_epoch.py')}})
    plan['queue'] = []
    for seed, variant in previous.EXPECTED_ORDER:
        resume = str(checkpoint_path) if (seed, variant) == (43, 'A') else None
        folder = job / f'{variant}_seed{seed}_epochs{94 if resume else 1:03d}_100'
        plan['queue'].append({'seed': seed, 'variant': variant, 'folder': str(folder), 'resume': resume})
    plan['variant_folders'] = {'A': plan['queue'][0]['folder']}
    core.json_write(job / 'job_plan.json', plan)
    integrity(job, plan)
    core.json_write(job / 'preflight_verification.json', {'status': 'pass', 'at': timestamp(),
        'checkpoint93_full_verification_pass': True, 'data_protocol_runtime_and_settings_exact': True,
        'io_fault_and_fullsize_CUDA_equivalence_checks_pass': True,
        'sources_and_previous_artifacts_unchanged': True, 'resume_authorized': True})
    io.atomic_json(job / 'job_status.json', {'status': 'prepared', 'training_started': False,
        'remaining_epochs': 307, 'estimated_training_hours': plan['estimated_training_hours'],
        'completed_runs': 0, 'total_runs': 4})
    print(json.dumps({'status': 'prepared', 'remaining_epochs': 307,
        'estimated_training_hours': plan['estimated_training_hours']}, ensure_ascii=False), flush=True)


def verify_first(job, row, plan):
    folder, seed, variant = Path(row['folder']), row['seed'], row['variant']
    if row['resume']:
        result = subprocess.run([sys.executable, str(core.HERE / 'verify_resumed_epoch.py'),
            '--job', str(job), '--output', str(job / 'first_resumed_epoch_check.json')], check=False)
        if result.returncode:
            raise RuntimeError('First resumed full epoch failed CPU verification')
        proof = read(job / 'first_resumed_epoch_check.json')
        assert proof['status'] == 'pass' and proof['completed_epoch'] == 94
        assert read(plan['first_epoch_source'])['status'] == 'pass'
        shutil.copyfile(plan['first_epoch_source'], folder / 'first_epoch_verification.json')
        return
    cp = core.torch.load(folder / 'checkpoints/epoch_001.pth', map_location='cpu', weights_only=False)
    assert cp['completed_epoch'] == cp['scheduler_state']['last_epoch'] == 1
    assert cp['settings'] == plan['expected_settings'][f'{variant}_seed{seed}']
    assert len(cp['records']) == 1
    r = cp['records'][0]
    assert r['train_images'] == 2975 and r['val_images'] == 500
    assert r['train_batch_sizes'] == [4] * 743 + [3]
    assert sum(map(sum, r['val_confusion'])) == 500 * 768 * 1024
    gen = core.generators(seed)['sample_order']
    assert r['train_order'] == core.torch.randperm(2975, generator=gen).tolist()
    assert core.torch.equal(gen.get_state(), cp['rng']['generators']['sample_order'])
    assert all(v['step'].item() == 744 for v in cp['optimizer_state']['state'].values())
    assert all(v.item() == 744 for k, v in cp['model_state'].items() if k.endswith('num_batches_tracked'))
    assert all(bool(core.torch.isfinite(v).all()) for v in cp['model_state'].values() if v.is_floating_point())
    assert read(folder / 'initialization.json')['backbone_state_sha256'] == plan['initializations'][str(seed)]['shared_backbone_sha256']
    core.json_write(folder / 'first_epoch_verification.json', {'status': 'pass', 'seed': seed,
        'variant': variant, 'completed_epoch': 1, 'updates': 744,
        'sample_order_and_generator_exact': True, 'shared_backbone_matches_preflight': True,
        'checkpoint_sha256': core.digest(folder / 'checkpoints/epoch_001.pth')})


def journal(job, mode, seed=None):
    return previous.update_journal(job, mode, seed)


def train(job):
    plan = read(job / 'job_plan.json')
    assert read(job / 'job_status.json')['status'] == 'prepared'
    assert read(job / 'preflight_verification.json')['status'] == 'pass'
    assert [(r['seed'], r['variant']) for r in plan['queue']] == previous.EXPECTED_ORDER
    assert all(not Path(r['folder']).exists() for r in plan['queue'])
    integrity(job, plan)
    previous.assert_integrity = integrity
    env = os.environ.copy()
    env.update(PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
    began, completed, pending = time.perf_counter(), 0, []
    for row in plan['queue']:
        folder, seed, variant = Path(row['folder']), row['seed'], row['variant']
        if (job / 'STOP_AFTER_EPOCH').exists():
            io.atomic_json(job / 'job_status.json', {'status': 'stopped_at_epoch_boundary',
                'next_run': row, 'completed_runs': completed})
            return
        assert fast.expected_settings(variant, seed) == plan['expected_settings'][f'{variant}_seed{seed}']
        command = [sys.executable, '-u', str(core.HERE / 'run_training_io_safe.py'),
            '--mode', 'train', '--variant', variant, '--seed', str(seed), '--output', str(folder)]
        if row['resume']:
            assert core.digest(row['resume']) == plan['resume_checkpoints']['A']['sha256']
            command.extend(['--resume', row['resume']])
        with (job / f'{variant}_seed{seed}_console.log').open('x', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=str(core.PROJECT.parent), env=env,
                stdout=log, stderr=subprocess.STDOUT)
            first_checked = False
            try:
                while process.poll() is None:
                    live = {}
                    saved_epoch = 93 if row['resume'] else 0
                    try:
                        if (folder / 'training_progress.json').exists():
                            live = read(folder / 'training_progress.json')
                        if (folder / 'status.json').exists():
                            saved_epoch = read(folder / 'status.json')['completed_epoch']
                        threshold = 94 if row['resume'] else 1
                        if not first_checked and saved_epoch >= threshold:
                            verify_first(job, row, plan)
                            first_checked = True
                            if row['resume'] and not journal(job, 'recovery-check'):
                                pending.append('recovery_check')
                    except (PermissionError, json.JSONDecodeError):
                        pass
                    io.atomic_json(job / 'job_status.json', {'status': 'running', 'variant': variant,
                        'seed': seed, 'variant_folder': str(folder), 'saved_completed_epoch': saved_epoch,
                        'target_epochs': 100, 'pid': os.getpid(), 'child_pid': process.pid,
                        'completed_runs': completed, 'total_runs': 4, 'training': live,
                        'elapsed_seconds': time.perf_counter() - began,
                        'work_record_update_pending': pending})
                    time.sleep(15)
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=30)
                raise
            code = process.wait()
        if code == 75 and (job / 'STOP_AFTER_EPOCH').exists():
            io.atomic_json(job / 'job_status.json', {'status': 'stopped_at_epoch_boundary',
                'variant': variant, 'seed': seed, 'variant_folder': str(folder), 'completed_runs': completed})
            return
        if code:
            raise RuntimeError(f'{variant} seed{seed} failed {code}; see console; checkpoints retained')
        if not first_checked:
            verify_first(job, row, plan)
        completed += 1
        if variant == 'B':
            io.atomic_json(job / 'job_status.json', {'status': 'verifying', 'seed': seed,
                'pid': os.getpid(), 'completed_runs': completed, 'total_runs': 4})
            previous.verify_pair(job, seed)
            if not journal(job, 'repeat-pair', seed):
                pending.append(f'seed{seed}')
    previous.finish(job)
    ok = journal(job, 'repeat-completion')
    if not ok:
        pending.append('three_seed_completion')
    io.atomic_json(job / 'job_status.json', {'status': 'complete', 'verified': True,
        'three_seed_study_complete': True, 'completed_runs': 4, 'total_runs': 4,
        'completed': timestamp(), 'wall_seconds': time.perf_counter() - began,
        'work_record_updated': ok, 'work_record_update_pending': pending,
        'report': str(previous.JOURNAL), 'next_step': 'review three paired seeds before C; no new modules started'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('prepare', 'train'), required=True)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--source-job', type=Path)
    parser.add_argument('--io-proof', type=Path)
    args = parser.parse_args()
    job = previous.local(args.job)
    core.torch.set_num_threads(2)
    if args.mode == 'prepare':
        prepare(job, args.source_job, args.io_proof)
        return
    with Path(str(job) + '.stdout.log').open('x', encoding='utf-8', buffering=1) as out, \
            Path(str(job) + '.stderr.log').open('x', encoding='utf-8', buffering=1) as err:
        sys.stdout, sys.stderr = out, err
        io.install(fast)
        with (core.HERE / 'seed42_training.lock').open('a+b') as lock:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                with fast.keep_awake():
                    try:
                        train(job)
                    except BaseException:
                        io.atomic_json(job / 'job_status.json', {'status': 'failed', 'at': timestamp(),
                            'traceback': traceback.format_exc(), 'completed_checkpoints_preserved': True,
                            'no_automatic_retry': True})
                        traceback.print_exc()
                        raise
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    main()
