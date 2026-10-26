"""C43/44 orchestration only. The scientific C training source stays frozen.

Each seed: full epoch1, independent resume2..100, paired audit against its B.
No retry on failure. STOP_AFTER_EPOCH preserves a completed checkpoint.
"""
import argparse
import copy
import json
import math
import msvcrt
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback

import common_runner as core
import common_runner_v2 as fast
import common_runner_c as c
import run_c_exploration as old
import run_ab_repeats as ab
import safe_progress_io as io
from ca_hswish_control import build_c
from check_step03 import equal_tree
from evaluate_existing import timestamp

read = old.read
FIELDS = ab.FIELDS


def verify_checkpoint(cp, completed, seed, expected, reference):
    assert cp['format'] == c.FORMAT and cp['settings'] == expected
    assert expected['variant'] == 'C' and expected['seed'] == seed
    assert reference['settings']['variant'] == 'B' and reference['settings']['seed'] == seed
    assert cp['completed_epoch'] == cp['scheduler_state']['last_epoch'] == completed
    assert [r['epoch'] for r in cp['records']] == list(range(1, completed + 1))
    for row in cp['records']:
        assert row['train_images'] == 2975 and row['val_images'] == 500
        assert row['train_batch_sizes'] == [4] * 743 + [3]
        assert sorted(row['train_order']) == list(range(2975))
        matrix = row['val_confusion']
        assert sum(map(sum, matrix)) == 500 * 768 * 1024
        tn, fp, fn, tp = matrix[0] + matrix[1]
        assert abs(tp / (tp + fp + fn) - row['val_iou_sidewalk_global']) < 1e-12
        assert abs((tp / (tp + fp + fn) + tn / (tn + fp + fn)) / 2 - row['val_miou_global']) < 1e-12
        assert abs(row['lr_next'] - (1e-6 + (1e-4 - 1e-6) * (1 + math.cos(math.pi * row['epoch'] / 100)) / 2)) < 1e-14
    assert all(s['step'].item() == completed * 744 for s in cp['optimizer_state']['state'].values())
    assert all(v.item() == completed * 744 for k, v in cp['model_state'].items() if k.endswith('num_batches_tracked'))
    assert all(bool(core.torch.isfinite(v).all()) for v in cp['model_state'].values() if v.is_floating_point())
    assert [r['train_order'] for r in cp['records']] == [r['train_order'] for r in reference['records'][:completed]]
    gen = core.generators(seed)['sample_order']
    for _ in range(completed):
        core.torch.randperm(2975, generator=gen)
    assert core.torch.equal(gen.get_state(), cp['rng']['generators']['sample_order'])
    if completed == 100:
        assert equal_tree(cp['rng'], reference['rng'])


def audit(folder, completed, entry):
    cp = core.torch.load(folder / f'checkpoints/epoch_{completed:03d}.pth', map_location='cpu', weights_only=False)
    ref = core.torch.load(entry['B_checkpoint'], map_location='cpu', weights_only=False)
    assert core.digest(entry['B_checkpoint']) == entry['B_checkpoint_sha256']
    verify_checkpoint(cp, completed, entry['seed'], entry['settings'], ref)
    assert read(folder / 'initialization.json')['backbone_state_sha256'] == entry['initial_backbone_sha256']
    return cp


def integrity(job, plan):
    old_job = Path(plan['C42_job'])
    old.integrity(old_job, read(old_job / 'job_plan.json'))
    assert c.sources() == plan['source_hashes']
    assert core.digest(__file__) == plan['supervisor_sha256']
    assert core.digest(plan['check_script']) == plan['check_script_sha256']
    assert core.digest(plan['check_proof']) == plan['check_proof_sha256']
    assert core.digest(plan['study_plan']) == plan['study_plan_sha256']
    for name, sha in read(job / 'integrity_before.json').items():
        assert core.digest(core.HERE / name) == sha


def prepare(job, source, proof):
    assert not job.exists() and job.is_relative_to(core.HERE)
    old_plan = read(source / 'job_plan.json')
    old.integrity(source, old_plan)
    assert read(source / 'final_verification.json')['status'] == 'pass'
    checks = read(proof)
    assert checks['status'] == 'pass' and checks['source_hashes'] == c.sources()
    assert checks['supervisor_sha256'] == core.digest(__file__)
    assert checks['check_script_sha256'] == core.digest(checks['check_script'])
    assert shutil.disk_usage(core.HERE).free > 15 * 2**30
    rows = read(Path(old_plan['ab_job']) / 'three_seed_results.json')['results']
    entries = []
    for seed in (43, 44):
        b = next(r for r in rows if r['variant'] == 'B' and r['seed'] == seed)
        bc = core.torch.load(b['final_checkpoint'], map_location='cpu', weights_only=False)
        expected = c.settings(seed, 'train')
        for key in ('protocol', 'manifest_sha256', 'runtime', 'epoch_limit', 'mode', 'seed'):
            assert expected[key] == bc['settings'][key]
        cm, bm = build_c(seed), core.build_paired_model('B', seed)
        assert equal_tree(cm.state_dict(), bm.state_dict())
        entries.append({'seed': seed, 'settings': expected, 'B_checkpoint': b['final_checkpoint'],
            'B_checkpoint_sha256': core.digest(b['final_checkpoint']),
            'initial_backbone_sha256': core.state_digest({k: v for k, v in cm.state_dict().items() if not k.startswith('ca.')}),
            'stage1': str(job / f'C_seed{seed}_epoch001'),
            'stage2': str(job / f'C_seed{seed}_epochs002_100')})
    study = core.HERE / 'step_06_2026-10-23/study_review_001/next_experiments.json'
    plan = {'prepared': timestamp(), 'C42_job': str(source), 'ab_job': old_plan['ab_job'],
        'source_hashes': c.sources(), 'supervisor_sha256': core.digest(__file__),
        'check_script': checks['check_script'], 'check_script_sha256': checks['check_script_sha256'],
        'check_proof': str(proof), 'check_proof_sha256': core.digest(proof),
        'study_plan': str(study), 'study_plan_sha256': core.digest(study), 'entries': entries,
        'scope': 'C43 then C44, full epoch1 and independent resume2..100, audited before progression',
        'authorization': 'User2026-10-24 requests autonomous completion of planned later steps; advance only after validation gates',
        'no_automatic_retry': True, 'estimated_hours': [26, 30]}
    job.mkdir(parents=True)
    core.json_write(job / 'integrity_before.json', ab.snapshot(source))
    core.json_write(job / 'job_plan.json', plan)
    integrity(job, plan)
    core.json_write(job / 'preflight_verification.json', {'status': 'pass', 'seeds': [43, 44],
        'B_C_initial_tensors_exact': True, 'protocol_data_runtime_paired': True,
        'frozen_C_sources_unchanged': True, 'seed_aware_audit_regression_pass': True})
    io.atomic_json(job / 'job_status.json', {'status': 'prepared', 'completed_runs': 0, 'total_runs': 2,
        'ab_three_seed_study_complete': True, 'C_three_seed_study_complete': False})
    print(json.dumps({'status': 'prepared', 'job': str(job)}, ensure_ascii=False), flush=True)


def verify_resume(job, entry):
    cp = audit(Path(entry['stage2']), 2, entry)
    path = Path(entry['stage1']) / 'checkpoints/epoch_001.pth'
    first = core.torch.load(path, map_location='cpu', weights_only=False)
    assert equal_tree(cp['records'][:1], first['records']) and cp['settings'] == first['settings']
    provenance = read(Path(entry['stage2']) / 'resume_provenance.json')
    assert provenance['starting_epoch'] == 2 and provenance['sha256'] == core.digest(path)
    core.json_write(job / f'C_seed{entry["seed"]}_resume_verification.json', {'status': 'pass',
        'seed': entry['seed'], 'completed_epoch': 2, 'updates': 1488,
        'fresh_process_resume': True, 'full_data_and_paired_order': True})


def result(job, entry):
    folder = Path(entry['stage2'])
    cp = audit(folder, 100, entry)
    best = max(cp['records'], key=lambda r: r['val_miou_global'])
    assert best['epoch'] == cp['best_epoch']
    weights = folder / f'best_weights/epoch_{best["epoch"]:03d}.pth'
    pure = core.torch.load(weights, map_location='cpu', weights_only=True)
    assert equal_tree(pure, cp['best_model_state'])
    build_c(entry['seed']).load_state_dict(pure, strict=True)
    row = {'variant': 'C', 'seed': entry['seed'], 'epochs': 100, 'best_epoch': best['epoch'],
        'best_global_miou_pct': 100 * best['val_miou_global'],
        'best_sidewalk_iou_pct': 100 * best['val_iou_sidewalk_global'],
        'best_boundary_iou_pct': 100 * best['val_boundary_iou_author_fg_mean'],
        'best_weights': str(weights), 'final_checkpoint': str(folder / 'checkpoints/epoch_100.pth'),
        'final_checkpoint_sha256': core.digest(folder / 'checkpoints/epoch_100.pth')}
    core.json_write(job / f'C_seed{entry["seed"]}_result.json', row)
    return row


def journal(job, phase, seed=None):
    command = [str(ab.DOC_PYTHON), str(core.HERE / 'update_work_record.py'),
        '--mode', 'c-repeats', '--job', str(job), '--phase', phase]
    if seed is not None:
        command += ['--seed', str(seed)]
    return subprocess.run(command, check=False).returncode == 0


def finish(job, plan):
    integrity(job, plan)
    rows = read(Path(plan['ab_job']) / 'three_seed_results.json')['results']
    rows += [r for r in read(Path(plan['C42_job']) / 'c_seed42_results.json')['results'] if r['variant'] == 'C']
    rows += [read(job / f'C_seed{s}_result.json') for s in (43, 44)]
    summary = {v: {f: {'mean': statistics.mean(r[f] for r in rows if r['variant'] == v),
        'sample_std_ddof1': statistics.stdev(r[f] for r in rows if r['variant'] == v)} for f in FIELDS} for v in ('A', 'B', 'C')}
    paired = {}
    for base in ('A', 'B'):
        paired['C_minus_' + base] = {}
        for field in FIELDS:
            deltas = {str(s): next(r[field] for r in rows if r['variant'] == 'C' and r['seed'] == s)
                - next(r[field] for r in rows if r['variant'] == base and r['seed'] == s) for s in (42, 43, 44)}
            paired['C_minus_' + base][field] = {'per_seed': deltas, 'mean': statistics.mean(deltas.values()),
                'sample_std_ddof1': statistics.stdev(deltas.values())}
    core.json_write(job / 'three_seed_ABC_results.json', {'results': rows, 'summary': summary,
        'paired_percentage_points': paired, 'selection': 'same earliest best global validation mIoU checkpoint for all metrics',
        'interpretation': 'three paired seeds under one protocol; not independent-test performance or a significance claim'})
    core.json_write(job / 'final_verification.json', {'status': 'pass', 'C_three_seed_study_complete': True,
        'ab_three_seed_study_complete': True, 'all_9_runs_100epochs': True,
        'paired_orders_rng_and_best_weights_verified': True, 'old_artifacts_unchanged': True})


def train(job):
    plan = read(job / 'job_plan.json')
    assert read(job / 'job_status.json')['status'] == 'prepared'
    integrity(job, plan)
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    pending, completed = [], 0
    began = time.perf_counter()
    for entry in plan['entries']:
        seed = entry['seed']
        for stage in (1, 2):
            folder = Path(entry[f'stage{stage}'])
            if (job / 'STOP_AFTER_EPOCH').exists():
                io.atomic_json(job / 'job_status.json', {'status': 'stopped_at_epoch_boundary', 'seed': seed,
                    'stage': stage, 'completed_runs': completed, 'total_runs': 2})
                return
            command = [sys.executable, '-u', str(core.HERE / 'common_runner_c.py'), '--mode', 'train',
                '--seed', str(seed), '--output', str(folder)]
            command += ['--stop-after-epoch', '1'] if stage == 1 else ['--resume', str(Path(entry['stage1']) / 'checkpoints/epoch_001.pth')]
            with (job / f'C_seed{seed}_stage{stage}.log').open('x', encoding='utf-8') as log:
                process = subprocess.Popen(command, cwd=str(core.PROJECT.parent), env=env, stdout=log, stderr=subprocess.STDOUT)
                checked = False
                try:
                    while process.poll() is None:
                        live, saved = {}, stage - 1
                        try:
                            if (folder / 'training_progress.json').exists():
                                live = read(folder / 'training_progress.json')
                            if (folder / 'status.json').exists():
                                saved = read(folder / 'status.json')['completed_epoch']
                        except (PermissionError, json.JSONDecodeError):
                            pass
                        if stage == 2 and saved >= 2 and not checked:
                            verify_resume(job, entry)
                            checked = True
                            if not journal(job, 'resume', seed):
                                pending.append(f'C{seed}_resume')
                        io.atomic_json(job / 'job_status.json', {'status': 'running', 'variant': 'C', 'seed': seed,
                            'stage': stage, 'variant_folder': str(folder), 'saved_completed_epoch': saved,
                            'pid': os.getpid(), 'child_pid': process.pid, 'target_epochs': 100, 'training': live,
                            'completed_runs': completed, 'total_runs': 2, 'ab_three_seed_study_complete': True,
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
                io.atomic_json(job / 'job_status.json', {'status': 'stopped_at_epoch_boundary', 'variant': 'C',
                    'seed': seed, 'stage': stage, 'variant_folder': str(folder), 'completed_runs': completed, 'total_runs': 2})
                return
            if code:
                raise RuntimeError(f'C{seed} stage{stage} failed {code}; checkpoints preserved, no automatic retry')
            if stage == 1:
                audit(folder, 1, entry)
                core.json_write(job / f'C_seed{seed}_first_epoch_verification.json', {'status': 'pass',
                    'seed': seed, 'epoch': 1, 'updates': 744, 'full_data_and_paired_order': True})
            elif not checked:
                verify_resume(job, entry)
        result(job, entry)
        integrity(job, plan)
        completed += 1
        if not journal(job, 'seed', seed):
            pending.append(f'C{seed}_completion')
    finish(job, plan)
    ok = journal(job, 'complete')
    io.atomic_json(job / 'job_status.json', {'status': 'complete', 'verified': True,
        'completed_runs': 2, 'total_runs': 2, 'ab_three_seed_study_complete': True, 'C_three_seed_study_complete': True,
        'completed': timestamp(), 'work_record_updated': ok, 'work_record_update_pending': pending + ([] if ok else ['summary']),
        'next_step': 'frozen-weight sensitivity evaluation and diagnosis before D/E; user has authorized subsequent validated stages'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('prepare', 'train'), required=True)
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--source-job', type=Path)
    p.add_argument('--proof', type=Path)
    a = p.parse_args()
    job = ab.local(a.job)
    core.torch.set_num_threads(2)
    if a.mode == 'prepare':
        prepare(job, a.source_job.resolve(), a.proof.resolve())
        return
    with Path(str(job) + '.stdout.log').open('x', encoding='utf-8', buffering=1) as out, Path(str(job) + '.stderr.log').open('x', encoding='utf-8', buffering=1) as err:
        sys.stdout, sys.stderr = out, err
        with (core.HERE / 'seed42_training.lock').open('a+b') as lock:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                with fast.keep_awake():
                    train(job)
            except BaseException:
                io.atomic_json(job / 'job_status.json', {'status': 'failed', 'at': timestamp(),
                    'traceback': traceback.format_exc(), 'checkpoints_preserved': True, 'no_automatic_retry': True})
                traceback.print_exc()
                raise
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    main()
