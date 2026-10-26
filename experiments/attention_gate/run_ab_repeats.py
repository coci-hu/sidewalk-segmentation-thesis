"""Finish the preplanned A/B seeds43/44 using the frozen v1_2 trainer.

Only orchestration and verification live here. No new scientific settings,
automatic retries, paid services, or C/D/E training are introduced.
"""
import argparse
import csv
import json
import msvcrt
import os
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import common_runner as core
import common_runner_v2 as fast
from check_step03 import equal_tree
from evaluate_existing import protected_hashes, timestamp
from run_step05_benchmark import previous_hashes
from verify_seed42_training import verify_variant

DOC_PYTHON = Path(r'C:\Users\13699\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe')
JOURNAL = Path(r'C:\Users\13699\Desktop\毕设\工作记录.docx')
FIELDS = ('best_global_miou_pct', 'best_sidewalk_iou_pct', 'best_boundary_iou_pct')
EXPECTED_ORDER = [(43, 'A'), (43, 'B'), (44, 'A'), (44, 'B')]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def local(path):
    path = Path(path).resolve()
    if path == core.HERE or not path.is_relative_to(core.HERE):
        raise ValueError('Expected a subdirectory inside attention_gate')
    return path


def snapshot(folder):
    return {str(p.relative_to(core.HERE)): core.digest(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def assert_integrity(job, plan):
    baseline = read(job / 'integrity_before.json')
    assert protected_hashes() == baseline['original']
    assert previous_hashes() == baseline['previous']
    for group in ('benchmark', 'seed42'):
        assert all(core.digest(core.HERE / name) == sha for name, sha in baseline[group].items())
    assert fast.sources() == plan['source_hashes']
    for name, sha in plan['orchestration_hashes'].items():
        assert core.digest(core.HERE / name) == sha


def prepare(job, seed42):
    job, seed42 = local(job), local(seed42)
    if job.exists():
        raise FileExistsError('Use a NEW repeat job directory')
    if read(seed42 / 'final_verification.json')['status'] != 'pass':
        raise RuntimeError('Seed42 is not verified')
    assert read(seed42 / 'job_status.json')['verified']
    assert fast.sources() == read(seed42 / 'job_plan.json')['source_hashes']
    assert shutil.disk_usage(core.HERE).free > 10 * 2**30
    assert core.torch.cuda.is_available()
    core.torch.set_num_threads(2)
    initializations, expected, first_orders = {}, {}, {}
    old_settings = read(seed42 / 'A_seed42_epochs002_100/settings.json')
    for seed in (43, 44):
        a, b = core.build_paired_model('A', seed), core.build_paired_model('B', seed)
        common = {k: v for k, v in b.state_dict().items() if not k.startswith('ca.')}
        assert equal_tree(a.state_dict(), common)
        digest = core.state_digest(common)
        initializations[str(seed)] = {'shared_backbone_sha256': digest,
            'A_parameters': sum(p.numel() for p in a.parameters()),
            'B_parameters': sum(p.numel() for p in b.parameters()),
            'attention_seed': seed + 10000}
        for variant in ('A', 'B'):
            key = f'{variant}_seed{seed}'
            settings = fast.expected_settings(variant, seed)
            for field in ('protocol', 'source_hashes', 'manifest_sha256', 'runtime', 'epoch_limit', 'mode'):
                assert settings[field] == old_settings[field]
            expected[key] = settings
        first_orders[str(seed)] = core.torch.randperm(2975, generator=core.generators(seed)['sample_order']).tolist()
    old_init = read(seed42 / 'A_seed42_epochs002_100/initialization.json')['backbone_state_sha256']
    assert len({old_init, *(x['shared_backbone_sha256'] for x in initializations.values())}) == 3
    assert first_orders['43'] != first_orders['44']
    for row in read(seed42 / 'seed42_results.json')['results']:
        assert core.digest(row['final_checkpoint']) == row['final_checkpoint_sha256']
    job.mkdir(parents=True)
    baseline = {'original': protected_hashes(), 'previous': previous_hashes(),
        'benchmark': snapshot(core.HERE / 'step_05_2026-10-14/benchmark_001'), 'seed42': snapshot(seed42)}
    core.json_write(job / 'integrity_before.json', baseline)
    times = []
    for variant in ('A', 'B'):
        rows = read(seed42 / f'{variant}_seed42_epochs002_100/epoch_timing.json')[-20:]
        times.append(statistics.mean(r['train_seconds'] + r['validation_seconds'] for r in rows))
    queue = [{'seed': seed, 'variant': variant,
        'folder': str(job / f'{variant}_seed{seed}_epochs001_100')} for seed, variant in EXPECTED_ORDER]
    plan = {'prepared': timestamp(), 'scope': 'A/B seeds43/44, 100 epochs each; stop before C/D/E or deployment',
        'source_hashes': fast.sources(), 'seed42_job': str(seed42), 'queue': queue,
        'initializations': initializations, 'expected_settings': expected,
        'estimated_training_hours': 200 * sum(times) / 3600,
        'estimate_basis': 'last20 complete epochs of each seed42 variant; excludes overhead and interruptions',
        'no_automatic_retry': True, 'stop_file': str(job / 'STOP_AFTER_EPOCH'),
        'orchestration_hashes': {name: core.digest(core.HERE / name) for name in
            ('run_ab_repeats.py', 'verify_seed42_training.py')},
        'journal': str(JOURNAL)}
    core.json_write(job / 'job_plan.json', plan)
    core.json_write(job / 'preflight_verification.json', {'status': 'pass', 'at': timestamp(),
        'fixed_sources_match_seed42': True, 'same_data_manifest_protocol_runtime': True,
        'paired_backbone_initialization_exact': True, 'three_seeds_distinct': True,
        'seed42_final_checkpoint_hashes_match': True, 'expected_first_train_orders': first_orders,
        'D_free_GiB': shutil.disk_usage(core.HERE).free / 2**30})
    fast.atomic_json(job / 'job_status.json', {'status': 'prepared', 'at': timestamp(),
        'training_started': False, 'completed_runs': 0, 'total_runs': 4,
        'estimated_training_hours': plan['estimated_training_hours']})
    print(json.dumps({'status': 'prepared', 'job': str(job),
        'estimated_training_hours': plan['estimated_training_hours']}, ensure_ascii=False), flush=True)


def first_epoch_check(job, row):
    folder, seed, variant = Path(row['folder']), row['seed'], row['variant']
    checkpoint, result = verify_variant(folder, variant, 1, seed=seed)
    generator = core.generators(seed)['sample_order']
    order = core.torch.randperm(2975, generator=generator).tolist()
    assert checkpoint['records'][0]['train_order'] == order
    assert core.torch.equal(checkpoint['rng']['generators']['sample_order'], generator.get_state())
    initial = read(folder / 'initialization.json')
    assert initial['backbone_state_sha256'] == read(job / 'job_plan.json')['initializations'][str(seed)]['shared_backbone_sha256']
    core.json_write(folder / 'first_epoch_verification.json', {'status': 'pass', 'seed': seed,
        'variant': variant, 'at': timestamp(), 'completed_epoch': 1, 'updates': 744,
        'sample_order_and_generator_exact': True, 'shared_backbone_matches_preflight': True,
        'checkpoint_sha256': result['final_checkpoint_sha256']})


def verify_pair(job, seed):
    plan = read(job / 'job_plan.json')
    checkpoints, results = {}, []
    for row in plan['queue']:
        if row['seed'] != seed:
            continue
        variant, folder = row['variant'], Path(row['folder'])
        checkpoint, result = verify_variant(folder, variant, 100, seed=seed)
        assert read(folder / 'initialization.json')['backbone_state_sha256'] == plan['initializations'][str(seed)]['shared_backbone_sha256']
        assert read(folder / 'first_epoch_verification.json')['status'] == 'pass'
        checkpoints[variant] = checkpoint
        results.append(result)
    assert [r['train_order'] for r in checkpoints['A']['records']] == [r['train_order'] for r in checkpoints['B']['records']]
    assert equal_tree(checkpoints['A']['rng'], checkpoints['B']['rng'])
    assert_integrity(job, plan)
    delta = {field: results[1][field] - results[0][field] for field in FIELDS}
    core.json_write(job / f'seed{seed}_results.json', {'results': results,
        'CA_minus_ENet_percentage_points': delta, 'seed': seed})
    core.json_write(job / f'seed{seed}_final_verification.json', {'status': 'pass', 'seed': seed,
        'epochs_per_variant': 100, 'all_sample_orders_and_final_rng_paired': True,
        'each_parameter_optimizer_steps': 74400, 'BN_training_batches': 74400,
        'best_weights_strict_load': True, 'old_artifacts_unchanged': True, 'sources_unchanged': True})
    return results


def finish(job):
    plan = read(job / 'job_plan.json')
    seed42 = read(Path(plan['seed42_job']) / 'seed42_results.json')
    rows = list(seed42['results'])
    deltas = {42: seed42['CA_minus_ENet_percentage_points']}
    for seed in (43, 44):
        assert read(job / f'seed{seed}_final_verification.json')['status'] == 'pass'
        data = read(job / f'seed{seed}_results.json')
        rows.extend(data['results'])
        deltas[seed] = data['CA_minus_ENet_percentage_points']
    summary = {variant: {field: {'mean': statistics.mean(r[field] for r in rows if r['variant'] == variant),
        'sample_std_ddof1': statistics.stdev(r[field] for r in rows if r['variant'] == variant)}
        for field in FIELDS} for variant in ('A', 'B')}
    changes = {field: {'mean': statistics.mean(deltas[s][field] for s in (42, 43, 44)),
        'sample_std_ddof1': statistics.stdev(deltas[s][field] for s in (42, 43, 44)),
        'per_seed': {str(s): deltas[s][field] for s in (42, 43, 44)}} for field in FIELDS}
    assert_integrity(job, plan)
    core.json_write(job / 'three_seed_results.json', {'results': rows, 'summary': summary,
        'paired_CA_minus_ENet_percentage_points': changes,
        'interpretation': 'three paired seeds under this fixed protocol; no claim of universality or significance'})
    with (job / 'three_seed_results.csv').open('x', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    core.json_write(job / 'final_verification.json', {'status': 'pass', 'three_seed_study_complete': True,
        'all_six_runs_100_epochs': True, 'source_and_old_artifact_integrity': True,
        'paired_order_and_rng_each_seed': True, 'mean_and_sample_std_ddof1': True})


def update_journal(job, mode, seed=None):
    command = [str(DOC_PYTHON), str(core.HERE / 'update_work_record.py'), '--mode', mode, '--job', str(job)]
    if seed is not None:
        command.extend(['--seed', str(seed)])
    try:
        result = subprocess.run(command, cwd=str(core.HERE), check=False)
        return result.returncode == 0
    except OSError:
        traceback.print_exc()
        return False


def train(job):
    plan = read(job / 'job_plan.json')
    assert read(job / 'preflight_verification.json')['status'] == 'pass'
    assert read(job / 'job_status.json')['status'] == 'prepared', 'No automatic restart; use a new recovery job'
    assert [(r['seed'], r['variant']) for r in plan['queue']] == EXPECTED_ORDER
    for row in plan['queue']:
        assert not Path(row['folder']).exists()
    assert_integrity(job, plan)
    env = os.environ.copy()
    env['PYTHONIOENCODING'], env['PYTHONUTF8'] = 'utf-8', '1'
    began = time.perf_counter()
    completed = 0
    journal_pending = []
    for row in plan['queue']:
        folder, seed, variant = Path(row['folder']), row['seed'], row['variant']
        if (job / 'STOP_AFTER_EPOCH').exists():
            fast.atomic_json(job / 'job_status.json', {'status': 'stopped_at_epoch_boundary',
                'completed_runs': completed, 'next_run': row})
            return
        assert fast.expected_settings(variant, seed) == plan['expected_settings'][f'{variant}_seed{seed}']
        command = [sys.executable, '-u', str(core.HERE / 'common_runner_v2.py'), '--mode', 'train',
            '--variant', variant, '--seed', str(seed), '--output', str(folder)]
        with (job / f'{variant}_seed{seed}_console.log').open('x', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=str(core.PROJECT.parent), env=env,
                stdout=log, stderr=subprocess.STDOUT)
            first_checked = False
            try:
                while process.poll() is None:
                    progress = folder / 'training_progress.json'
                    status = folder / 'status.json'
                    live = {}
                    try:
                        if progress.exists():
                            live = read(progress)
                        if not first_checked and status.exists() and read(status).get('completed_epoch', 0) >= 1:
                            first_epoch_check(job, row)
                            first_checked = True
                    except (json.JSONDecodeError, PermissionError):
                        pass
                    fast.atomic_json(job / 'job_status.json', {'status': 'running', 'variant': variant, 'seed': seed,
                        'variant_folder': str(folder), 'target_epochs': 100, 'pid': os.getpid(),
                        'child_pid': process.pid, 'completed_runs': completed, 'total_runs': 4,
                        'elapsed_seconds': time.perf_counter() - began, 'training': live,
                        'work_record_update_pending': journal_pending})
                    time.sleep(15)
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=30)
                raise
            code = process.wait()
        if code == 75 and (job / 'STOP_AFTER_EPOCH').exists():
            fast.atomic_json(job / 'job_status.json', {'status': 'stopped_at_epoch_boundary',
                'variant': variant, 'seed': seed, 'variant_folder': str(folder), 'completed_runs': completed})
            return
        if code:
            raise RuntimeError(f'{variant} seed{seed} exited {code}; see console; complete checkpoints preserved')
        if not first_checked:
            first_epoch_check(job, row)
        completed += 1
        if variant == 'B':
            fast.atomic_json(job / 'job_status.json', {'status': 'verifying', 'seed': seed,
                'pid': os.getpid(), 'completed_runs': completed, 'total_runs': 4})
            verify_pair(job, seed)
            if not update_journal(job, 'repeat-pair', seed):
                journal_pending.append(f'seed{seed}')
    finish(job)
    ok = update_journal(job, 'repeat-completion')
    if not ok:
        journal_pending.append('three_seed_completion')
    fast.atomic_json(job / 'job_status.json', {'status': 'complete', 'verified': True,
        'completed': timestamp(), 'completed_runs': completed, 'total_runs': 4,
        'three_seed_study_complete': True, 'wall_seconds': time.perf_counter() - began,
        'work_record_updated': ok, 'work_record_update_pending': journal_pending,
        'report': str(JOURNAL), 'next_step': 'review three paired seeds; C CUDA and resume checks before training'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('prepare', 'train'), required=True)
    parser.add_argument('--job', required=True, type=Path)
    parser.add_argument('--seed42-job', type=Path)
    args = parser.parse_args()
    job = local(args.job)
    if args.mode == 'prepare':
        prepare(job, args.seed42_job)
        return
    with Path(str(job) + '.stdout.log').open('x', encoding='utf-8', buffering=1) as out, \
            Path(str(job) + '.stderr.log').open('x', encoding='utf-8', buffering=1) as err:
        sys.stdout, sys.stderr = out, err
        core.torch.set_num_threads(2)
        with (core.HERE / 'seed42_training.lock').open('a+b') as lock:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                with fast.keep_awake():
                    try:
                        train(job)
                    except BaseException:
                        fast.atomic_json(job / 'job_status.json', {'status': 'failed', 'at': timestamp(),
                            'traceback': traceback.format_exc(), 'completed_checkpoints_preserved': True,
                            'no_automatic_retry': True})
                        traceback.print_exc()
                        raise
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    main()
