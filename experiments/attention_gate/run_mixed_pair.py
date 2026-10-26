"""Audited D42/E42 exploration: complete first epochs, fresh resume, then 100 epochs.

No automatic retry or promotion. Frozen candidate/scientific sources are read only.
"""
import argparse
import json
import math
import msvcrt
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import common_runner as core
import common_runner_mixed as mixed
import common_runner_v2 as fast
import safe_progress_io as io
from check_step03 import equal_tree
from evaluate_existing import protected_hashes, timestamp
from mixed_ca_candidate import build_candidate


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_checkpoint(cp, expected, reference):
    mixed.validate_resume(cp, expected)
    epoch = cp['completed_epoch']
    train_n, val_n = (7, 3) if expected['mode'] == 'smoke' else (2975, 500)
    batches = [4] * (train_n // 4) + ([train_n % 4] if train_n % 4 else [])
    assert 1 <= epoch <= expected['epoch_limit']
    assert expected['seed'] == reference['settings']['seed']
    for row in cp['records']:
        assert row['train_images'] == train_n and row['val_images'] == val_n
        assert row['train_batch_sizes'] == batches and sorted(row['train_order']) == list(range(train_n))
        matrix = row['val_confusion']
        assert sum(map(sum, matrix)) == val_n * 768 * 1024
        tn, fp, fn, tp = matrix[0] + matrix[1]
        fg = tp / (tp+fp+fn) if tp+fp+fn else 0
        bg = tn / (tn+fp+fn) if tn+fp+fn else 0
        assert abs(fg-row['val_iou_sidewalk_global']) < 1e-12
        assert abs((fg+bg)/2-row['val_miou_global']) < 1e-12
        assert abs(row['lr_next']-(1e-6+(1e-4-1e-6)*(1+math.cos(math.pi*row['epoch']/100))/2)) < 1e-14
        assert 0 < row['alpha_before'] < 1 and 0 < row['alpha_after'] < 1
        if expected['variant'] == 'D':
            assert row['alpha_before'] == row['alpha_after'] == cp['records'][0]['alpha_before']
    assert all(int(s['step']) == epoch*len(batches) for s in cp['optimizer_state']['state'].values())
    assert all(int(v) == epoch*len(batches) for k,v in cp['model_state'].items() if k.endswith('num_batches_tracked'))
    assert all(bool(core.torch.isfinite(v).all()) for v in cp['model_state'].values() if v.is_floating_point())
    assert [r['train_order'] for r in cp['records']] == [r['train_order'] for r in reference['records'][:epoch]]
    generator = core.generators(expected['seed'])['sample_order']
    for _ in range(epoch):
        core.torch.randperm(train_n, generator=generator)
    assert core.torch.equal(generator.get_state(), cp['rng']['generators']['sample_order'])
    if epoch == reference['completed_epoch']:
        assert equal_tree(cp['rng'], reference['rng'])
    best = max(cp['records'], key=lambda r:r['val_miou_global'])
    assert cp['best_epoch'] == best['epoch'] and cp['best_miou'] == best['val_miou_global']
    checked_model = build_candidate(expected['variant'], expected['seed'])
    checked_model.load_state_dict(cp['model_state'], strict=True)
    checked_model.load_state_dict(cp['best_model_state'], strict=True)


def integrity(job, plan, original=False):
    mixed.require_acceptance(plan['acceptance_proof'])
    assert mixed.sources() == plan['source_hashes']
    assert core.digest(__file__) == plan['supervisor_sha256']
    for path, sha in plan['evidence_hashes'].items():
        assert core.digest(path) == sha, path
    if original:
        assert protected_hashes() == read(job / 'original_artifacts_before.json')


def prepare(a):
    job = a.job.resolve()
    assert job.is_relative_to(core.HERE) and job != core.HERE and not job.exists()
    assert shutil.disk_usage(job.parent).free > 12 * 2**30
    mixed.require_acceptance(a.acceptance_proof)
    proof = read(a.supervisor_proof)
    assert proof['status'] == 'pass' and proof['supervisor_sha256'] == core.digest(__file__)
    assert proof['check_script_sha256'] == core.digest(proof['check_script'])
    assert proof['source_hashes'] == mixed.sources()
    source = a.source_job.resolve()
    assert read(source / 'final_verification.json')['status'] == 'pass'
    rows = read(source / 'three_seed_ABC_results.json')['results']
    reference = next(r for r in rows if r['variant'] == 'C' and r['seed'] == 42)
    cp = core.torch.load(reference['final_checkpoint'], map_location='cpu', weights_only=False)
    entries = []
    for variant in ('D','E'):
        settings = mixed.settings(variant, 42, 'train')
        for key in ('protocol', 'manifest_sha256', 'runtime', 'epoch_limit', 'mode', 'seed'):
            assert settings[key] == cp['settings'][key], key
        model = build_candidate(variant, 42)
        entries.append(dict(variant=variant, seed=42, settings=settings,
            initial_backbone_sha256=core.state_digest({k:v for k,v in model.state_dict().items() if not k.startswith('ca.')}),
            first=str(job / f'{variant}_seed42_epoch001'), remaining=str(job / f'{variant}_seed42_epochs002_100')))
    evidence = [a.acceptance_proof.resolve(), a.supervisor_proof.resolve(), Path(proof['check_script']),
        HERE_STUDY, source / 'final_verification.json', source / 'three_seed_ABC_results.json']
    evidence += [Path(r[k]) for r in rows for k in ('best_weights','final_checkpoint')]
    for path, expected_sha in read(a.acceptance_proof)['evidence_hashes'].items():
        assert core.digest(path) == expected_sha, 'Reviewed evidence changed: ' + path
        evidence.append(Path(path))
    original = protected_hashes()
    job.mkdir(parents=True)
    core.json_write(job / 'original_artifacts_before.json', original)
    plan = dict(prepared=timestamp(), source_hashes=mixed.sources(), supervisor_sha256=core.digest(__file__),
        evidence_hashes={str(p):core.digest(p) for p in evidence}, entries=entries,
        acceptance_proof=str(a.acceptance_proof.resolve()), reference=reference, source_job=str(source),
        execution_order=['D epoch1','E epoch1','D epochs2-100','E epochs2-100'],
        budget='Exactly two complete seed42 runs; no accuracy-based early stop or alpha search',
        total_runs=2, no_automatic_retry=True, automatic_promotion=False)
    core.json_write(job / 'job_plan.json', plan)
    integrity(job, plan)
    io.atomic_json(job / 'job_status.json', dict(status='prepared', completed_runs=0, total_runs=2))
    print('Paired exploration prepared and pinned.', flush=True)


def audit(folder, epoch, entry, plan):
    cp = core.torch.load(folder / f'checkpoints/epoch_{epoch:03d}.pth', map_location='cpu', weights_only=False)
    reference = core.torch.load(plan['reference']['final_checkpoint'], map_location='cpu', weights_only=False)
    verify_checkpoint(cp, entry['settings'], reference)
    assert cp['completed_epoch'] == epoch
    assert read(folder / 'initialization.json')['backbone_state_sha256'] == entry['initial_backbone_sha256']
    if epoch >= 2:
        first = Path(entry['first']) / 'checkpoints/epoch_001.pth'
        initial = core.torch.load(first, map_location='cpu', weights_only=False)
        assert equal_tree(cp['records'][:1], initial['records'])
        assert read(folder / 'resume_provenance.json')['sha256'] == core.digest(first)
    return cp


def results(job, plan):
    rows = read(Path(plan['source_job']) / 'three_seed_ABC_results.json')['results']
    rows = [r for r in rows if r['seed'] == 42]
    for e in plan['entries']:
        folder = Path(e['remaining'])
        cp = audit(folder,100,e,plan)
        best = cp['records'][cp['best_epoch']-1]
        weights = folder / f'best_weights/epoch_{cp["best_epoch"]:03d}.pth'
        assert equal_tree(core.torch.load(weights, map_location='cpu', weights_only=True), cp['best_model_state'])
        rows.append(dict(variant=e['variant'], seed=42, epochs=100, best_epoch=cp['best_epoch'],
            best_global_miou_pct=100*best['val_miou_global'], best_sidewalk_iou_pct=100*best['val_iou_sidewalk_global'],
            best_boundary_iou_pct=100*best['val_boundary_iou_author_fg_mean'],
            alpha_at_selected_epoch=best['alpha_after'], final_alpha=cp['records'][-1]['alpha_after'],
            best_weights=str(weights), final_checkpoint=str(folder / 'checkpoints/epoch_100.pth'),
            best_weights_sha256=core.digest(weights),
            final_checkpoint_sha256=core.digest(folder / 'checkpoints/epoch_100.pth')))
    baseline = next(r for r in rows if r['variant']=='C')
    passing = [r['variant'] for r in rows if r['variant'] in ('D','E') and
        r['best_sidewalk_iou_pct'] >= baseline['best_sidewalk_iou_pct'] and r['best_boundary_iou_pct'] >= baseline['best_boundary_iou_pct'] and
        (r['best_sidewalk_iou_pct'] > baseline['best_sidewalk_iou_pct'] or r['best_boundary_iou_pct'] > baseline['best_boundary_iou_pct'])]
    core.json_write(job / 'seed42_ABCDE_results.json', dict(results=rows, promotion_screen_passed=bool(passing),
        candidates_passing_screen=passing, exploratory_seed_only=True, automatic_repeat_training_started=False,
        decision_rule='If any candidate passes, review then repeat BOTH D/E seeds43/44. Compare A/B/C as well.'))


def train(job):
    plan = read(job / 'job_plan.json')
    assert read(job / 'job_status.json')['status'] == 'prepared'
    integrity(job,plan,original=True)
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    began, completed = time.perf_counter(), 0
    for stage in ('first','remaining'):
        for entry in plan['entries']:
            if (job / 'STOP_AFTER_EPOCH').exists():
                io.atomic_json(job / 'job_status.json', dict(status='stopped_at_epoch_boundary',completed_runs=completed))
                return
            integrity(job,plan)
            folder, variant = Path(entry[stage]), entry['variant']
            cmd = [sys.executable,'-u',str(core.HERE / 'common_runner_mixed.py'),'--variant',variant,
                '--seed','42','--mode','train','--output',str(folder),'--acceptance-proof',plan['acceptance_proof']]
            cmd += ['--stop-after-epoch','1'] if stage=='first' else ['--resume',str(Path(entry['first']) / 'checkpoints/epoch_001.pth')]
            with (job / f'{variant}_{stage}.log').open('x',encoding='utf-8') as log:
                process = subprocess.Popen(cmd,cwd=core.PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
                checked = False
                try:
                    while process.poll() is None:
                        saved,live = {},{}
                        try:
                            if (folder/'status.json').exists(): saved=read(folder/'status.json')
                            if (folder/'training_progress.json').exists(): live=read(folder/'training_progress.json')
                        except (PermissionError,json.JSONDecodeError): pass
                        if stage=='remaining' and saved.get('completed_epoch',0)>=2 and not checked:
                            audit(folder,2,entry,plan)
                            core.json_write(job/f'{variant}_resume_verification.json',dict(status='pass',epoch=2,updates=1488,
                                complete_first_epoch_history_preserved=True,paired_C_order=True))
                            checked=True
                        io.atomic_json(job/'job_status.json',dict(status='running',variant=variant,seed=42,stage=stage,
                            variant_folder=str(folder),saved_completed_epoch=saved.get('completed_epoch',0),training=live,
                            pid=os.getpid(),child_pid=process.pid,completed_runs=completed,total_runs=2,
                            elapsed_seconds=time.perf_counter()-began))
                        time.sleep(15)
                except BaseException:
                    if process.poll() is None:
                        # Stop politely at the next fully saved epoch; never discard a partial checkpoint.
                        (job/'STOP_AFTER_EPOCH').touch(exist_ok=True)
                    raise
                code=process.wait()
            if code==75 and (job/'STOP_AFTER_EPOCH').exists():
                io.atomic_json(job/'job_status.json',dict(status='stopped_at_epoch_boundary',variant=variant,
                    variant_folder=str(folder),completed_runs=completed,total_runs=2))
                return
            assert code==0, f'{variant} {stage} failed; original outputs preserved, no retry'
            if stage=='first':
                audit(folder,1,entry,plan)
                core.json_write(job/f'{variant}_first_epoch_verification.json',dict(status='pass',updates=744,paired_C_order=True))
            else:
                if not checked:
                    audit(folder,2,entry,plan)
                    core.json_write(job/f'{variant}_resume_verification.json',dict(status='pass',epoch=2,updates=1488))
                audit(folder,100,entry,plan)
                completed+=1
    integrity(job,plan,original=True)
    results(job,plan)
    core.json_write(job/'final_verification.json',dict(status='pass',completed_runs=2,all_100_epochs=True,
        source_hashes=mixed.sources(),historical_artifacts_unchanged=True,paired_order_and_final_rng_verified=True))
    io.atomic_json(job/'job_status.json',dict(status='complete',verified=True,completed_runs=2,total_runs=2,
        next_step='Review exploratory results and predefined repeat rule; no automatic promotion'))


HERE_STUDY = core.HERE / 'step_06_2026-10-23/study_review_001/next_experiments.json'
if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('prepare','train'),required=True)
    p.add_argument('--job',type=Path,required=True)
    p.add_argument('--source-job',type=Path)
    p.add_argument('--acceptance-proof',type=Path)
    p.add_argument('--supervisor-proof',type=Path)
    a=p.parse_args(); a.job=a.job.resolve()
    assert a.job.is_relative_to(core.HERE) and a.job!=core.HERE
    core.torch.set_num_threads(2)
    if a.mode=='prepare': prepare(a)
    else:
        with (a.job/'supervisor.lock').open('a+b') as lock:
            lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
            try:
                with fast.keep_awake(): train(a.job)
            except BaseException:
                io.atomic_json(a.job/'job_status.json',dict(status='failed',traceback=traceback.format_exc(),
                    checkpoints_preserved=True,no_automatic_retry=True))
                raise
            finally:
                lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
