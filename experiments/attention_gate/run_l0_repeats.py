"""User-approved L0-only queue: full500 check, seeds43/44, selected reevaluations.

Frozen training source and hyperparameters are reused. L1 repeats are forbidden.
No seed is dropped because of poor accuracy; failures stop without auto-retry.
"""
import argparse
import datetime
import json
import msvcrt
import os
from pathlib import Path
import statistics
import subprocess
import sys
import traceback
import common_runner as core
import common_runner_compact as runner
import common_runner_v2 as fast
import safe_progress_io as io


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sources():
    result=runner.sources()
    for path in (Path(__file__),core.HERE/'evaluate_compact_selected.py'):
        result[str(path.relative_to(core.PROJECT))]=core.digest(path)
    return result


def main(output):
    output=output.resolve()
    if output==core.HERE or not output.is_relative_to(core.HERE) or output.exists():raise ValueError('NEW local output directory required')
    authorization=read(core.HERE/'lightweight_revision_plan.json')
    assert authorization['execution_authorized'] and authorization['approved_option']=='L0_only'
    proof=core.HERE/'step_12_2026-11-01/compact_preflight_002/verification.json'
    runner.require_acceptance(proof)
    report_path=core.PROJECT/'results/compact_seed42_audit_20261103.json'
    report=read(report_path)
    assert report['status']=='pass' and report['source_hashes']==runner.sources()
    selected=report['results']['L0']
    assert core.digest(core.PROJECT/selected['best_weights'])==selected['best_weights_sha256']
    source_run=(core.PROJECT/selected['final_checkpoint']).parents[1]
    output.mkdir(parents=True)
    pinned=sources()
    core.json_write(output/'job_plan.json',dict(variant='L0',seeds=[43,44],epochs=100,
        user_choice='只补L0',L1_repeats=False,authorization_snapshot=authorization,
        source_hashes=pinned,seed42_audit_sha256=core.digest(report_path),
        acceptance_sha256=core.digest(proof),poor_accuracy_does_not_cancel_seed44=True,
        reevaluation_tolerance_fraction=1e-6,no_automatic_retries=True))
    state=dict(status='running',phase='seed42_reevaluation',pid=os.getpid(),
        started_at=datetime.datetime.now().isoformat(),stop_marker=str(output/'STOP_AFTER_EPOCH'),
        active_variant='L0',active_seed=42,source_hashes=pinned)
    io.atomic_json(output/'job_status.json',state)
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    def child(name,script,args):
        if (output/'STOP_AFTER_EPOCH').exists():raise fast.StopAtEpochBoundary()
        assert sources()==pinned,'Execution source changed'
        state.update(phase=name,active_output=str(output/name))
        with (output/(name+'.log')).open('x',encoding='utf-8') as log:
            process=subprocess.Popen([sys.executable,'-u',str(core.HERE/script)]+args,
                cwd=core.PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=process.pid;io.atomic_json(output/'job_status.json',state)
            code=process.wait()
        if code==75:raise fast.StopAtEpochBoundary()
        if code:raise RuntimeError(f'{name} failed with exit {code}; retained log; no automatic retry')
    def evaluate(seed,folder):
        name=f'L0_seed{seed}_reevaluation'
        state['active_seed']=seed
        child(name,'evaluate_compact_selected.py',['--run',str(folder),'--output',str(output/name)])
        evidence=read(output/name/'verification.json')
        assert evidence['status']=='pass' and evidence['seed']==seed and evidence['variant']=='L0'
        assert evidence['weights_sha256']==core.digest(evidence['weights'])
        assert evidence['source_hashes'][str((core.HERE/'evaluate_compact_selected.py').relative_to(core.PROJECT))]==pinned[str((core.HERE/'evaluate_compact_selected.py').relative_to(core.PROJECT))]
        return evidence
    try:
        evaluations={42:evaluate(42,source_run)}
        actual=evaluations[42]['actual'];base=report['reference_seed42']['A']
        target=authorization['working_targets_not_supervisor_thresholds']
        reduction=selected['reduction_vs_A_percent']
        passed=(reduction['params']>=target['params_reduction_vs_A_percent_at_least'] and
            reduction['conv_linear_flops']>=target['conv_linear_flops_reduction_vs_A_percent_at_least'] and
            100*actual['val_iou_sidewalk_global']>=base['best_sidewalk_iou_pct']-target['sidewalk_iou_loss_vs_A_pp_at_most'] and
            100*actual['val_boundary_iou_author_fg_mean']>=base['best_boundary_iou_pct']-target['boundary_iou_26_loss_vs_A_pp_at_most'])
        core.json_write(output/'repeat_gate.json',dict(status='pass' if passed else 'failed',variant='L0',seed42_reevaluation_passed=True,
            target=target,actual=actual,repeat_seeds=[43,44],L1_repeats=False))
        if not passed:raise RuntimeError('Original L0 screening targets failed after reevaluation')
        references=read(core.PROJECT/'results/abc_summary.json')['results']
        for seed in (43,44):
            state['active_seed']=seed
            name=f'L0_seed{seed}_epochs001_100';folder=output/name
            child(name,'common_runner_compact.py',['--variant','L0','--seed',str(seed),'--mode','train',
                '--output',str(folder),'--acceptance-proof',str(proof)])
            status=read(folder/'status.json');assert status['status']=='complete' and status['completed_epoch']==100
            checkpoint=core.torch.load(status['last_checkpoint'],map_location='cpu',weights_only=False)
            runner.validate_resume(checkpoint,runner.settings('L0',seed,'train'))
            assert all(r['train_images']==2975 and r['val_images']==500 and r['train_batch_sizes']==[4]*743+[3] for r in checkpoint['records'])
            baseline=next(r for r in references if r['variant']=='A' and r['seed']==seed)
            baseline_file=core.PROJECT/baseline['final_checkpoint']
            assert core.digest(baseline_file)==baseline['final_checkpoint_sha256']
            baseline_state=core.torch.load(baseline_file,map_location='cpu',weights_only=False)
            assert checkpoint['settings']['manifest_sha256']==baseline_state['settings']['manifest_sha256']
            assert checkpoint['settings']['protocol']==baseline_state['settings']['protocol']
            assert [r['train_order'] for r in checkpoint['records']]==[r['train_order'] for r in baseline_state['records']]
            core.json_write(folder/'completion_audit.json',dict(status='pass',seed=seed,
                final_checkpoint_sha256=core.digest(status['last_checkpoint']),paired_A_sample_orders_verified=True,
                all100_epochs_verified=True,source_hashes=runner.sources()))
            del checkpoint,baseline_state
            evaluations[seed]=evaluate(seed,folder)
        metrics=['val_miou_global','val_iou_sidewalk_global','val_boundary_iou_author_fg_mean']
        summary={k:dict(mean_pct=statistics.mean(v['actual'][k]*100 for v in evaluations.values()),
            sample_std_pct=statistics.stdev(v['actual'][k]*100 for v in evaluations.values())) for k in metrics}
        core.json_write(output/'three_seed_summary.json',dict(status='complete',variant='L0',seeds=[42,43,44],
            selected_reevaluations={str(k):v for k,v in evaluations.items()},summary=summary,
            L1_scope='seed42 exploratory only, user chose no repeats',
            interpretation='Same validation set used for model selection; no independent test or significance claim'))
        state.update(status='complete',phase='L0_three_seeds_verified',next_action='Review paired A/B differences; plan selected-model export and timing')
        io.atomic_json(output/'job_status.json',state)
    except fast.StopAtEpochBoundary:
        state.update(status='stopped_by_user');io.atomic_json(output/'job_status.json',state)
    except BaseException:
        state.update(status='failed',error=traceback.format_exc());io.atomic_json(output/'job_status.json',state)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    with (core.HERE/'compact_queue.lock').open('a+b') as lock:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            with fast.keep_awake():main(a.output)
        finally:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
