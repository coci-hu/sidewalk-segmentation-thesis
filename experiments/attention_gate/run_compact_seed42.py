"""Bounded queue: two full epoch pilots, then resume each to epoch 100.

No architecture search, seed cherry-picking, deletion, automatic failure retry,
or resume from smoke checkpoints. STOP_AFTER_EPOCH is checked between jobs
and by the training loop; every child owns the shared GPU lock.
"""
import argparse
import datetime
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import common_runner as core
import common_runner_compact as runner
import common_runner_v2 as fast
import safe_progress_io as io


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main(output, proof, max_pair_hours):
    output,proof=output.resolve(),proof.resolve()
    assert output.is_relative_to(core.HERE) and output!=core.HERE and not output.exists()
    runner.require_acceptance(proof)
    output.mkdir(parents=True)
    pinned=runner.sources()
    status=dict(status='running',phase='pilots',pid=os.getpid(),started_at=datetime.datetime.now().isoformat(),
        source_hashes=pinned,acceptance_proof=str(proof),acceptance_sha256=core.digest(proof),
        seed=42,variants=['L0','L1'],epochs=100,max_pair_hours=max_pair_hours,
        stop_marker=str(output/'STOP_AFTER_EPOCH'),automatic_failure_retry=False)
    io.atomic_json(output/'job_status.json',status)
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')

    def launch(variant,phase,extra):
        if (output/'STOP_AFTER_EPOCH').exists():
            raise fast.StopAtEpochBoundary()
        assert runner.sources()==pinned,'Training source changed'
        child=output/f'{variant}_{phase}'
        status.update(active_variant=variant,phase=phase,active_output=str(child))
        io.atomic_json(output/'job_status.json',status)
        command=[sys.executable,'-u',str(core.HERE/'common_runner_compact.py'),
            '--variant',variant,'--seed','42','--mode','train','--output',str(child),
            '--acceptance-proof',str(proof)]+extra
        with (output/f'{variant}_{phase}.log').open('x',encoding='utf-8') as stream:
            process=subprocess.Popen(command,cwd=core.PROJECT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            status['child_pid']=process.pid
            io.atomic_json(output/'job_status.json',status)
            code=process.wait()
        if code==75:
            raise fast.StopAtEpochBoundary()
        if code:
            raise RuntimeError(f'{variant} {phase} exited {code}; inspect its retained log')
        return child

    try:
        pilots={}
        for variant in ('L0','L1'):
            child=launch(variant,'epoch001',['--stop-after-epoch','1'])
            checkpoint=core.torch.load(child/'checkpoints/epoch_001.pth',map_location='cpu',weights_only=False)
            runner.validate_resume(checkpoint,runner.settings(variant,42,'train'))
            row=checkpoint['records'][0]
            assert row['train_images']==2975 and row['val_images']==500
            assert row['train_batch_sizes']==[4]*743+[3]
            assert row['lr_used']==fast.protocol()['formal_training']['learning_rate']
            timing=read(child/'epoch_timing.json')[0]
            seconds=timing['train_seconds']+timing['validation_seconds']
            assert seconds>0
            pilots[variant]=dict(epoch_seconds=seconds,train_order=row['train_order'],
                checkpoint=str(child/'checkpoints/epoch_001.pth'),checkpoint_sha256=core.digest(child/'checkpoints/epoch_001.pth'))
        assert pilots['L0']['train_order']==pilots['L1']['train_order']
        pair_hours=sum(x['epoch_seconds'] for x in pilots.values())*100/3600
        # Allow 25% overhead above measured epoch time, with a finite 48h ceiling.
        estimate=dict(pair_hours=pair_hours,pair_hours_with_25percent_margin=pair_hours*1.25,
            six_runs_hours_with_25percent_margin=pair_hours*3*1.25,
            convention='100 epochs per variant, each epoch includes full 2975 train and 500 val',
            pilots={k:{n:v for n,v in x.items() if n!='train_order'} for k,x in pilots.items()})
        core.json_write(output/'pilot_verification.json',dict(status='pass',**estimate))
        status.update(pilot_verification=str(output/'pilot_verification.json'),estimated_pair_hours=pair_hours)
        if pair_hours*1.25>max_pair_hours:
            status.update(status='needs_time_budget_review',phase='pilots_complete')
            io.atomic_json(output/'job_status.json',status)
            return
        completed={}
        for variant in ('L0','L1'):
            child=launch(variant,'epochs002_100',['--resume',pilots[variant]['checkpoint']])
            state=read(child/'status.json')
            assert state['completed_epoch']==100 and state['status']=='complete'
            checkpoint=core.torch.load(state['last_checkpoint'],map_location='cpu',weights_only=False)
            runner.validate_resume(checkpoint,runner.settings(variant,42,'train'))
            assert all(r['train_images']==2975 and r['val_images']==500 for r in checkpoint['records'])
            best=max(checkpoint['records'],key=lambda r:r['val_miou_global'])
            assert best['epoch']==checkpoint['best_epoch']
            completed[variant]=dict(best_epoch=best['epoch'],
                best_weights=str(child/f'best_weights/epoch_{best["epoch"]:03d}.pth'),
                selected_metrics={k:v for k,v in best.items() if k.startswith('val_')},
                last_checkpoint_sha256=core.digest(state['last_checkpoint']))
        core.json_write(output/'seed42_training_summary.json',dict(status='complete',variants=completed,
            interpretation='Training-loop validation metrics; independent best-weight reevaluation and screening review still required. No automatic seed43/44 launch.'))
        status.update(status='complete_pending_evaluation_review',phase='seed42_pair_complete',completed=completed,
            next_action='Reevaluate selected weights, apply predeclared accuracy targets, then repeats43/44 only for passing candidates')
        io.atomic_json(output/'job_status.json',status)
    except fast.StopAtEpochBoundary:
        status.update(status='stopped_at_epoch_boundary')
        io.atomic_json(output/'job_status.json',status)
    except BaseException:
        status.update(status='failed',error=traceback.format_exc())
        io.atomic_json(output/'job_status.json',status)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--acceptance-proof',required=True,type=Path)
    p.add_argument('--max-pair-hours',type=float,default=48)
    p.add_argument('--wait-for-proof-minutes',type=float,default=0)
    a=p.parse_args()
    with (core.HERE/'compact_queue.lock').open('a+b') as lock:
        lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            with fast.keep_awake():
                # This small launcher can outlive the interactive check. Missing or
                # failed proof can never unlock training; wait is bounded, no retry.
                deadline=time.monotonic()+60*a.wait_for_proof_minutes
                wait_status=a.output.parent/(a.output.name+'_waiting.json')
                stop_pending=a.output.parent/(a.output.name+'_STOP_REQUESTED')
                while not a.acceptance_proof.is_file():
                    if stop_pending.exists():
                        io.atomic_json(wait_status,dict(status='stopped_before_training',pid=os.getpid()))
                        raise SystemExit(75)
                    if time.monotonic()>=deadline:
                        io.atomic_json(wait_status,dict(status='acceptance_timeout_no_training',pid=os.getpid()))
                        raise RuntimeError('CUDA acceptance not available before bounded deadline')
                    io.atomic_json(wait_status,dict(status='waiting_for_acceptance',pid=os.getpid(),
                        acceptance_proof=str(a.acceptance_proof.resolve()),stop_marker=str(stop_pending.resolve()),
                        minutes_remaining=(deadline-time.monotonic())/60))
                    time.sleep(10)
                runner.require_acceptance(a.acceptance_proof)
                if stop_pending.exists(): raise SystemExit(75)
                io.atomic_json(wait_status,dict(status='acceptance_passed_starting_pilots',pid=os.getpid()))
                main(a.output,a.acceptance_proof,a.max_pair_hours)
        finally:
            lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
