"""Audit immutable completed checkpoints while a separate training process continues."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
import datetime
import statistics
from pathlib import Path
import common_runner as core
import run_mixed_pair as supervisor


def main(a):
    assert a.output.resolve().is_relative_to(core.HERE) and not a.output.exists()
    core.torch.set_num_threads(1)
    plan=supervisor.read(a.job/'job_plan.json')
    supervisor.integrity(a.job,plan)
    progress={}
    for entry in plan['entries']:
        folder=Path(entry['remaining'])
        if not (folder/'status.json').exists(): folder=Path(entry['first'])
        state=supervisor.read(folder/'status.json')
        epoch=state['completed_epoch']
        checkpoint=folder/f'checkpoints/epoch_{epoch:03d}.pth'
        before=core.digest(checkpoint)
        cp=supervisor.audit(folder,epoch,entry,plan)
        assert core.digest(checkpoint)==before
        times=supervisor.read(folder/'epoch_timing.json')
        recent=[r['train_seconds']+r['validation_seconds'] for r in times[-10:]]
        progress[entry['variant']]=dict(completed_epochs=epoch,completed_updates=epoch*744,
            checkpoint=str(checkpoint),checkpoint_sha256=before,all_completed_history_verified=True,
            recent_mean_epoch_seconds=statistics.mean(recent),timed_recent_epochs=len(recent))
    remaining_hours=sum((100-r['completed_epochs'])*r['recent_mean_epoch_seconds']/3600 for r in progress.values())
    a.output.mkdir(parents=True)
    core.json_write(a.output/'verification.json',dict(status='pass',variants=progress,
        completed_epochs=sum(r['completed_epochs'] for r in progress.values()),target_epochs=200,
        snapshot_at=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
        remaining_hours_rough=remaining_hours,estimate_basis='Each variant latest up to10 complete epochs; current partial epoch and launch overhead not deducted',
        training_interrupted=False,training_parameters_changed=False,GPU_used=False,
        sources_and_reviewed_evidence_unchanged=True,job_plan_sha256=core.digest(a.job/'job_plan.json'),
        script_sha256=core.digest(__file__),scope='Interim checkpoint integrity, not final model comparison'))
    print({v:r['completed_epochs'] for v,r in progress.items()},'remaining hours~',round(remaining_hours,1))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
