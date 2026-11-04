"""Fresh-process full500 reevaluation of a completed L0 run's selected weight.

Same validation set and frozen scientific protocol, not an independent test set.
Metric tolerance is declared before inference and is never relaxed on failure.
"""
import argparse
import hashlib
import json
import msvcrt
from pathlib import Path
import time
import traceback
import common_runner_compact as runner
import common_runner as core
import common_runner_v2 as fast
import safe_progress_io as io
from compact_candidate import build_compact


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def run(folder,output):
    folder,output=folder.resolve(),output.resolve()
    if not folder.is_relative_to(core.HERE) or not output.is_relative_to(core.HERE) or output==core.HERE or output.exists():
        raise ValueError('Local source run and NEW output directory required')
    status=read(folder/'status.json'); settings=read(folder/'settings.json')
    assert status['status']=='complete' and status['completed_epoch']==100
    assert settings['variant']=='L0' and settings['seed'] in (42,43,44)
    checkpoint=core.torch.load(status['last_checkpoint'],map_location='cpu',weights_only=False)
    runner.validate_resume(checkpoint,runner.settings('L0',settings['seed'],'train'))
    expected=max(checkpoint['records'],key=lambda row:row['val_miou_global'])
    assert expected['epoch']==checkpoint['best_epoch']
    weights=folder/f'best_weights/epoch_{expected["epoch"]:03d}.pth'
    state=core.torch.load(weights,map_location='cpu',weights_only=True)
    assert all(core.torch.equal(v,checkpoint['best_model_state'][k]) for k,v in state.items())
    protocol=fast.protocol()
    train,val,manifest=core.prepare('train',protocol)
    assert settings['manifest_sha256']==hashlib.sha256(json.dumps(manifest,sort_keys=True).encode('utf-8')).hexdigest()
    sources=runner.sources();sources[str(Path(__file__).relative_to(core.PROJECT))]=core.digest(__file__)
    metrics=['val_loss','val_miou_global','val_iou_sidewalk_global','val_boundary_iou_author_fg_mean','val_pixel_accuracy_global']
    output.mkdir(parents=True)
    plan=dict(variant='L0',seed=settings['seed'],source_run=str(folder),best_epoch=expected['epoch'],
        weights=str(weights),weights_sha256=core.digest(weights),source_hashes=sources,
        absolute_tolerance_fraction=1e-6,loss_absolute_tolerance=1e-6,
        note='1e-6 metric fraction = 0.0001 percentage points; confusion exactness reported separately; no threshold relaxation',
        selection='earliest maximum global mIoU, no reselection',validation_images=500,
        manifest_sha256=settings['manifest_sha256'],scientific_protocol=protocol)
    core.json_write(output/'plan.json',plan)
    core.json_write(output/'sample_manifest.json',manifest)
    started=time.perf_counter()
    try:
        core.configure(settings['seed'])
        model=build_compact('L0',settings['seed']).cuda()
        model.load_state_dict(state,strict=True)
        before=core.state_digest(model.state_dict())
        _,loader,_=core.make_loaders(train,val,protocol,core.generators(settings['seed']))
        class Progress:
            def __len__(self):return len(loader)
            def __iter__(self):
                for count,batch in enumerate(loader,1):
                    if (output.parent/'STOP_AFTER_EPOCH').exists():raise fast.StopAtEpochBoundary()
                    yield batch
                    if count==1 or count%50==0:
                        io.atomic_json(output/'training_progress.json',dict(status='evaluating',completed_images=count,total_images=500))
                        print(f'L0 seed{settings["seed"]} reevaluation {count}/500',flush=True)
        actual=fast.BASE_VALIDATE(model,Progress(),protocol,core.torch.device('cuda:0'))
        delta={k:actual[k]-expected[k] for k in metrics}
        passed=all(abs(v)<=1e-6 for v in delta.values())
        assert actual['val_images']==500 and actual['boundary_valid_images']==expected['boundary_valid_images']
        assert core.state_digest(model.state_dict())==before and core.digest(weights)==plan['weights_sha256']
        assert runner.sources()=={k:v for k,v in sources.items() if k!=str(Path(__file__).relative_to(core.PROJECT))}
        _,_,after=core.prepare('train',protocol);assert after==manifest
        record=dict(status='pass' if passed else 'failed',**plan,actual=actual,
            expected={k:expected[k] for k in metrics},metric_differences=delta,
            confusion_exact=actual['val_confusion']==expected['val_confusion'],model_unchanged=True,
            dataset_unchanged=True,wall_seconds=time.perf_counter()-started)
        core.json_write(output/'verification.json',record)
        io.atomic_json(output/'status.json',dict(status='complete' if passed else 'failed',passed=passed))
        if not passed:raise RuntimeError('Reevaluation differs beyond predeclared tolerance; no repeats authorized by this proof')
        print('PASS fresh-process selected-weight full500 reevaluation',flush=True)
    except fast.StopAtEpochBoundary:
        io.atomic_json(output/'status.json',dict(status='stopped_by_user',passed=False))
        raise
    except BaseException:
        io.atomic_json(output/'status.json',dict(status='failed',passed=False,error=traceback.format_exc()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    with (core.HERE/'seed42_training.lock').open('a+b') as lock:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            with fast.keep_awake():run(a.run,a.output)
        except fast.StopAtEpochBoundary:raise SystemExit(75)
        finally:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
