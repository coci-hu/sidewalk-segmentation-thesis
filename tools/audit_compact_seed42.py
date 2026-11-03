"""CPU-only audit of completed L0/L1 seed42 artifacts; no new evaluation.

Keep training-time validation distinct from independent best-weight reevaluation.
This audit never launches training, inference, repeats, or deployment benchmarks.
"""
import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'experiments/attention_gate'
sys.path.insert(0,str(HERE))
import common_runner_compact as runner
from compact_candidate import build_compact


def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def relative(path): return path.relative_to(ROOT).as_posix()


def equal(a,b):
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray): return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main(output):
    assert output.resolve().is_relative_to(ROOT) and not output.exists()
    torch.set_num_threads(1)
    profile_path=ROOT/'results/lightweight_profile_20261101_v2.json'
    profile=read(profile_path)
    assert all(sha(ROOT/p)==h for p,h in profile['source_hashes'].items())
    abc_path=ROOT/'results/abc_summary.json'
    references={r['variant']:r for r in read(abc_path)['results'] if r['seed']==42 and r['variant'] in ('A','B')}
    for row in references.values():
        assert sha(ROOT/row['best_weights'])==row['best_weights_sha256']
    paths={
        'L0':HERE/'step_12_2026-11-01/compact_seed42_001/L0_epochs002_100',
        'L1':HERE/'step_12_2026-11-02/L1_seed42_recovery_001/L1_epochs073_100'}
    cached,results={},{}
    for variant,path in paths.items():
        status=read(path/'status.json')
        assert status['status']=='complete' and status['completed_epoch']==100 and status['dataset_identity_unchanged']
        checkpoint=Path(status['last_checkpoint'])
        c=torch.load(checkpoint,map_location='cpu',weights_only=False)
        settings=read(path/'settings.json')
        runner.validate_resume(c,settings)
        assert settings['variant']==variant and settings['seed']==42 and settings['mode']=='train'
        assert settings['protocol']==runner.fast.protocol() and settings['source_hashes']==runner.sources()
        rows=c['records'];assert len(rows)==100
        for row in rows:
            assert row['train_images']==2975 and row['val_images']==500
            assert row['train_batch_sizes']==[4]*743+[3]
            confusion=np.array(row['val_confusion'],dtype=np.int64)
            assert confusion.sum()==500*768*1024
            scores=runner.core.scores_from_confusion(confusion)
            assert abs(scores['miou']-row['val_miou_global'])<1e-12
            assert abs(scores['iou'][1]-row['val_iou_sidewalk_global'])<1e-12
        best=max(rows,key=lambda r:r['val_miou_global'])
        assert best['epoch']==c['best_epoch']==status['best_epoch']
        best_path=path/f'best_weights/epoch_{best["epoch"]:03d}.pth'
        weights=torch.load(best_path,map_location='cpu',weights_only=True)
        assert equal(weights,c['best_model_state'])
        model=build_compact(variant,42);model.load_state_dict(weights,strict=True)
        assert all(torch.isfinite(t).all() for t in weights.values())
        assert sum(p.numel() for p in model.parameters())==profile['results'][variant]['params']
        csv_rows=list(csv.DictReader((path/'metrics.csv').open(encoding='utf-8-sig')))
        assert len(csv_rows)==100
        for row,actual in zip(csv_rows,rows):
            assert all(float(value)==actual[key] for key,value in row.items())
        for name in ('A','B'):
            baseline_settings=read((ROOT/references[name]['final_checkpoint']).parents[1]/'settings.json')
            assert settings['manifest_sha256']==baseline_settings['manifest_sha256']
            assert settings['protocol']==baseline_settings['protocol']
        initial=read(path/'initialization.json')
        timing=read(path/'epoch_timing.json')[-10:]
        scores=dict(global_miou_pct=100*best['val_miou_global'],sidewalk_iou_pct=100*best['val_iou_sidewalk_global'],boundary_iou_pct=100*best['val_boundary_iou_author_fg_mean'])
        delta={name:{key:scores[key]-base['best_'+key] for key in scores} for name,base in references.items()}
        reduction=profile['results'][variant]['reduction_vs_A']
        passed=(reduction['params']>=20 and reduction['conv_linear_flops']>=20 and
            delta['A']['sidewalk_iou_pct']>=-1 and delta['A']['boundary_iou_pct']>=-1)
        results[variant]=dict(seed=42,epochs=100,best_epoch=best['epoch'],**scores,
            delta_vs_baseline_pp=delta,params=profile['results'][variant]['params'],
            conv_linear_gflops=profile['results'][variant]['gflops_conv_linear'],
            reduction_vs_A_percent=reduction,predeclared_targets_met_in_training_log=passed,
            best_weights=relative(best_path),best_weights_sha256=sha(best_path),
            final_checkpoint=relative(checkpoint),final_checkpoint_sha256=sha(checkpoint),
            source_status=relative(path/'status.json'),status_sha256=sha(path/'status.json'),
            metrics_sha256=sha(path/'metrics.csv'),initial_backbone_sha256=initial['backbone_state_sha256'],
            recent_epoch_seconds=statistics.mean(r['train_seconds']+r['validation_seconds'] for r in timing))
        cached[variant]=c
    assert results['L0']['initial_backbone_sha256']==results['L1']['initial_backbone_sha256']
    assert all(a['train_order']==b['train_order'] for a,b in zip(cached['L0']['records'],cached['L1']['records']))
    assert equal(cached['L0']['rng'],cached['L1']['rng'])
    resumed=read(paths['L1']/'resume_provenance.json')
    old_path=Path(resumed['checkpoint']);assert sha(old_path)==resumed['sha256']
    old=torch.load(old_path,map_location='cpu',weights_only=False)
    assert old['completed_epoch']==72 and resumed['starting_epoch']==73
    assert equal(old['records'],cached['L1']['records'][:72])
    report=dict(status='pass',scope='CPU artifact integrity and training-log screening only',
        training_complete=True,independent_best_weight_reevaluation_complete=False,
        pending_user_confirmation=True,automatic_next_experiments=False,
        results=results,reference_seed42=references,
        checks=['100 epochs and dataset counts','same fixed protocol and manifest as A/B42',
            'CSV and checkpoint metrics exact','global metrics reconstructed from confusion matrices',
            'earliest maximum mIoU selects same stored best weights','full optimizer/scheduler history',
            'paired initialization, 100 sample orders and final RNG equal','L1 resume prefix72 preserved'],
        screening_rule='Predeclared >=20% Params and Conv/Linear FLOPs reduction vs A; sidewalk and boundary IoU loss <=1 percentage point vs paired A42. Working targets, not supervisor-specified thresholds.',
        interpretation='Both pass training-log targets. L1 has lower sidewalk IoU than L0 in this single seed; no stable attention benefit or failure conclusion. No independent test-set claim.',
        source_hashes=runner.sources(),audit_source_sha256=sha(Path(__file__)),
        profile_sha256=sha(profile_path),reference_summary_sha256=sha(abc_path))
    output.write_bytes((json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'))
    print(json.dumps({k:{n:r[n] for n in ('best_epoch','sidewalk_iou_pct','boundary_iou_pct','recent_epoch_seconds','predeclared_targets_met_in_training_log')} for k,r in results.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    main(p.parse_args().output)
