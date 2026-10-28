"""Reconstruct CUDA FP32 acceptance from all saved per-image evidence."""
import argparse
import datetime
import math
from pathlib import Path
import sys
import numpy as np
from export_repository_summary import ROOT, EXP, digest, dump
from export_followup_summary import read, sanitize
sys.path.insert(0,str(EXP))
from deployment_equivalence import Agreement, provider_evidence


def same(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: same(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b): same(x,y)
    elif isinstance(a,float): assert math.isclose(a,b,rel_tol=0,abs_tol=1e-10)
    else: assert a==b


def main(a):
    folder=a.job.resolve()
    proof=read(folder/'verification.json'); plan=read(folder/'plan.json')
    results=read(folder/'results.json')
    assert proof['mode']==plan['mode']==results['mode']=='full'
    assert proof['backend']==plan['backend']==results['backend']=='cuda'
    assert proof['models']==3 and proof['images_per_model']==500
    assert plan['indices']==list(range(500))
    assert plan['precision']==dict(dtype='float32',pytorch_tf32=False,onnx_cuda_use_tf32=0)
    for path,sha in proof['source_hashes'].items(): assert digest(ROOT/path)==sha
    registry=ROOT/'results/abc_summary.json'
    assert digest(registry)==plan['selected_results_sha256']
    models=a.models.resolve()
    assert digest(models)==plan['models_manifest_sha256']
    manifest=read(models)
    frozen={(r['variant'],r['seed']):r for r in read(registry)['results']}
    reports=[]; source_paths=[folder/'verification.json',folder/'results.json',folder/'plan.json',models,registry,Path(__file__)]
    assert len(results['results'])==3 and {r['variant'] for r in results['results']}==set('ABC')
    for row in results['results']:
        assert row['seed']==42
        expected=frozen[(row['variant'],42)]
        assert row['selected_epoch']==expected['best_epoch'] and row['weights_sha256']==expected['best_weights_sha256']
        artifact=next(r for r in manifest['models'] if r['variant']==row['variant'])
        assert digest(Path(artifact['weights']))==row['weights_sha256']
        assert digest(Path(artifact['onnx']))==row['onnx_sha256']
        path=folder/(row['name']+'_per_image.json'); per_image=read(path)
        assert len(per_image)==500 and [r['image_id'] for r in per_image]==[str(i) for i in range(500)]
        agreement=Agreement(); agreement.records=per_image
        for item in per_image:
            assert item['pixels']==768*1024
            assert 0<=item['argmax_disagreement_pixels']<=item['pixels']
            assert math.isfinite(item['maximum_absolute_logit_error'])
            for backend in ('pytorch','onnx'):
                matrix=np.asarray(item[backend]['confusion'],dtype=np.int64)
                assert matrix.shape==(2,2) and (matrix>=0).all() and matrix.sum()==item['pixels']
                agreement.confusion[backend]+=matrix
                value=item[backend]['boundary_iou']
                if value is not None:
                    assert math.isfinite(value) and 0<=value<=1
                    agreement.boundary[backend].append(value)
        reconstructed=agreement.summary()
        for key,value in reconstructed.items(): same(value,row[key])
        profiles=list(folder.glob(row['name']+'_provider*.json'))
        assert len(profiles)==1 and digest(profiles[0])==row['profile_sha256']
        same(provider_evidence(read(profiles[0]),'CUDAExecutionProvider'),row['provider_evidence'])
        assert row['provider_options']['CUDAExecutionProvider']['use_tf32']=='0'
        current=dict(row)
        current['signed_vs_frozen_selected_metrics_pp']={backend:dict(
            sidewalk_iou=current['metrics'][backend]['sidewalk_iou_pct']-expected['best_sidewalk_iou_pct'],
            boundary_iou=current['metrics'][backend]['boundary_iou_pct']-expected['best_boundary_iou_pct']) for backend in ('pytorch','onnx')}
        reports.append(current)
        source_paths.extend([path,profiles[0]])
    accepted=all(r['status']=='pass' for r in reports)
    assert (proof['status']=='pass')==accepted and proof['full500_metrics_verified']==accepted
    execution=plan.get('execution',dict(export_variant='original_fused'))
    if execution['export_variant']=='preserved_BN':
        assert execution['disabled_optimizers']==['ConvBNFusion','ConvActivationFusion']
        for r in reports:
            assert r['disabled_optimizers']==execution['disabled_optimizers']
            assert r['original_BN_nodes']==r['optimized_BN_nodes']>0
    report=dict(status='pass' if accepted else 'fail',snapshot_date=datetime.date.today().isoformat(),
        results=reports,verification=proof,precision=plan['precision'],versions=plan['versions'],
        execution=execution,
        raw_aggregate_reconstruction_passed=True,actual_cuda_nodes_verified=True,
        full500_acceptance_passed=accepted,FP16_may_start=accepted,latency_measured=False,
        scope='Three fixed seed42 models, same500 validation inputs, PyTorch versus ONNX FP32 on local RTX4060. NoFPS or physical embedded-device claim.',
        sources={str(p.relative_to(ROOT)):digest(p) for p in source_paths})
    output=a.output.resolve()
    assert output.is_relative_to(ROOT/'results') and output.suffix=='.json'
    dump(output,sanitize(report))
    print('Reconstructed CUDA full500 acceptance:',report['status'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job',type=Path,required=True)
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'results/deployment_cuda_fp32.json')
    main(p.parse_args())
