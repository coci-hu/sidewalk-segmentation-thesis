"""Independently reconstruct paired pipeline samples and verify identical predictions."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'experiments/attention_gate'
sys.path.insert(0,str(HERE))
from deployment_equivalence import provider_evidence

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b): assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8),(a,b)
def rel(p): return Path(p).resolve().relative_to(ROOT).as_posix()
def main(a):
    job=a.job.resolve(); report=read(job/'results.json'); plan=read(job/'plan.json')
    mode=report['mode']; assert mode=='full' or a.allow_smoke
    assert report['status']=='pass' and report['all_masks_and_overlays_exact']
    protocol=read(HERE/'pipeline_comparison_protocol.json'); assert plan['protocol']==protocol
    assert sha(HERE/'benchmark_pipeline_pair.py')==plan['script_sha256']
    count=500 if mode=='full' else 3; warmup=50 if mode=='full' else 2
    orders=protocol['round_orders'] if mode=='full' else [['original','optimized']]
    assert len(report['results'])==2*len(orders)
    accepted=read(ROOT/'results/deployment_cuda_fp32_preserved_bn.json')
    model=next(x for x in accepted['results'] if x['variant']=='B')
    sources=[job/'results.json',job/'plan.json']; pooled={'original':[],'optimized':[]}; sessions=[]; code={}
    for number,order in enumerate(orders,1):
        paired={}
        for name in order:
            folder=job/f'round{number}_{name}'
            result=read(folder/'result.json'); samples=read(folder/'samples.json'); worker_plan=read(folder/'plan.json')
            assert result==report['results'][len(sessions)]
            assert result['status']=='pass' and result['images']==count and result['warmup']==warmup
            assert result['mode']==mode and result['pipeline']==name and result['round']==number
            assert worker_plan['protocol']==protocol and worker_plan['source_hashes']==result['source_hashes']
            for p,h in result['source_hashes'].items():
                assert sha(p)==h,p
                if Path(p).suffix in ('.py','.json') and 'step_' not in rel(p): code[rel(p)]=h
            assert sha(folder/'samples.json')==result['samples_sha256']
            assert [x['index'] for x in samples]==list(range(count))
            values=[x['total_ms'] for x in samples]
            assert all(math.isfinite(x) and x>0 for x in values)
            for sample in samples:
                assert all(math.isfinite(x) and x>=0 for x in sample['stage_ms'].values())
                close(sum(sample['stage_ms'].values()),sample['total_ms'])
            close(np.mean(values),result['mean_ms']); close(np.median(values),result['median_ms'])
            close(np.percentile(values,95),result['p95_ms'])
            close(count/result['loop_wall_seconds'],result['actual_loop_fps'])
            assert result['loop_wall_seconds']>=sum(values)/1000
            for stage in result['stage_means_ms']:
                close(np.mean([x['stage_ms'][stage] for x in samples]),result['stage_means_ms'][stage])
            for power in (result['power_before'],result['power_after']):
                assert power['AC']==1 and power['scheme']=='a1841308-3541-4fab-bc81-f71556f20b4a'
            execution=result['execution']['1']; identity=execution['model']
            assert identity['variant']=='B' and identity['seed']==42 and identity['selected_epoch']==84
            assert identity['onnx_sha256']==model['onnx_sha256'] and identity['weights_sha256']==model['weights_sha256']
            assert execution['provider_options']['CUDAExecutionProvider']['use_tf32']=='0'
            assert execution['disabled_optimizers']==['ConvBNFusion','ConvActivationFusion']
            profiles=list((folder/'provider_evidence').glob('*.json'))
            assert len(profiles)==1 and sha(profiles[0])==execution['profile_sha256']
            assert provider_evidence(read(profiles[0]),'CUDAExecutionProvider')==execution['provider_evidence']
            paired[name]=samples; pooled[name].extend(values)
            sessions.append({k:result[k] for k in ('pipeline','round','images','warmup','mean_ms','median_ms','p95_ms','loop_wall_seconds','actual_loop_fps','stage_means_ms')})
            sources.extend([folder/'result.json',folder/'samples.json',folder/'plan.json',profiles[0]])
        assert all(x['mask']==y['mask'] and x['overlay']==y['overlay'] for x,y in zip(paired['original'],paired['optimized']))
    summary={}
    for name,values in pooled.items():
        rows=[x for x in sessions if x['pipeline']==name]
        summary[name]=dict(images=len(values),mean_ms=float(np.mean(values)),median_ms=float(np.median(values)),
            p95_ms=float(np.percentile(values,95)),actual_loop_fps=len(values)/sum(x['loop_wall_seconds'] for x in rows),
            session_mean_ms_sd=statistics.stdev(x['mean_ms'] for x in rows) if len(rows)>1 else None)
    old,new=summary['original'],summary['optimized']
    output=dict(status='pass',mode=mode,formal_timing=mode=='full',sessions=sessions,summary=summary,
        all_paired_masks_and_overlays_bitwise_equal=True,raw_statistics_and_provider_reconstructed=True,
        mean_latency_reduction_percent=100*(1-new['mean_ms']/old['mean_ms']),
        actual_loop_throughput_ratio=new['actual_loop_fps']/old['actual_loop_fps'],
        source_hashes=code,exporter_sha256=sha(Path(__file__)),sources={rel(p):sha(p) for p in sources},
        claim='One bundled implementation comparison on B42; no architecture gain, new accuracy or camera/video real-time claim')
    assert not a.output.exists(),'Use a new report path; preserve earlier evidence'
    a.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='pass',mode=mode,sessions=len(sessions),images_per_pipeline=len(pooled['original'])),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--job',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--allow-smoke',action='store_true')
    main(p.parse_args())
