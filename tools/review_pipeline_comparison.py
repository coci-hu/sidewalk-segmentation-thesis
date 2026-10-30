"""Audit the stopped exact-output comparison without turning partial timings into acceptance."""
import argparse
import json
from pathlib import Path
import numpy as np
from export_pipeline_comparison import ROOT,HERE,read,sha,rel,close,provider_evidence

def main(job):
    job=job.resolve(); failure=read(job/'failure.json'); plan=read(job/'plan.json')
    assert failure['status']=='failed' and 'Output mismatch' in failure['traceback']
    assert not (job/'results.json').exists() and not (job/'round3_original').exists()
    assert plan['protocol']==read(HERE/'pipeline_comparison_protocol.json')
    assert sha(HERE/'benchmark_pipeline_pair.py')==plan['script_sha256']
    accepted=next(r for r in read(ROOT/'results/deployment_cuda_fp32_preserved_bn.json')['results'] if r['variant']=='B')
    sources=[job/'failure.json',job/'plan.json']; code={}; samples={}; sessions=[]
    for name in ('round1_original','round1_optimized','round2_optimized','round2_original'):
        folder=job/name; result=read(folder/'result.json'); rows=read(folder/'samples.json')
        assert read(folder/'status.json')==dict(status='complete',images=500,checks_passed=True)
        assert result['status']=='pass' and result['images']==500 and result['warmup']==50
        assert result['mode']=='full' and result['pipeline']==name.split('_')[1]
        assert sha(folder/'samples.json')==result['samples_sha256']
        assert [r['index'] for r in rows]==list(range(500))
        for p,h in result['source_hashes'].items():
            assert sha(p)==h,p
            if 'step_' not in rel(p): code[rel(p)]=h
        values=[r['total_ms'] for r in rows]
        assert all(np.isfinite(v) and v>0 for v in values)
        close(np.mean(values),result['mean_ms']); close(np.median(values),result['median_ms'])
        close(np.percentile(values,95),result['p95_ms']); close(500/result['loop_wall_seconds'],result['actual_loop_fps'])
        for row in rows: close(sum(row['stage_ms'].values()),row['total_ms'])
        for power in (result['power_before'],result['power_after']):
            assert power['AC']==1 and power['scheme']=='a1841308-3541-4fab-bc81-f71556f20b4a'
        proof=result['execution']['1']; identity=proof['model']
        for k in ('selected_epoch','weights_sha256','onnx_sha256'): assert identity[k]==accepted[k]
        assert sha(identity['onnx'])==identity['onnx_sha256'] and sha(identity['weights'])==identity['weights_sha256']
        assert proof['provider_options']['CUDAExecutionProvider']['use_tf32']=='0'
        assert proof['disabled_optimizers']==['ConvBNFusion','ConvActivationFusion']
        profiles=list((folder/'provider_evidence').glob('*.json'))
        assert len(profiles)==1 and sha(profiles[0])==proof['profile_sha256']
        assert provider_evidence(read(profiles[0]),'CUDAExecutionProvider')==proof['provider_evidence']
        sessions.append(dict(session=name,images=500,mean_ms=result['mean_ms'],p95_ms=result['p95_ms'],
            actual_loop_fps=result['actual_loop_fps'],scope='Partial run only; failed overall acceptance'))
        samples[name]=rows; sources.extend([folder/'result.json',folder/'samples.json',folder/'plan.json',profiles[0]])
    mismatch={}
    for first,second in [('round1_original','round1_optimized'),('round2_original','round2_optimized'),
                         ('round1_original','round2_original'),('round1_optimized','round2_optimized')]:
        mismatch[first+'_vs_'+second]=[a['index'] for a,b in zip(samples[first],samples[second]) if a['mask']!=b['mask'] or a['overlay']!=b['overlay']]
    assert mismatch==dict(round1_original_vs_round1_optimized=[],round2_original_vs_round2_optimized=[207],
        round1_original_vs_round2_original=[207],round1_optimized_vs_round2_optimized=[])
    report=dict(status='failed_acceptance',audit_status='pass',candidate_accepted=False,formal_three_round_comparison_complete=False,
        completed_sessions=4,planned_sessions=6,mismatched_image_indices=mismatch,partial_sessions=sessions,
        decision='stop_candidate_branch_use_reference_pre_and_postprocessing',
        inference='The original pipeline also changed across sessions on index207. This does not isolate the optimized implementation as the cause. Raw masks/logits were not stored, so changed-pixel counts cannot be recovered from hashes.',
        no_threshold_or_weights_changed=True,no_third_round_or_speedup_claim=True,
        source_hashes=code,sources={rel(p):sha(p) for p in sources},reviewer_sha256=sha(Path(__file__)))
    out=ROOT/'results/pipeline_comparison_review.json'; assert not out.exists()
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(audit='pass',comparison='failed_acceptance',mismatches=mismatch),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--job',type=Path,required=True); main(p.parse_args().job)
