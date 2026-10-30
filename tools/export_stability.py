"""Reconstruct recorded repeat-input observations; smoke output cannot become a30min claim."""
import argparse
import json
from pathlib import Path
import numpy as np
from export_pipeline_comparison import ROOT,HERE,read,sha,rel,provider_evidence
from export_followup_summary import sanitize

def main(a):
    job=a.job.resolve(); r=read(job/'result.json'); plan=read(job/'plan.json')
    assert r['status']=='pass' and (r['mode']=='full' or a.allow_smoke)
    protocol=read(HERE/'stability_protocol.json'); assert plan['protocol']==protocol
    assert r['full30_minutes']==(r['mode']=='full') and r['duration_seconds']>=plan['duration_seconds']
    assert not r['formal_FPS_claim'] and not r['natural_video_or_external_dataset']
    for p,h in r['source_hashes'].items(): assert sha(p)==h,p
    for p,h in r['input_sha256'].items(): assert sha(p)==h,p
    assert sha(job/'frames.jsonl')==r['frames_sha256']
    ids=protocol['image_indices']; baseline={}
    for i in ids:
        m=np.load(job/f'baseline_mask_{i}.npy',allow_pickle=False)
        assert m.shape==(768,1024) and np.isin(m,[0,1]).all()
        import hashlib
        baseline[i]=hashlib.sha256(m.tobytes()).hexdigest()
    with (job/'frames.jsonl').open(encoding='utf-8') as stream: rows=[json.loads(line) for line in stream]
    assert len(rows)==r['frames'] and [x['frame'] for x in rows]==list(range(len(rows)))
    assert all(x['image_index']==ids[i%len(ids)] for i,x in enumerate(rows))
    assert all(0<=x['changed_pixels']<=768*1024 and (x['changed_pixels']==0)==(x['mask_sha256']==baseline[x['image_index']]) for x in rows)
    assert all(a['elapsed_seconds']<b['elapsed_seconds'] for a,b in zip(rows,rows[1:]))
    # The loop checks its deadline after writing the final frame record.
    assert 0<rows[-1]['elapsed_seconds']<=r['duration_seconds']
    assert r['duration_seconds']-rows[-1]['elapsed_seconds']<1
    assert sum(x['changed_pixels']>0 for x in rows)==r['changed_frames']
    assert sum(x['changed_pixels'] for x in rows)==r['total_changed_pixels']
    assert max(x['changed_pixels'] for x in rows)==r['maximum_changed_pixels_per_frame']
    assert r['exact_repeat_masks']==(r['changed_frames']==0)
    proof=r['execution']['1']; profiles=list((job/'provider_evidence').glob('*.json'))
    assert len(profiles)==1 and sha(profiles[0])==proof['profile_sha256']
    assert provider_evidence(read(profiles[0]),'CUDAExecutionProvider')==proof['provider_evidence']
    assert proof['provider_options']['CUDAExecutionProvider']['use_tf32']=='0'
    if r['mode']=='full': assert r['duration_seconds']>=1800 and len(rows)>=100
    memory=r['process_memory']; assert len(memory)>=2
    summary={k:dict(start=memory[0][k],end=memory[-1][k],maximum_observed=max(x[k] for x in memory),
        end_minus_start=memory[-1][k]-memory[0][k]) for k in ('working_set_bytes','private_bytes')}
    output=dict(status='pass',mode=r['mode'],full30_minutes=r['full30_minutes'],duration_seconds=r['duration_seconds'],
        frames=r['frames'],image_indices=ids,exact_repeat_masks=r['exact_repeat_masks'],changed_frames=r['changed_frames'],
        maximum_changed_pixels_per_frame=r['maximum_changed_pixels_per_frame'],process_memory=summary,
        GPU_memory_before=r['GPU_memory_before'],GPU_memory_after=r['GPU_memory_after'],execution=proof,
        source_hashes={rel(p):h for p,h in r['source_hashes'].items() if 'step_' not in rel(p)},
        sources={rel(p):sha(p) for p in [job/'result.json',job/'plan.json',job/'frames.jsonl',profiles[0]]},
        exporter_sha256=sha(Path(__file__)),raw_frame_records_reconstructed=True,
        formal_FPS_claim=False,limitations=['Repeated eight existing still images; no natural temporal accuracy',
          'Processing loop only; excludes GUI and long output-video writing',
          'Memory snapshots and repeat observations do not prove leak-free or universally deterministic execution'])
    assert not a.output.exists()
    a.output.write_text(json.dumps(sanitize(output),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='pass',mode=r['mode'],seconds=r['duration_seconds'],frames=len(rows),changed_frames=r['changed_frames'])))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--job',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--allow-smoke',action='store_true'); main(p.parse_args())
