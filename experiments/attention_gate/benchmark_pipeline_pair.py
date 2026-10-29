"""One bounded B42 CPU-pipeline comparison with exact-output verification."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from benchmark_verified_fp32 import power, memory_snapshot

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PROTOCOL=HERE/'pipeline_comparison_protocol.json'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,data): Path(p).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def worker(a):
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    import common_runner as core
    import common_runner_v2 as fast
    import deployment_pipeline as pipe
    import verified_deployment_runtime as runtime_module
    import safe_progress_io as progress
    protocol=read(PROTOCOL)
    proof=read(a.correctness/'verification.json')
    assert proof['status']=='pass' and proof['images']==500 and proof['all_input_pixels_and_tensors_bitwise_equal']
    for p,h in proof['source_hashes'].items(): assert sha(ROOT/p)==h
    assert not a.output.exists() and a.output.is_relative_to(HERE)
    torch.set_num_threads(1); cv2.setNumThreads(1)
    sources={str(p):sha(p) for p in [Path(__file__),PROTOCOL,a.models,a.correctness/'verification.json',
        Path(pipe.__file__),Path(runtime_module.__file__),Path(core.__file__),Path(fast.__file__),HERE/'benchmark_verified_fp32.py']}
    current_power=power(); assert current_power['AC']==1 and current_power['scheme']=='a1841308-3541-4fab-bc81-f71556f20b4a'
    dataset=core.SidewalkDataset(fast.protocol(),'val'); before=dataset.manifest()
    count=3 if a.mode=='smoke' else protocol['measured_images']
    warmup=2 if a.mode=='smoke' else protocol['warmup']
    a.output.mkdir(parents=True)
    write(a.output/'plan.json',dict(protocol=protocol,pipeline=a.pipeline,round=a.round,mode=a.mode,
        count=count,warmup=warmup,source_hashes=sources,power=current_power))
    with runtime_module.VerifiedRuntime(a.models,'cuda',a.output/'provider_evidence') as runtime:
        index=next(i for i,x in enumerate(runtime.models) if x['variant']=='B')
        runtime.session_for(index)
        def frame(i):
            torch.cuda.synchronize(); times=[time.perf_counter()]
            if a.pipeline=='original':
                with Image.open(dataset.pairs[i][0]) as source:
                    rgb=source.convert('RGB'); rgb_array=np.array(rgb)
                    times.append(time.perf_counter())
                    tensor=dataset.transform(rgb).unsqueeze(0).contiguous().numpy()
            else:
                bgr=pipe.read_bgr(dataset.pairs[i][0]); times.append(time.perf_counter())
                tensor=pipe.preprocess_bgr(bgr)
            times.append(time.perf_counter())
            logits=runtime.infer(index,tensor); torch.cuda.synchronize(); times.append(time.perf_counter())
            if a.pipeline=='original':
                low=logits.argmax(1)[0].astype(np.uint8)
                restored=cv2.resize(low,(rgb_array.shape[1],rgb_array.shape[0]),interpolation=cv2.INTER_NEAREST)
                selected=restored==1
                times.append(time.perf_counter())
                overlay=rgb_array.copy(); overlay[selected]=(.6*overlay[selected]+.4*np.array([40,210,40])).astype(np.uint8)
                times.append(time.perf_counter())
            else:
                low=pipe.binary_mask(logits); restored=pipe.restore_mask(low,bgr)
                times.append(time.perf_counter())
                overlay=pipe.overlay_bgr(bgr,restored)
                overlay=cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB); times.append(time.perf_counter())
            # Both timed pipelines produce RGB bytes before verification.
            fingerprints=dict(mask=hashlib.sha256(low.tobytes()).hexdigest(),overlay=hashlib.sha256(overlay.tobytes()).hexdigest())
            return dict(index=i,total_ms=1000*(times[-1]-times[0]),
                stage_ms=dict(zip(['read_decode','preprocess','transfer_inference','mask_restore','overlay'],
                    [1000*(b-a) for a,b in zip(times,times[1:])])),**fingerprints)
        for i in range(warmup): frame(i)
        memory_before=memory_snapshot()
        rows=[]; started=time.perf_counter()
        for i in range(count):
            if i%25==0:
                if (a.output.parent/'STOP_REQUESTED').exists(): raise InterruptedError('User stop marker')
                progress.atomic_json(a.output/'status.json',dict(status='running',images=i,target=count))
            rows.append(frame(i))
        wall=time.perf_counter()-started
        memory_after=memory_snapshot()
        after=power(); assert after['AC']==1 and after['scheme']==current_power['scheme']
        assert dataset.manifest()==before
        for p,h in sources.items(): assert sha(p)==h
        values=[r['total_ms'] for r in rows]
        write(a.output/'samples.json',rows)
        write(a.output/'result.json',dict(status='pass',mode=a.mode,pipeline=a.pipeline,round=a.round,
            images=count,warmup=warmup,mean_ms=float(np.mean(values)),median_ms=float(np.median(values)),
            p95_ms=float(np.percentile(values,95)),actual_loop_fps=count/wall,loop_wall_seconds=wall,
            stage_means_ms={k:float(np.mean([r['stage_ms'][k] for r in rows])) for k in rows[0]['stage_ms']},
            source_hashes=sources,samples_sha256=sha(a.output/'samples.json'),
            power_before=current_power,power_after=after,memory_before=memory_before,memory_after=memory_after,
            execution=runtime.proofs))
        progress.atomic_json(a.output/'status.json',dict(status='complete',images=count,checks_passed=True))


def main(a):
    for key in ('models','output','correctness'): setattr(a,key,getattr(a,key).resolve())
    if a.pipeline:
        worker(a); return
    assert not a.output.exists() and a.output.is_relative_to(HERE)
    a.output.mkdir(parents=True)
    protocol=read(PROTOCOL)
    orders=[['original','optimized']] if a.mode=='smoke' else protocol['round_orders']
    write(a.output/'plan.json',dict(mode=a.mode,protocol=protocol,script_sha256=sha(Path(__file__)),correctness=str(a.correctness)))
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        results=[]
        for number,order in enumerate(orders,1):
            for name in order:
                folder=a.output/f'round{number}_{name}'
                write(a.output/'status.json',dict(status='running',round=number,pipeline=name,completed=len(results)))
                args=[sys.executable,str(Path(__file__).resolve()),'--models',str(a.models),'--correctness',str(a.correctness),
                    '--mode',a.mode,'--output',str(folder),'--pipeline',name,'--round',str(number)]
                with (a.output/f'round{number}_{name}.stdout.log').open('wb') as so,(a.output/f'round{number}_{name}.stderr.log').open('wb') as se:
                    subprocess.run(args,stdout=so,stderr=se,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
                results.append(read(folder/'result.json'))
            first=read(a.output/f'round{number}_original/samples.json')
            second=read(a.output/f'round{number}_optimized/samples.json')
            assert len(first)==len(second)
            assert all(a['index']==b['index'] and a['mask']==b['mask'] and a['overlay']==b['overlay'] for a,b in zip(first,second)), 'Output mismatch; retain evidence and reject candidate'
        write(a.output/'results.json',dict(status='pass',mode=a.mode,results=results,all_masks_and_overlays_exact=True,
            formal_timing=a.mode=='full',candidate_count=1))
        write(a.output/'status.json',dict(status='complete',sessions=len(results),checks_passed=True))
    except Exception:
        write(a.output/'failure.json',dict(status='failed',traceback=traceback.format_exc())); raise
    finally: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,required=True); p.add_argument('--correctness',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--mode',choices=['smoke','full'],required=True)
    p.add_argument('--pipeline',choices=['original','optimized']); p.add_argument('--round',type=int,default=1)
    main(p.parse_args())
