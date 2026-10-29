"""Bounded stage profiling of the existing B42 PNG pipeline, before optimization."""
import argparse
import json
import msvcrt
from pathlib import Path
import time
import numpy as np
import torch
import cv2
from PIL import Image
import onnxruntime as ort
import common_runner as core
import common_runner_v2 as fast
from deployment_equivalence import provider_evidence


def main(a):
    assert a.output.resolve().is_relative_to(core.HERE) and not a.output.exists()
    manifest=json.loads(a.models.read_text(encoding='utf-8'))
    item=next(r for r in manifest['models'] if r['variant']=='B')
    accepted=json.loads((core.PROJECT/'results/deployment_cuda_fp32_preserved_bn.json').read_text(encoding='utf-8'))
    original=next(r for r in accepted['results'] if r['variant']=='B')
    assert accepted['full500_acceptance_passed'] and core.digest(Path(item['onnx']))==original['onnx_sha256']
    torch.set_num_threads(1); cv2.setNumThreads(1)
    with (core.HERE/'seed42_training.lock').open('a+b') as lock:
        lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            a.output.mkdir(parents=True)
            dataset=core.SidewalkDataset(fast.protocol(),'val')
            indices=list(range(0,500,25))
            core.json_write(a.output/'plan.json',dict(variant='B',seed=42,indices=indices,warmup=[0,1,2],
                purpose='Stage diagnosis only; not final comparative timing or accuracy evidence',
                graph_sha256=original['onnx_sha256'],source_sha256=core.digest(Path(__file__))))
            opts=ort.SessionOptions(); opts.intra_op_num_threads=1; opts.inter_op_num_threads=1
            opts.add_session_config_entry('session.disable_cpu_ep_fallback','1')
            opts.enable_profiling=True; opts.profile_file_prefix=str(a.output/'provider')
            session=ort.InferenceSession(item['onnx'],sess_options=opts,providers=[('CUDAExecutionProvider',{'use_tf32':0})],
                disabled_optimizers=manifest['execution']['disabled_optimizers'])
            session.disable_fallback()
            def run(i):
                torch.cuda.synchronize(); t=[time.perf_counter()]
                with Image.open(dataset.pairs[i][0]) as source:
                    rgb=source.convert('RGB'); frame=np.array(rgb)
                    t.append(time.perf_counter())
                    tensor=dataset.transform(rgb).unsqueeze(0).contiguous().numpy()
                t.append(time.perf_counter())
                logits=session.run(['logits'],{'image':tensor})[0]; torch.cuda.synchronize()
                t.append(time.perf_counter())
                mask=logits.argmax(1)[0].astype(np.uint8)
                mask=cv2.resize(mask,(frame.shape[1],frame.shape[0]),interpolation=cv2.INTER_NEAREST)
                selected=mask==1
                t.append(time.perf_counter())
                overlay=frame.copy(); overlay[selected]=(.6*overlay[selected]+.4*np.array([40,210,40])).astype(np.uint8)
                t.append(time.perf_counter())
                assert np.isfinite(logits).all()
                return dict(index=i,stage_ms=dict(zip(['read_decode','resize_normalize','transfer_inference','argmax_restore','overlay'],
                    [1000*(b-a) for a,b in zip(t,t[1:])])),total_ms=1000*(t[-1]-t[0]))
            for i in [0,1,2]: run(i)
            profile=Path(session.end_profiling())
            provider=provider_evidence(json.loads(profile.read_text(encoding='utf-8')),'CUDAExecutionProvider')
            rows=[run(i) for i in indices]
            means={key:float(np.mean([r['stage_ms'][key] for r in rows])) for key in rows[0]['stage_ms']}
            result=dict(status='pass',results=rows,stage_means_ms=means,provider=provider,
                profile_sha256=core.digest(profile),scope='20 fixed PNGs, warm-cache diagnosis; no formal FPS claim')
            core.json_write(a.output/'results.json',result)
            print(json.dumps(means,indent=2))
        finally:
            lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
