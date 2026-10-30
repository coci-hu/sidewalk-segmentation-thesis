"""Bounded repeat-input stability observation using the conservative deployment path."""
import argparse
import ctypes
from ctypes import wintypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import cv2
import numpy as np
import torch
from PIL import Image
import common_runner as core
import common_runner_v2 as fast
import reference_deployment_pipeline as pipeline
from verified_deployment_runtime import VerifiedRuntime
from benchmark_verified_fp32 import memory_snapshot,power
from safe_progress_io import atomic_json

PROTOCOL=core.HERE/'stability_protocol.json'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fingerprint(x): return hashlib.sha256(x.tobytes()).hexdigest()
def process_memory():
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(x,ctypes.c_size_t) for x in
            ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
             'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]
    counters=Counters(); counters.cb=ctypes.sizeof(counters)
    f=ctypes.windll.psapi.GetProcessMemoryInfo
    f.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]; f.restype=wintypes.BOOL
    assert f(wintypes.HANDLE(-1),ctypes.byref(counters),counters.cb)
    return dict(working_set_bytes=counters.WorkingSetSize,private_bytes=counters.PrivateUsage,
        process_peak_working_set_bytes=counters.PeakWorkingSetSize)

def main(a):
    out=a.output.resolve(); assert out.is_relative_to(core.HERE) and not out.exists()
    protocol=read(PROTOCOL); ids=protocol['image_indices']; duration=5 if a.mode=='smoke' else protocol['duration_seconds']
    demo=read(a.demo/'verification.json')
    assert demo['status']=='pass' and demo['pipeline']==protocol['pipeline']
    for p,h in demo['source_hashes'].items(): assert sha(core.PROJECT/p)==h
    dataset=core.SidewalkDataset(fast.protocol(),'val'); torch.set_num_threads(1); cv2.setNumThreads(1)
    sources={str(p.resolve()):sha(p) for p in (Path(__file__),PROTOCOL,Path(pipeline.__file__),
        core.HERE/'verified_deployment_runtime.py',Path(core.__file__),Path(fast.__file__),a.models,a.demo/'verification.json')}
    inputs={str(dataset.pairs[i][0]):sha(dataset.pairs[i][0]) for i in ids}
    for i in ids:
        path=dataset.pairs[i][0]
        with Image.open(path) as source: expected=dataset.transform(source.convert('RGB')).unsqueeze(0).contiguous().numpy()
        assert np.array_equal(expected,pipeline.preprocess_bgr(pipeline.read_bgr(path)))
    out.mkdir(parents=True)
    core.json_write(out/'plan.json',dict(mode=a.mode,protocol=protocol,duration_seconds=duration,source_hashes=sources,
        input_sha256=inputs,pid=os.getpid(),created_at=datetime.datetime.now().isoformat(),input_tensors_verified=len(ids)))
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        with VerifiedRuntime(a.models,'cuda',out/'provider_evidence') as runtime:
            runtime.session_for(1)
            def frame(i):
                pixels=pipeline.read_bgr(dataset.pairs[i][0]); tensor=pipeline.preprocess_bgr(pixels)
                logits=runtime.infer(1,tensor); mask=pipeline.binary_mask(logits)
                overlay=pipeline.overlay_bgr(pixels,pipeline.restore_mask(mask,pixels))
                assert overlay.shape==pixels.shape and overlay.dtype==np.uint8
                return mask,overlay
            warm=2 if a.mode=='smoke' else protocol['warmup_frames']
            for j in range(warm): frame(ids[j%len(ids)])
            memory_gpu_before=memory_snapshot(); power_before=power()
            baseline={}; memory=[dict(elapsed_seconds=0,**process_memory())]; changed_frames=0; changed_pixels=0; max_changed=0
            frames=0; started=time.perf_counter(); next_memory=60.; max_interval=0.
            atomic_json(out/'status.json',dict(status='running',elapsed_seconds=0,target_seconds=duration,frames=0))
            with (out/'frames.jsonl').open('x',encoding='utf-8') as stream:
                while time.perf_counter()-started<duration or len(baseline)<len(ids):
                    if (out/'STOP_REQUESTED').exists(): raise InterruptedError('User stop marker')
                    i=ids[frames%len(ids)]; before=time.perf_counter(); mask,overlay=frame(i)
                    frame_seconds=time.perf_counter()-before; elapsed=time.perf_counter()-started
                    if i not in baseline:
                        baseline[i]=mask.copy(); np.save(out/f'baseline_mask_{i}.npy',mask,allow_pickle=False)
                    changed=int(np.count_nonzero(mask!=baseline[i])); changed_frames+=int(changed>0)
                    changed_pixels+=changed; max_changed=max(max_changed,changed); max_interval=max(max_interval,frame_seconds)
                    row=dict(frame=frames,image_index=i,elapsed_seconds=elapsed,processing_seconds=frame_seconds,
                        changed_pixels=changed,mask_sha256=fingerprint(mask),overlay_sha256=fingerprint(overlay))
                    stream.write(json.dumps(row,separators=(',',':'))+'\n'); frames+=1
                    if elapsed>=next_memory:
                        memory.append(dict(elapsed_seconds=elapsed,**process_memory())); next_memory+=60
                    if frames%25==0:
                        stream.flush(); atomic_json(out/'status.json',dict(status='running',elapsed_seconds=elapsed,
                            target_seconds=duration,frames=frames,changed_frames=changed_frames))
            elapsed=time.perf_counter()-started; memory.append(dict(elapsed_seconds=elapsed,**process_memory()))
            assert elapsed>=duration and set(baseline)==set(ids)
            assert a.mode=='smoke' or frames>=100
            for p,h in sources.items(): assert sha(p)==h
            for p,h in inputs.items(): assert sha(p)==h
            result=dict(status='pass',mode=a.mode,full30_minutes=a.mode=='full',duration_seconds=elapsed,frames=frames,
                warmup_frames=warm,visited_image_indices=ids,exact_repeat_masks=changed_frames==0,
                changed_frames=changed_frames,total_changed_pixels=changed_pixels,maximum_changed_pixels_per_frame=max_changed,
                maximum_processing_interval_seconds=max_interval,process_memory=memory,GPU_memory_before=memory_gpu_before,
                GPU_memory_after=memory_snapshot(),power_before=power_before,power_after=power(),execution=runtime.proofs,
                frames_sha256=sha(out/'frames.jsonl'),source_hashes=sources,input_sha256=inputs,formal_FPS_claim=False,
                natural_video_or_external_dataset=False,claim='Run completion and observed repeat consistency; not universal determinism or leak-free proof')
            core.json_write(out/'result.json',result)
            atomic_json(out/'status.json',dict(status='complete',elapsed_seconds=elapsed,frames=frames,checks_passed=True,exact_repeat_masks=changed_frames==0))
            print(json.dumps(dict(status='pass',mode=a.mode,seconds=elapsed,frames=frames,changed_frames=changed_frames)),flush=True)
    except Exception:
        core.json_write(out/'failure.json',dict(status='failed_or_stopped',traceback=traceback.format_exc()))
        atomic_json(out/'status.json',dict(status='failed_or_stopped')); raise
    finally: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--models',type=Path,required=True)
    p.add_argument('--demo',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['smoke','full'],required=True); main(p.parse_args())
