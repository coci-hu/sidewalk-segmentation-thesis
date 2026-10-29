"""Real CLI image/video and occupied-lock checks using existing validation material."""
import argparse
import gc
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import cv2
from deployment_pipeline import read_bgr
import common_runner as core
import common_runner_v2 as fast
from ca_hswish_control import build_c


def main(a):
    out=a.output.resolve(); assert out.is_relative_to(core.HERE) and not out.exists()
    out.mkdir(parents=True)
    dataset=core.SidewalkDataset(fast.protocol(),'val')
    script=core.HERE/'sidewalk_demo.py'
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    sources={str(p.relative_to(core.PROJECT)):core.digest(p) for p in [Path(__file__),script,
        core.HERE/'deployment_pipeline.py',core.HERE/'verified_deployment_runtime.py']}
    def run(name,input_file,output_file,index=1,note='Existing Cityscapes validation image; not independent test'):
        args=[sys.executable,str(script),'--models',str(a.models.resolve()),'--input',str(input_file),
              '--output',str(output_file),'--backend','cuda','--model-index',str(index),'--headless','--source-note',note]
        result=subprocess.run(args,capture_output=True,timeout=240,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
        (out/(name+'.stdout.log')).write_bytes(result.stdout); (out/(name+'.stderr.log')).write_bytes(result.stderr)
        return result
    forbidden=out/'lock_must_not_start.png'
    with (core.HERE/'seed42_training.lock').open('a+b') as lock:
        lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            rejected=run('occupied_lock',dataset.pairs[0][0],forbidden)
            assert rejected.returncode!=0 and b'msvcrt.locking' in rejected.stderr
            assert not forbidden.exists() and not forbidden.with_suffix('.png.evidence').exists()
        finally:
            lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
    reports=[]
    for variant,index,image_id in [('A',0,0),('B',1,250),('C',2,499)]:
        image_out=out/f'{variant}_image_{image_id}.png'
        r=run('image_'+variant,dataset.pairs[image_id][0],image_out,index)
        assert r.returncode==0, r.stderr.decode('utf-8',errors='replace')
        report=json.loads(image_out.with_suffix('.png.json').read_text(encoding='utf-8'))
        assert len(report['frames'])==1 and report['frames'][0]['model']==variant+'_seed42'
        proof=report['execution_verified'][str(index)]
        assert set(proof['provider_evidence']['node_executions'])=={'CUDAExecutionProvider'}
        assert proof['provider_options']['CUDAExecutionProvider']['use_tf32']=='0'
        assert proof['disabled_optimizers']==['ConvBNFusion','ConvActivationFusion']
        frame=read_bgr(image_out); assert frame.shape==(1024,2048,3)
        reports.append(dict(variant=variant,image_id=image_id,output=str(image_out),report_sha256=core.digest(image_out.with_suffix('.png.json'))))
        print('Verified image',variant,flush=True)
    clip=out/'constructed_validation_clip.mp4'
    writer=cv2.VideoWriter(str(clip),cv2.VideoWriter_fourcc(*'mp4v'),8.,(2048,1024)); assert writer.isOpened()
    image_ids=[0,75,150,225,300,375,450,499]
    try:
        for i in image_ids:
            frame=read_bgr(dataset.pairs[i][0])
            for _ in range(4): writer.write(frame)
    finally: writer.release()
    note='Constructed clip from8 existing Cityscapes validation still images,4 repeats each; no natural temporal sequence or new dataset'
    output=out/'B_constructed_video_overlay.mp4'
    r=run('constructed_video',clip,output,1,note); assert r.returncode==0,r.stderr.decode('utf-8',errors='replace')
    report=json.loads(output.with_suffix('.mp4.json').read_text(encoding='utf-8'))
    assert len(report['frames'])==32 and report['source_note']==note
    cap=cv2.VideoCapture(str(output)); decoded=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            assert frame.shape==(1024,2048,3); decoded+=1
    finally: cap.release()
    assert decoded==32
    params={}
    for v in 'ABC':
        model=build_c(42) if v=='C' else core.build_paired_model(v,42)
        params[v]=sum(p.numel() for p in model.parameters()); del model; gc.collect()
    for p,h in sources.items(): assert core.digest(core.PROJECT/p)==h
    core.json_write(out/'verification.json',dict(status='pass',GPU_lock_rejected_before_inference=True,
        image_cases=reports,constructed_video=dict(source=str(clip),output=str(output),frames=32,image_ids=image_ids,
            fps_metadata=8,source_note=note,all_output_frames_decoded=True,report_sha256=core.digest(output.with_suffix('.mp4.json'))),
        model_parameter_counts=params,source_hashes=sources,formal_FPS_claim=False,
        long_duration_stability_checked=False,self_collected_or_independent_dataset=False))
    print('PASS:3 image cases,32 video frames,actualCUDA evidence and occupied lock guard')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--models',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); main(p.parse_args())
