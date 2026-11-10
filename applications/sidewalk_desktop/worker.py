"""Local desktop worker using the frozen FP32 runtime and reference pipeline."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'experiments/attention_gate'
sys.path.insert(0,str(EXP))
from model_catalog import default_models,resolve_models
DEFAULT_MODELS=default_models()
IMAGE_EXT={'.png','.jpg','.jpeg','.bmp','.webp'}
VIDEO_EXT={'.mp4','.avi','.mov','.mkv','.m4v'}

def write_status(folder,**value):
    temp=folder/'status.pending.json'
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temp,folder/'status.json')


def run(a):
    folder=a.output.resolve();folder.mkdir(parents=True,exist_ok=True)
    # Reserve a fresh job. Refuse reusing an old output, including interrupted jobs.
    with (folder/'request.json').open('x',encoding='utf-8') as f:
        json.dump(dict(input=str(a.input.resolve()),model=a.model,backend='cuda'),f,ensure_ascii=False)
    writer=capture=None
    frames=0;durations=[]
    try:
        if not a.input.is_file():raise ValueError('输入文件不存在。')
        suffix=a.input.suffix.lower()
        if suffix not in IMAGE_EXT|VIDEO_EXT:raise ValueError('不支持该文件格式，请选择常见图片或视频。')
        write_status(folder,state='loading',message='正在加载模型，首次运行可能需要几十秒',frames=0)
        import cv2
        import numpy as np
        from verified_compact_runtime import VerifiedCompactRuntime
        from reference_deployment_pipeline import read_bgr,preprocess_bgr,binary_mask,restore_mask,overlay_bgr
        cv2.setNumThreads(1)
        image_mode=suffix in IMAGE_EXT
        resolved=resolve_models(a.models,folder)
        manifest=json.loads(resolved.read_text(encoding='utf-8'))
        index=next(i for i,m in enumerate(manifest['models']) if m['variant']==a.model)
        if not image_mode:
            capture=cv2.VideoCapture(str(a.input))
            if not capture.isOpened():raise ValueError('无法读取视频，请检查文件或编码格式。')
            total=max(0,int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            fps=capture.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(fps) or fps<=0:fps=25.
        else:total=1
        began=time.perf_counter()
        with VerifiedCompactRuntime(resolved,'cuda',folder/'runtime_evidence') as runtime:
            while True:
                if (folder/'CANCEL_REQUESTED').exists():
                    write_status(folder,state='cancelled',message='已取消；未完成输出保留在本次目录',frames=frames,total=total)
                    return
                tick=time.perf_counter()
                if image_mode:
                    frame=read_bgr(a.input);ok=frame is not None
                else:ok,frame=capture.read()
                if not ok:
                    if not frames:raise ValueError('文件中没有可读取的图像。')
                    break
                tensor=preprocess_bgr(frame)
                infer_start=time.perf_counter();logits=runtime.infer(index,tensor)
                infer_ms=(time.perf_counter()-infer_start)*1000
                low=binary_mask(logits);mask=restore_mask(low,frame);overlay=overlay_bgr(frame,mask)
                durations.append((time.perf_counter()-tick)*1000);frames+=1
                if image_mode:
                    for name,value in [('overlay.png',overlay),('mask.png',mask*255)]:
                        ok,encoded=cv2.imencode('.png',value)
                        if not ok:raise RuntimeError('图片保存失败。')
                        encoded.tofile(folder/name)
                else:
                    if writer is None:
                        writer=cv2.VideoWriter(str(folder/'overlay.partial.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(frame.shape[1],frame.shape[0]))
                        if not writer.isOpened():raise RuntimeError('视频输出编码器不可用。')
                    writer.write(overlay)
                # Save bounded previews; UI reads only after the atomic status notification.
                if frames==1 or frames%10==0 or image_mode:
                    for name,value in [('input_preview.jpg',frame),('output_preview.jpg',overlay),('mask_preview.jpg',cv2.cvtColor(mask*255,cv2.COLOR_GRAY2BGR))]:
                        h,w=value.shape[:2];value=cv2.resize(value,(min(840,w),max(1,round(h*min(840,w)/w))))
                        ok,encoded=cv2.imencode('.jpg',value)
                        if not ok:raise RuntimeError('预览生成失败。')
                        tmp=folder/(name+'.pending');encoded.tofile(tmp);os.replace(tmp,folder/name)
                write_status(folder,state='running',message='正在处理',frames=frames,total=total,
                    inference_ms=infer_ms,frame_processing_ms=durations[-1],foreground_percent=float((mask==1).mean()*100),
                    preview_version=frames if frames==1 or frames%10==0 or image_mode else max(1,frames//10*10))
                if image_mode:break
            proofs=runtime.proofs
        if writer:writer.release();writer=None
        if not image_mode:os.replace(folder/'overlay.partial.mp4',folder/'overlay.mp4')
        result=dict(state='complete',message='处理完成',model=a.model,input=str(a.input.resolve()),frames=frames,
            total=total,elapsed_seconds=time.perf_counter()-began,output=str(folder/('overlay.png' if image_mode else 'overlay.mp4')),
            mask=str(folder/'mask.png') if image_mode else None,
            average_processing_ms=float(np.mean(durations)),execution_verified=proofs,
            speed_is_not_controlled_benchmark=True,first_frame_includes_initialization=True,
            semantic_sidewalk_only=True,source_note=a.source_note)
        (folder/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        write_status(folder,**result)
    except Exception as error:
        (folder/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        message=str(error)
        if isinstance(error,OSError) and ('locking' in traceback.format_exc() or getattr(error,'winerror',None)==33):
            message='显卡实验锁正在被其他任务使用，请等该任务结束后重试。'
        write_status(folder,state='error',message=message,frames=frames)
        raise
    finally:
        if writer:writer.release()
        if capture:capture.release()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',choices=['A','L0'],default='L0')
    p.add_argument('--models',type=Path,default=DEFAULT_MODELS)
    p.add_argument('--source-note',default='User-selected local file; not an independent accuracy experiment')
    run(p.parse_args())
