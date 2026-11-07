"""Exercise real desktop controls and worker image/video/cancellation/lock paths."""
import argparse
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time
import tkinter as tk
from PIL import Image,ImageGrab

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
EXP=ROOT/'experiments/attention_gate'
sys.path.insert(0,str(HERE));from app import App

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def capture_window(window,path):
    """Use window-owned GDI rendering when the session cannot capture the screen."""
    import ctypes
    from ctypes import wintypes
    user=ctypes.WinDLL('user32',use_last_error=True);gdi=ctypes.WinDLL('gdi32',use_last_error=True)
    user.GetDC.restype=wintypes.HDC;user.GetDC.argtypes=[wintypes.HWND]
    user.ReleaseDC.argtypes=[wintypes.HWND,wintypes.HDC]
    gdi.CreateCompatibleDC.restype=wintypes.HDC;gdi.CreateCompatibleDC.argtypes=[wintypes.HDC]
    gdi.CreateCompatibleBitmap.restype=wintypes.HBITMAP;gdi.CreateCompatibleBitmap.argtypes=[wintypes.HDC,ctypes.c_int,ctypes.c_int]
    gdi.SelectObject.restype=wintypes.HGDIOBJ;gdi.SelectObject.argtypes=[wintypes.HDC,wintypes.HGDIOBJ]
    gdi.DeleteObject.argtypes=[wintypes.HGDIOBJ];gdi.DeleteDC.argtypes=[wintypes.HDC]
    user.PrintWindow.argtypes=[wintypes.HWND,wintypes.HDC,wintypes.UINT]
    gdi.GetDIBits.argtypes=[wintypes.HDC,wintypes.HBITMAP,wintypes.UINT,wintypes.UINT,ctypes.c_void_p,ctypes.c_void_p,wintypes.UINT]
    hwnd=window.winfo_id();w=window.winfo_width();h=window.winfo_height()
    dc=user.GetDC(hwnd);memory=gdi.CreateCompatibleDC(dc);bitmap=gdi.CreateCompatibleBitmap(dc,w,h)
    previous=gdi.SelectObject(memory,bitmap)
    try:
        assert user.PrintWindow(hwnd,memory,3),'PrintWindow failed'
        class Header(ctypes.Structure):
            _fields_=[('size',wintypes.DWORD),('width',wintypes.LONG),('height',wintypes.LONG),('planes',wintypes.WORD),('bits',wintypes.WORD),('compression',wintypes.DWORD),('image_size',wintypes.DWORD),('x',wintypes.LONG),('y',wintypes.LONG),('colors',wintypes.DWORD),('important',wintypes.DWORD)]
        header=Header(40,w,-h,1,32,0,w*h*4,0,0,0,0);buffer=ctypes.create_string_buffer(w*h*4)
        assert gdi.GetDIBits(memory,bitmap,0,h,buffer,ctypes.byref(header),0)==h
        image=Image.frombuffer('RGB',(w,h),buffer.raw,'raw','BGRX',0,1)
        import numpy as np
        assert np.asarray(image).std()>5,'Blank window capture'
        image.save(path)
    finally:
        gdi.SelectObject(memory,previous);gdi.DeleteObject(bitmap);gdi.DeleteDC(memory);user.ReleaseDC(hwnd,dc)


def main(a):
    out=a.output.resolve();assert not out.exists();out.mkdir(parents=True)
    old=read(ROOT/'results/compact_application_20261107.json')
    clip=ROOT/old['verification']['constructed_video']['source']
    images=sorted((ROOT/'dataset/cityscapes/leftImg8bit/val').glob('*/*_leftImg8bit.png'))
    source=images[499];original=sha(source)
    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [HERE/'app.py',HERE/'worker.py',Path(__file__),
        HERE/'启动人行道识别.cmd',EXP/'verified_compact_runtime.py',EXP/'reference_deployment_pipeline.py']}
    window=tk.Tk();window.withdraw();app=App(window)
    app.input_path.set(str(source));app.start()
    deadline=time.monotonic()+240
    try:
        while app.process and app.process.poll() is None or app.log_handles:
            window.update();time.sleep(.05)
            assert time.monotonic()<deadline,'Desktop worker did not complete in time'
        image_job=app.job;result=read(image_job/'result.json')
        assert result['state']=='complete' and result['model']=='L0' and result['frames']==1
        assert float(app.progress['value'])==100 and str(app.start_button['state'])=='normal'
        assert Path(result['output']).is_file() and Path(result['mask']).is_file()
        with Image.open(result['output']) as im:assert im.size==(2048,1024)
        assert sha(source)==original
        window.deiconify();window.geometry('1160x790+30+30');window.update()
        app.show_previews();window.update();time.sleep(.4)
        box=(window.winfo_rootx(),window.winfo_rooty(),window.winfo_rootx()+window.winfo_width(),window.winfo_rooty()+window.winfo_height())
        try:ImageGrab.grab(bbox=box).save(out/'desktop_verified.png')
        except OSError:capture_window(window,out/'desktop_verified.png')
        window.withdraw()
        # Actual UI cancellation request after at least one video frame.
        app.input_path.set(str(clip));app.start();deadline=time.monotonic()+240;cancelled=False
        while app.process and app.process.poll() is None or app.log_handles:
            window.update();time.sleep(.05)
            p=app.job/'status.json'
            if p.exists():
                s=read(p)
                if s.get('frames',0)>=1 and s.get('state')=='running' and not cancelled:
                    app.cancel();cancelled=True
            assert time.monotonic()<deadline
        cancel_job=app.job;s=read(cancel_job/'status.json')
        assert cancelled and s['state']=='cancelled' and not (cancel_job/'result.json').exists()
        assert not (cancel_job/'overlay.mp4').exists() and str(app.start_button['state'])=='normal'
    finally:window.destroy()

    def worker(name,input_file,model='L0'):
        job=out/name
        command=[sys.executable,str(HERE/'worker.py'),'--input',str(input_file),'--output',str(job),'--model',model,
                 '--source-note','Application acceptance on existing validation material; constructed clip is not natural video']
        r=subprocess.run(command,capture_output=True,timeout=240,creationflags=subprocess.CREATE_NO_WINDOW,
            env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8'))
        (out/(name+'.stderr.log')).write_bytes(r.stderr)
        return job,r
    baseline,r=worker('baseline_image',source,'A');assert r.returncode==0
    video,r=worker('full_video',clip);assert r.returncode==0
    import cv2
    cap=cv2.VideoCapture(str(video/'overlay.mp4'));count=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert frame.shape==(1024,2048,3);count+=1
    cap.release();assert count==32 and read(video/'result.json')['frames']==32
    invalid,r=worker('missing_input',out/'missing.png');assert r.returncode!=0
    assert read(invalid/'status.json')['state']=='error'
    with (EXP/'seed42_training.lock').open('a+b') as lock:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            busy,r=worker('busy_gpu',source);assert r.returncode!=0
            assert read(busy/'status.json')['state']=='error' and not (busy/'overlay.png').exists()
        finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
    before=sha(baseline/'result.json')
    _,r=worker('baseline_image',source,'A');assert r.returncode!=0 and sha(baseline/'result.json')==before
    for p,h in source_hashes.items():assert sha(ROOT/p)==h
    for job in (image_job,baseline,video):
        report=read(job/'result.json')
        for p in report['execution_verified'].values():
            assert set(p['provider_evidence']['node_executions'])=={'CUDAExecutionProvider'}
    proof=dict(status='pass',actual_Tk_UI_start_and_cancel_tested=True,UI_recovers_after_cancel=True,
        image_models=['A','L0'],full_constructed_video_frames=32,missing_input_handled=True,
        GPU_lock_refused=True,existing_result_overwrite_refused=True,original_input_unchanged=True,
        media_jobs=[str(p.relative_to(ROOT)) for p in (image_job,baseline,video)],cancel_job=str(cancel_job.relative_to(ROOT)),
        screenshot=str((out/'desktop_verified.png').relative_to(ROOT)),screenshot_sha256=sha(out/'desktop_verified.png'),
        source_hashes=source_hashes,scope='Existing configured Windows environment; constructed validation clip only. Not a fresh-machine installer or long-duration test.')
    (out/'verification.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in proof.items() if k not in ('source_hashes','media_jobs')},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args())
