"""Check all500 unchanged decoded pixels/tensors plus exact binary postprocessing."""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from PIL import Image
import common_runner as core
import common_runner_v2 as fast
import deployment_pipeline as pipeline
import safe_progress_io as progress


def main(out):
    assert out.resolve().is_relative_to(core.HERE) and not out.exists()
    torch.set_num_threads(1); cv2.setNumThreads(1)
    dataset=core.SidewalkDataset(fast.protocol(),'val')
    out.mkdir(parents=True)
    source_hashes={str(p.relative_to(core.PROJECT)):core.digest(p) for p in
        [Path(__file__),Path(pipeline.__file__),Path(core.__file__),Path(fast.__file__),Path(progress.__file__)]}
    core.json_write(out/'plan.json',dict(images=500,source_hashes=source_hashes))
    start=time.perf_counter(); records=[]
    before=dataset.manifest()
    for i,(path,_) in enumerate(dataset.pairs):
        with Image.open(path) as image:
            rgb=image.convert('RGB'); pixels=np.array(rgb)
            reference=dataset.transform(rgb).unsqueeze(0).contiguous().numpy()
        bgr=pipeline.read_bgr(path)
        decoded_equal=np.array_equal(pixels,cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        optimized=pipeline.preprocess_bgr(bgr)
        tensor_equal=np.array_equal(reference,optimized)
        assert decoded_equal and tensor_equal, ('Input differs',i,float(np.max(np.abs(reference-optimized))))
        # Structured masks exercise edges, all-background, all-foreground and sparse pixels.
        mask=np.zeros(bgr.shape[:2],dtype=np.uint8)
        mask[:,(i*31)%mask.shape[1]:]=1
        if i%5==0: mask[:]=0
        if i%7==0: mask[:]=1
        if i%11==0: mask[::2,::3]=0
        old=bgr.copy(); chosen=mask==1
        old[chosen]=(.6*old[chosen]+.4*np.array([40,210,40])).astype(np.uint8)
        assert np.array_equal(old,pipeline.overlay_bgr(bgr,mask)), ('Overlay differs',i)
        records.append(dict(index=i,decoded_pixels_equal=True,input_tensor_equal=True,overlay_equal=True))
        if (i+1)%50==0:
            progress.atomic_json(out/'status.json',dict(status='running',images=i+1,target=500))
            print('Verified',i+1,flush=True)
    rng=np.random.default_rng(42)
    logits=rng.normal(size=(1,2,768,1024)).astype(np.float32)
    logits[:,1,::3,::5]=logits[:,0,::3,::5]
    assert np.array_equal(logits.argmax(1)[0],pipeline.binary_mask(logits))
    logits[0,0,0,0]=np.nan
    try: pipeline.binary_mask(logits)
    except ValueError: pass
    else: raise AssertionError('Nonfinite accepted')
    assert dataset.manifest()==before
    for path,sha in source_hashes.items(): assert core.digest(core.PROJECT/path)==sha
    core.json_write(out/'per_image.json',records)
    core.json_write(out/'verification.json',dict(status='pass',images=500,all_input_pixels_and_tensors_bitwise_equal=True,
        binary_argmax_and_ties_equal=True,overlays_bitwise_equal=True,nonfinite_rejected=True,
        source_hashes=source_hashes,
        elapsed_seconds=time.perf_counter()-start,CUDA_inference=False,
        meaning='Same exact model input and binary output rule; this is implementation equivalence, not new model accuracy'))
    progress.atomic_json(out/'status.json',dict(status='complete',images=500,checks_passed=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
