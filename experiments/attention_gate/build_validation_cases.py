"""Pick six cases by fixed historical IoU ranks before rendering predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import common_runner as core
import common_runner_v2 as fast
import reference_deployment_pipeline as pipeline
from verified_deployment_runtime import VerifiedRuntime
from metrics_v1 import confusion_matrix_numpy,scores_from_confusion,boundary_iou_author

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(a):
    out=a.output.resolve(); assert out.is_relative_to(core.HERE) and not out.exists()
    historical=core.HERE/'step_10_2026-10-28/ABC_fp32_cuda_preserved_bn_full_001/B_seed42_per_image.json'
    records=read(historical); assert len(records)==500
    eligible=[]
    for row in records:
        matrix=np.array(row['onnx']['confusion']); index=int(row['image_id'])
        if matrix[1].sum()>0:
            eligible.append(dict(index=index,iou=scores_from_confusion(matrix)['iou'][1]))
    eligible.sort(key=lambda x:(x['iou'],x['index']))
    chosen=[]
    for quantile in (0,.1,.25,.5,.75,1):
        rank=int(round(quantile*(len(eligible)-1)))
        chosen.append(dict(quantile=quantile,rank_zero_based=rank,**eligible[rank]))
    assert len({x['index'] for x in chosen})==6
    dataset=core.SidewalkDataset(fast.protocol(),'val'); torch.set_num_threads(1); cv2.setNumThreads(1)
    sources={str(p.relative_to(core.PROJECT)):sha(p) for p in [Path(__file__),Path(pipeline.__file__),
        core.HERE/'verified_deployment_runtime.py',core.HERE/'metrics_v1.py',historical]}
    input_hashes={str(p.relative_to(core.PROJECT)):sha(p) for row in chosen for p in dataset.pairs[row['index']]}
    out.mkdir(parents=True)
    core.json_write(out/'selection.json',dict(rule='Sort historical B42 ONNX per-image sidewalkIoU among GT-positive images; tie by index; ranks round(q*(n-1)) at q=[0,.1,.25,.5,.75,1]. Fixed before rendering.',
        eligible_images=len(eligible),excluded_GT_empty_images=500-len(eligible),selected=chosen,
        historical_sha256=sha(historical),source_hashes=sources,input_hashes=input_hashes))
    panels=[]; result_rows=[]
    with VerifiedRuntime(a.models,'cuda',out/'provider_evidence') as runtime:
        for row in chosen:
            index=row['index']; frame=pipeline.read_bgr(dataset.pairs[index][0])
            tensor=pipeline.preprocess_bgr(frame); frozen,target=dataset[index]
            assert np.array_equal(frozen.unsqueeze(0).contiguous().numpy(),tensor)
            logits=runtime.infer(1,tensor); pred=pipeline.binary_mask(logits); target=target.numpy().astype(np.uint8)
            matrix=confusion_matrix_numpy(pred,target); iou=scores_from_confusion(matrix)['iou'][1]
            boundary=boundary_iou_author(target==1,pred==1,width_pixels=26)
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            truth_full=pipeline.restore_mask(target,frame); pred_full=pipeline.restore_mask(pred,frame)
            gt_overlay=cv2.cvtColor(pipeline.overlay_bgr(frame,truth_full),cv2.COLOR_BGR2RGB)
            pred_overlay=cv2.cvtColor(pipeline.overlay_bgr(frame,pred_full),cv2.COLOR_BGR2RGB)
            error=(rgb*.35).astype(np.uint8)
            error[(truth_full==1)&(pred_full==1)]=[45,200,80]
            error[(truth_full==0)&(pred_full==1)]=[245,70,55]
            error[(truth_full==1)&(pred_full==0)]=[65,145,255]
            panels.append([rgb,gt_overlay,pred_overlay,error])
            np.savez_compressed(out/f'case_{index}_masks.npz',pred=pred,target=target)
            result_rows.append(dict(**row,current_iou_pct=100*iou,current_boundary_iou_pct=100*boundary if boundary is not None else None,
                historical_iou_pct=100*row['iou'],current_minus_historical_iou_pp=100*(iou-row['iou']),
                confusion=matrix.tolist(),image=str(dataset.pairs[index][0].relative_to(core.PROJECT)),
                label=str(dataset.pairs[index][1].relative_to(core.PROJECT)),mask_file=f'case_{index}_masks.npz',
                mask_file_sha256=sha(out/f'case_{index}_masks.npz')))
            print('Rendered source case',index,flush=True)
        execution=runtime.proofs
    fig,axes=plt.subplots(6,4,figsize=(18,16))
    for r,images in enumerate(panels):
        item=result_rows[r]
        for col,image in enumerate(images):
            axes[r,col].imshow(image); axes[r,col].set_xticks([]); axes[r,col].set_yticks([])
            if r==0: axes[r,col].set_title(['Input','Ground truth sidewalk','B42 FP32 prediction','Error map'][col],fontsize=13,pad=10)
            for spine in axes[r,col].spines.values(): spine.set_visible(False)
        axes[r,0].set_ylabel(f'Rank {item["rank_zero_based"]+1}/{len(eligible)} | ID {item["index"]}\nIoU {item["current_iou_pct"]:.2f}%',fontsize=10,labelpad=10)
    fig.suptitle('Fixed IoU-rank examples from reused Cityscapes validation data',fontsize=17,y=.99)
    fig.text(.5,.012,'Error colors: green = true sidewalk | red = false positive | blue = missed sidewalk. Six examples are not an aggregate test result.',ha='center',fontsize=11)
    fig.subplots_adjust(left=.095,right=.995,top=.955,bottom=.04,hspace=.13,wspace=.025)
    fig.savefig(out/'B42_validation_cases.png',dpi=180); plt.close(fig)
    for p,h in sources.items(): assert sha(core.PROJECT/p)==h
    for p,h in input_hashes.items(): assert sha(core.PROJECT/p)==h
    core.json_write(out/'result.json',dict(status='pass',selection_rule=read(out/'selection.json')['rule'],
        eligible_images=len(eligible),excluded_GT_empty_images=500-len(eligible),cases=result_rows,execution=execution,
        figure='B42_validation_cases.png',figure_sha256=sha(out/'B42_validation_cases.png'),
        source_hashes=sources,input_hashes=input_hashes,selection_sha256=sha(out/'selection.json'),
        validation_reused_for_checkpoint_selection=True,causal_failure_diagnosis=False,
        note='Rank-based visual illustration only; current per-case confusion is stored, and any difference from historical values is reported.'))
    print('Six fixed-rank cases saved; no new training or dataset.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--models',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); main(p.parse_args())
