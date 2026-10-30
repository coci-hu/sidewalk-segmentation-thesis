"""Verify the reference demo and fixed-rank case masks before compact publication."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from export_pipeline_comparison import ROOT,HERE,read,sha,rel,provider_evidence
from export_followup_summary import sanitize
sys.path.insert(0,str(HERE))
from metrics_v1 import confusion_matrix_numpy,scores_from_confusion,boundary_iou_author

def main(a):
    demo=read(a.demo/'verification.json'); cases=read(a.cases/'result.json'); selection=read(a.cases/'selection.json')
    assert demo['status']==cases['status']=='pass'
    assert demo['reference_input_exact_indices']==[0,207,499]
    assert demo['pipeline']=='reference_torchvision_argmax_numpy_overlay' and demo['GPU_lock_rejected_before_inference']
    assert demo['constructed_video']['frames']==32 and demo['constructed_video']['all_output_frames_decoded']
    sources=[a.demo/'verification.json',a.cases/'result.json',a.cases/'selection.json']; code={}
    for hashes in (demo['source_hashes'],cases['source_hashes']):
        for p,h in hashes.items():
            assert sha(ROOT/p)==h
            if 'step_' not in p: code[p.replace('\\','/')]=h
    accepted=read(ROOT/'results/deployment_cuda_fp32_preserved_bn.json')
    media=[]
    for item in demo['image_cases']+[demo['constructed_video']]:
        p=Path(item['output']); report_path=p.with_suffix(p.suffix+'.json'); report=read(report_path)
        assert sha(report_path)==item['report_sha256'] and report['pipeline']==demo['pipeline']
        assert len(report['frames'])==(32 if p.suffix=='.mp4' else 1)
        for execution in report['execution_verified'].values():
            model=execution['model']; original=next(x for x in accepted['results'] if x['variant']==model['variant'])
            assert model['onnx_sha256']==original['onnx_sha256'] and model['weights_sha256']==original['weights_sha256']
            assert execution['provider_options']['CUDAExecutionProvider']['use_tf32']=='0'
            profiles=list(p.with_suffix(p.suffix+'.evidence').glob('*.json'))
            assert len(profiles)==1 and sha(profiles[0])==execution['profile_sha256']
            assert provider_evidence(read(profiles[0]),'CUDAExecutionProvider')==execution['provider_evidence']
            sources.extend(profiles)
        media.append(dict(path=rel(p),sha256=sha(p),frames=len(report['frames'])))
        sources.append(report_path)
    historical=HERE/'step_10_2026-10-28/ABC_fp32_cuda_preserved_bn_full_001/B_seed42_per_image.json'
    ranked=sorted((dict(index=int(x['image_id']),iou=scores_from_confusion(x['onnx']['confusion'])['iou'][1])
        for x in read(historical) if sum(x['onnx']['confusion'][1])>0),key=lambda x:(x['iou'],x['index']))
    assert selection['historical_sha256']==sha(historical) and cases['selection_sha256']==sha(a.cases/'selection.json')
    expected=[ranked[round(q*(len(ranked)-1))]['index'] for q in (0,.1,.25,.5,.75,1)]
    assert [x['index'] for x in cases['cases']]==expected
    for row in cases['cases']:
        mask_file=a.cases/row['mask_file']; assert sha(mask_file)==row['mask_file_sha256']
        with np.load(mask_file,allow_pickle=False) as masks:
            pred,target=masks['pred'],masks['target']
            matrix=confusion_matrix_numpy(pred,target); bi=boundary_iou_author(target==1,pred==1,width_pixels=26)
        assert matrix.tolist()==row['confusion']
        assert abs(100*scores_from_confusion(matrix)['iou'][1]-row['current_iou_pct'])<1e-10
        assert abs(100*bi-row['current_boundary_iou_pct'])<1e-10
        sources.append(mask_file)
    assert sha(a.cases/cases['figure'])==cases['figure_sha256']
    output=dict(status='pass',reference_demo=dict(pipeline=demo['pipeline'],source_input_exact_indices=[0,207,499],
        GPU_lock_rejected=True,media=media,constructed_video_only=True,long_duration_checked=False),
        case_selection=dict(rule=selection['rule'],eligible_images=len(ranked),excluded_GT_empty_images=500-len(ranked),
            rows=cases['cases'],figure=rel(a.cases/cases['figure']),figure_sha256=cases['figure_sha256'],
            label_metrics_independently_reconstructed=True,validation_reused_for_selection=True),
        source_hashes=code,sources={rel(p):sha(p) for p in sources},exporter_sha256=sha(Path(__file__)))
    target=ROOT/'results/reference_application.json'; assert not target.exists()
    target.write_text(json.dumps(sanitize(output),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('PASS: reference CLI and six predetermined case masks reconstructed')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--demo',type=Path,required=True)
    p.add_argument('--cases',type=Path,required=True); main(p.parse_args())
