"""Reconstruct exact-input and actual CLI proofs without publishing validation media."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'experiments/attention_gate'
sys.path.insert(0,str(HERE))
from deployment_equivalence import provider_evidence

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def relative(p): return Path(p).resolve().relative_to(ROOT).as_posix()
def check_sources(sources):
    for p,h in sources.items(): assert sha(ROOT/p)==h,p
def main(a):
    proof=read(a.correctness/'verification.json')
    rows=read(a.correctness/'per_image.json')
    assert proof['status']=='pass' and proof['images']==500
    assert [x['index'] for x in rows]==list(range(500))
    assert all(all(x[k] for k in ('decoded_pixels_equal','input_tensor_equal','overlay_equal')) for x in rows)
    assert all(proof[k] for k in ('all_input_pixels_and_tensors_bitwise_equal','binary_argmax_and_ties_equal','overlays_bitwise_equal','nonfinite_rejected'))
    check_sources(proof['source_hashes'])
    demo=read(a.demo/'verification.json'); check_sources(demo['source_hashes'])
    assert demo['status']=='pass' and demo['GPU_lock_rejected_before_inference']
    assert not demo['formal_FPS_claim'] and not demo['long_duration_stability_checked']
    assert not demo['self_collected_or_independent_dataset']
    accepted=read(ROOT/'results/deployment_cuda_fp32_preserved_bn.json')
    verified=[]; sources=[a.correctness/p for p in ('verification.json','per_image.json','plan.json')]
    sources.append(a.demo/'verification.json')
    cases=[(x['output'],x['report_sha256'],x['variant'],1) for x in demo['image_cases']]
    video=demo['constructed_video']; assert video['frames']==32 and video['all_output_frames_decoded']
    cases.append((video['output'],video['report_sha256'],'B',32))
    for filename,expected,variant,count in cases:
        media=Path(filename); report_path=media.with_suffix(media.suffix+'.json')
        assert media.is_file() and sha(report_path)==expected
        r=read(report_path); assert len(r['frames'])==count
        assert all(x['model']==variant+'_seed42' for x in r['frames'])
        index=str('ABC'.index(variant)); execution=r['execution_verified'][index]
        assert execution['provider_options']['CUDAExecutionProvider']['use_tf32']=='0'
        assert execution['disabled_optimizers']==['ConvBNFusion','ConvActivationFusion']
        model=execution['model']; original=next(x for x in accepted['results'] if x['variant']==variant)
        assert model['onnx_sha256']==original['onnx_sha256'] and model['weights_sha256']==original['weights_sha256']
        profiles=list(media.with_suffix(media.suffix+'.evidence').glob('*.json'))
        assert len(profiles)==1 and sha(profiles[0])==execution['profile_sha256']
        assert provider_evidence(read(profiles[0]),'CUDAExecutionProvider')==execution['provider_evidence']
        verified.append(dict(variant=variant,frames=count,media=relative(media),media_sha256=sha(media),
            report_sha256=sha(report_path),provider=execution['provider_evidence'],
            onnx_sha256=model['onnx_sha256'],source_note=r['source_note']))
        sources.extend([report_path,profiles[0]])
    output=dict(status='pass',input_implementation_proof=proof,demo_cases=verified,
        GPU_lock_rejected_before_inference=True,model_parameter_counts=demo['model_parameter_counts'],
        demo_source_hashes={p.replace('\\','/'):h for p,h in demo['source_hashes'].items()},
        exporter_sha256=sha(Path(__file__)),sources={relative(p):sha(p) for p in sources},
        constructed_video_only=True,constructed_video_frames=32,long_duration_stability_checked=False,
        formal_pipeline_comparison_complete=False,self_collected_dataset_required=False,
        claim='Functional interface and exact preprocessing proof; no new accuracy, natural-video temporal or formal FPS claim')
    (ROOT/'results/application_validation.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('PASS:500 exact inputs,3 CLI images,32 constructed video frames and actual CUDA proof')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--correctness',type=Path,required=True); p.add_argument('--demo',type=Path,required=True)
    main(p.parse_args())
