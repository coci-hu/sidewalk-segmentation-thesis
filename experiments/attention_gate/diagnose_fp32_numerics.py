"""Bounded diagnostics after full FP32 failure; no tolerance or weight changes."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
import argparse
import gc
import json
import msvcrt
from pathlib import Path
import numpy as np
import onnxruntime as ort
import common_runner as core
import common_runner_v2 as fast
from ca_hswish_control import build_c
from deployment_equivalence import LIMITS, provider_evidence


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def comparison(reference,actual):
    assert np.isfinite(reference).all() and np.isfinite(actual).all()
    diff=np.abs(actual-reference)
    allowed=LIMITS['logit_atol']+LIMITS['logit_rtol']*np.abs(reference)
    ratio=diff/allowed
    flat=np.argsort(ratio.reshape(-1))[-3:][::-1]
    examples=[]
    for i in flat:
        index=np.unravel_index(int(i),ratio.shape)
        examples.append(dict(index=list(map(int,index)),reference=float(reference[index]),actual=float(actual[index]),
            absolute_error=float(diff[index]),allowed=float(allowed[index]),ratio=float(ratio[index])))
    return dict(allclose=bool(np.allclose(actual,reference,rtol=LIMITS['logit_rtol'],atol=LIMITS['logit_atol'])),
        violation_elements=int(np.count_nonzero(ratio>1)),total_elements=reference.size,
        maximum_absolute_error=float(diff.max()),maximum_error_to_tolerance_ratio=float(ratio.max()),
        argmax_disagreement_pixels=int(np.count_nonzero(reference.argmax(1)!=actual.argmax(1))),worst_elements=examples)


def main(a):
    job=a.full_job.resolve(); out=a.output.resolve()
    if not out.is_relative_to(core.HERE) or out==core.HERE or out.exists(): raise ValueError('New output directory required')
    proof=read(job/'verification.json')
    assert proof['mode']=='full' and proof['backend']=='cuda' and proof['images_per_model']==500
    assert read(job/'status.json')['status']=='complete'
    manifest=read(a.models)
    assert core.digest(a.models)==read(job/'plan.json')['models_manifest_sha256']
    for path,sha in proof['source_hashes'].items(): assert core.digest(core.PROJECT/path)==sha
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    with (core.HERE/'seed42_training.lock').open('a+b') as lock:
        lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            out.mkdir(parents=True)
            conditions=[('original','EXHAUSTIVE',ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
                ('default_algorithm','DEFAULT',ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
                ('no_graph_optimizations','EXHAUSTIVE',ort.GraphOptimizationLevel.ORT_DISABLE_ALL)]
            indices={}
            for item in manifest['models']:
                per=read(job/(item['name']+'_per_image.json'))
                failing=[r for r in per if not r['logits_allclose']]
                worst=sorted(failing,key=lambda r:(-r['maximum_absolute_logit_error'],int(r['image_id'])))[:3]
                indices[item['name']]=sorted({0}|{int(r['image_id']) for r in failing[:3]+worst})
            core.json_write(out/'plan.json',dict(full_job=str(job),full_proof_sha256=core.digest(job/'verification.json'),
                models_manifest_sha256=core.digest(a.models),indices=indices,
                conditions=[dict(name=n,cudnn_conv_algo_search=alg,graph_optimization=str(opt)) for n,alg,opt in conditions],
                limits=LIMITS,scope='First3 and worst3 failing images plus index0; diagnostic selection, not full validation.',
                references=['https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html',
                    'https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html'],script_sha256=core.digest(__file__)))
            dataset=core.SidewalkDataset(fast.protocol(),'val'); all_reports=[]
            for item in manifest['models']:
                variant=item['variant']; assert variant in 'ABC'
                assert core.digest(item['weights'])==item['weights_sha256'] and core.digest(item['onnx'])==item['onnx_sha256']
                model=build_c(42) if variant=='C' else core.build_paired_model(variant,42)
                model.load_state_dict(torch.load(item['weights'],map_location='cpu',weights_only=True),strict=True); model.eval()
                before=core.state_digest(model.state_dict()); tensors={}; cpu={}; cuda={}; cpu_graph={}
                options=ort.SessionOptions(); options.intra_op_num_threads=1; options.inter_op_num_threads=1
                session=ort.InferenceSession(item['onnx'],sess_options=options,providers=['CPUExecutionProvider']); session.disable_fallback()
                with torch.inference_mode():
                    for i in indices[item['name']]:
                        tensors[i]=dataset[i][0].unsqueeze(0).contiguous()
                        cpu[i]=model(tensors[i]).numpy()
                        cpu_graph[i]=session.run(['logits'],{'image':tensors[i].numpy()})[0]
                    del session
                    model=model.cuda()
                    for i,tensor in tensors.items(): cuda[i]=model(tensor.cuda()).cpu().numpy()
                assert core.state_digest(model.cpu().state_dict())==before
                del model; gc.collect(); torch.cuda.empty_cache()
                base=[dict(image_id=i,pytorch_cuda_vs_cpu=comparison(cpu[i],cuda[i]),
                    onnx_cpu_vs_pytorch_cpu=comparison(cpu[i],cpu_graph[i])) for i in tensors]
                model_report=dict(name=item['name'],weights_sha256=item['weights_sha256'],onnx_sha256=item['onnx_sha256'],CPU_reference_comparisons=base,conditions=[])
                for name,algorithm,optimization in conditions:
                    session=None
                    try:
                        options=ort.SessionOptions(); options.intra_op_num_threads=1; options.inter_op_num_threads=1
                        options.graph_optimization_level=optimization
                        options.add_session_config_entry('session.disable_cpu_ep_fallback','1')
                        options.enable_profiling=True; options.profile_file_prefix=str(out/(item['name']+'_'+name))
                        session=ort.InferenceSession(item['onnx'],sess_options=options,
                            providers=[('CUDAExecutionProvider',dict(use_tf32=0,cudnn_conv_algo_search=algorithm))])
                        session.disable_fallback()
                        applied=session.get_provider_options()['CUDAExecutionProvider']
                        assert applied['use_tf32']=='0' and applied['cudnn_conv_algo_search']==algorithm
                        records=[]; profile=None
                        for i,tensor in tensors.items():
                            actual=session.run(['logits'],{'image':tensor.numpy()})[0]
                            records.append(dict(image_id=i,vs_pytorch_cuda=comparison(cuda[i],actual),vs_pytorch_cpu=comparison(cpu[i],actual)))
                            if profile is None: profile=Path(session.end_profiling())
                        provider=provider_evidence(read(profile),'CUDAExecutionProvider')
                        model_report['conditions'].append(dict(name=name,status='evaluated',provider_options=applied,
                            provider_evidence=provider,profile_sha256=core.digest(profile),records=records,
                            subset_allclose=all(r['vs_pytorch_cuda']['allclose'] for r in records)))
                    except Exception as exc:
                        model_report['conditions'].append(dict(name=name,status='failed',error=str(exc)))
                    finally:
                        del session; gc.collect(); torch.cuda.empty_cache()
                assert core.digest(item['weights'])==item['weights_sha256'] and core.digest(item['onnx'])==item['onnx_sha256']
                all_reports.append(model_report)
                core.json_write(out/(item['name']+'_diagnosis.json'),model_report)
                print(item['name'],[(r['name'],r.get('subset_allclose',r['status'])) for r in model_report['conditions']],flush=True)
            for path,sha in proof['source_hashes'].items(): assert core.digest(core.PROJECT/path)==sha
            core.json_write(out/'results.json',dict(status='diagnosis_complete',results=all_reports,
                full500_acceptance=False,tolerances_changed=False,weights_changed=False,
                limitation='Subset backend investigation only; any revised execution configuration needs fresh full500 verification.'))
        finally:
            lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full-job',type=Path,required=True)
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
