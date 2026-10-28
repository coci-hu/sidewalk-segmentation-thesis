"""Preserve explicit BN in CUDA FP32; frozen acceptance limits and v1 evidence unchanged."""
import argparse
import contextlib
import gc
import json
import msvcrt
import os
from pathlib import Path
import traceback


def main(a):
    if a.backend=='cpu': os.environ['CUDA_VISIBLE_DEVICES']='-1'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    # Set visibility before importing torch; CPU pilot never initializes CUDA.
    import numpy as np
    import torch
    import onnx
    import onnxruntime as ort
    import common_runner as core
    import common_runner_v2 as fast
    import common_runner_mixed as mixed
    from ca_hswish_control import build_c
    from mixed_ca_candidate import build_candidate
    from deployment_equivalence import Agreement, provider_evidence
    import deployment_equivalence as gates
    import safe_progress_io as io
    out=a.output.resolve()
    assert out.is_relative_to(core.HERE) and out!=core.HERE and not out.exists()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    manifest=json.loads(a.models.read_text(encoding='utf-8'))
    assert (manifest['height'],manifest['width'])==(768,1024)
    assert manifest['mean']==[.485,.456,.406] and manifest['std']==[.229,.224,.225]
    assert a.backend=='cuda', 'This version validates the preserved-BN CUDA path only'
    disabled=['ConvBNFusion','ConvActivationFusion']
    assert manifest['execution']==dict(disabled_optimizers=disabled)
    selected=manifest['models']
    assert selected and len({(m['variant'],m['seed']) for m in selected})==len(selected)
    registry=json.loads(a.selected_results.read_text(encoding='utf-8'))
    by_run={(r['variant'],r['seed']):r for r in registry['results']}
    registry_hash=core.digest(a.selected_results)
    for item in selected:
        row=by_run[(item['variant'],item['seed'])]
        assert row['epochs']==100 and row['best_epoch']==item['selected_epoch']
        assert row['best_weights_sha256']==item['weights_sha256']
    expected='CPUExecutionProvider' if a.backend=='cpu' else 'CUDAExecutionProvider'
    if expected not in ort.get_available_providers(): raise RuntimeError('Requested provider unavailable')
    lock=(core.HERE/'seed42_training.lock').open('a+b') if a.backend=='cuda' else contextlib.nullcontext()
    with lock as handle:
        if a.backend=='cuda':
            handle.seek(0)
            # Do not wait or disturb running training. Refuse concurrent GPU validation.
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        try:
            out.mkdir(parents=True)
            sources=mixed.sources()
            for path in (Path(__file__),Path(gates.__file__),core.HERE/'validate_deployment_fp32.py'):
                sources[str(path.relative_to(core.PROJECT))]=core.digest(path)
            config_hash=core.digest(a.models)
            dataset=core.SidewalkDataset(fast.protocol(),'val')
            assert len(dataset)==500
            original_manifest=dataset.manifest()
            ids=list(range(500)) if a.mode=='full' else [0,1,2]
            core.json_write(out/'plan.json',dict(mode=a.mode,backend=a.backend,indices=ids,
                sources=sources,models_manifest_sha256=config_hash,selected_results_sha256=registry_hash,data_manifest=original_manifest,
                precision=dict(dtype='float32',pytorch_tf32=False,onnx_cuda_use_tf32=0),
                execution=dict(export_variant='preserved_BN',disabled_optimizers=disabled),
                versions=dict(torch=torch.__version__,onnx=onnx.__version__,onnxruntime=ort.__version__),
                prior_plan='step_06_2026-10-23/study_review_001/next_experiments.json',
                scope='Core labels, fixed26px boundary; allclose(exported,PyTorch); no speed measurements'))
            reports=[]
            for item in selected:
                variant,seed=item['variant'],item['seed']
                assert variant in ('A','B','C','D','E') and seed==42
                assert 1<=item['selected_epoch']<=100
                weights=Path(item['weights']).resolve(); graph=Path(item['onnx']).resolve()
                assert weights.is_relative_to(core.PROJECT) and graph.is_relative_to(core.HERE)
                assert core.digest(weights)==item['weights_sha256'] and core.digest(graph)==item['onnx_sha256']
                model=build_candidate(variant,seed) if variant in ('D','E') else build_c(seed) if variant=='C' else core.build_paired_model(variant,seed)
                model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
                model=model.to(a.backend).eval()
                state_before=core.state_digest(model.state_dict())
                graph_model=onnx.load(str(graph))
                onnx.checker.check_model(graph_model)
                assert not any(n.op_type in ('If','Loop','Scan') for n in graph_model.graph.node), 'Profiler sampling requires a static graph without control flow'
                opts=ort.SessionOptions(); opts.intra_op_num_threads=1; opts.inter_op_num_threads=1
                opts.enable_profiling=True; opts.profile_file_prefix=str(out/(item['name']+'_provider'))
                optimized_graph=out/(item['name']+'_optimized.onnx')
                opts.optimized_model_filepath=str(optimized_graph)
                if a.backend=='cuda': opts.add_session_config_entry('session.disable_cpu_ep_fallback','1')
                provider_list=[(expected,{'use_tf32':0})] if a.backend=='cuda' else [expected]
                session=ort.InferenceSession(str(graph),sess_options=opts,providers=provider_list,disabled_optimizers=disabled)
                original_bn=sum(n.op_type=='BatchNormalization' for n in graph_model.graph.node)
                optimized_bn=sum(n.op_type=='BatchNormalization' for n in onnx.load(str(optimized_graph)).graph.node)
                assert original_bn>0 and optimized_bn==original_bn, 'BN was fused despite the requested policy'
                session.disable_fallback()
                if session.get_providers()[0]!=expected: raise RuntimeError('Provider initialization fell back')
                provider_options=session.get_provider_options()
                if a.backend=='cuda' and provider_options[expected].get('use_tf32')!='0': raise RuntimeError('Requested FP32 without TF32 was not applied')
                inp=session.get_inputs(); outputs=session.get_outputs()
                assert len(inp)==len(outputs)==1 and inp[0].name=='image' and outputs[0].name=='logits'
                assert inp[0].type==outputs[0].type=='tensor(float)'
                assert inp[0].shape==[1,3,768,1024] and outputs[0].shape==[1,2,768,1024]
                agreement=Agreement()
                profile=None
                with torch.inference_mode():
                    for i in ids:
                        image,target=dataset[i]
                        tensor=image.unsqueeze(0).contiguous()
                        reference=model(tensor.to(a.backend)).cpu().numpy()
                        exported=session.run(['logits'],{'image':tensor.numpy()})[0]
                        assert reference.dtype==exported.dtype==np.float32
                        agreement.add(reference,exported,target.numpy(),str(i))
                        if len(agreement.records)==1:
                            # Assignment is static and fallback is disabled. Limit trace memory;
                            # continue numeric evaluation on every declared input in this session.
                            profile=Path(session.end_profiling())
                        if len(agreement.records)%25==0 or i==ids[-1]:
                            io.atomic_json(out/'status.json',dict(status='running',model=item['name'],images=len(agreement.records),target=len(ids)))
                assert profile is not None
                provider=provider_evidence(json.loads(profile.read_text(encoding='utf-8')),expected)
                report=agreement.summary()
                report.update(name=item['name'],variant=variant,seed=seed,selected_epoch=item['selected_epoch'],
                    weights_sha256=item['weights_sha256'],onnx_sha256=item['onnx_sha256'],provider_evidence=provider,
                    provider_options=provider_options,profiled_images=1,profile_sha256=core.digest(profile),
                    disabled_optimizers=disabled,original_BN_nodes=original_bn,optimized_BN_nodes=optimized_bn)
                core.json_write(out/(item['name']+'_per_image.json'),agreement.records)
                core.json_write(out/(item['name']+'_summary.json'),report)
                assert core.state_digest(model.state_dict())==state_before
                assert core.digest(weights)==item['weights_sha256'] and core.digest(graph)==item['onnx_sha256']
                reports.append(report)
                print(item['name'],report['status'],'images',len(ids),'max error',report['maximum_absolute_logit_error'],flush=True)
                del model,session,agreement,reference,exported,tensor,image,target
                gc.collect()
                if a.backend=='cuda': torch.cuda.empty_cache()
            assert core.SidewalkDataset(fast.protocol(),'val').manifest()==original_manifest
            assert core.digest(a.models)==config_hash
            assert core.digest(a.selected_results)==registry_hash
            for path,sha in sources.items(): assert core.digest(core.PROJECT/path)==sha
            passed=all(r['status']=='pass' for r in reports)
            core.json_write(out/'results.json',dict(results=reports,mode=a.mode,backend=a.backend))
            core.json_write(out/'verification.json',dict(status='pass' if passed else 'fail',mode=a.mode,backend=a.backend,
                models=len(reports),images_per_model=len(ids),source_hashes=sources,model_and_graph_hashes_unchanged=True,
                export_variant='preserved_BN',disabled_optimizers=disabled,
                same_input_metrics_and_actual_node_provider_checked=True,full500_metrics_verified=passed and a.mode=='full',
                GPU_verified=passed and a.backend=='cuda',latency_measured=False,training_modified=False))
            io.atomic_json(out/'status.json',dict(status='complete',checks_passed=passed))
            if not passed: raise RuntimeError('Frozen FP32 acceptance limits failed; keep evidence and investigate')
        except Exception:
            if out.exists():
                io.atomic_json(out/'failure.json',dict(status='failed',traceback=traceback.format_exc(),no_tolerance_relaxation=True))
            raise
        finally:
            if a.backend=='cuda':
                handle.seek(0); msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--selected-results',type=Path,default=Path(__file__).resolve().parents[2]/'results/abc_summary.json')
    p.add_argument('--mode',choices=('pilot','full'),required=True)
    p.add_argument('--backend',choices=('cpu','cuda'),required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
