# Versioned A/L0 adapter of export_selected_fp32.py; old file and acceptance limits unchanged.
"""CPU-only export of already selected seed42 weights; never select new weights."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import gc
import json
from pathlib import Path
import traceback
import numpy as np
import torch
import onnx
import onnxruntime as ort
import common_runner as core
import common_runner_v2 as fast
import common_runner_mixed as mixed
from ca_hswish_control import build_c
from mixed_ca_candidate import build_candidate
from compact_candidate import build_compact
from deployment_equivalence import LIMITS


def main(a):
    out=a.output.resolve()
    if not out.is_relative_to(core.HERE) or out==core.HERE or out.exists():
        raise ValueError('Use a new output directory inside attention_gate')
    torch.set_num_threads(1)
    disabled=['ConvBNFusion','ConvActivationFusion'] if a.preserve_bn else []
    registry_hash=core.digest(a.selected_results)
    rows=json.loads(a.selected_results.read_text(encoding='utf-8'))['results']
    selected=[]
    if len(a.variants)!=len(set(a.variants)): raise ValueError('Repeated variant')
    for variant in a.variants:
        matches=[r for r in rows if r['variant']==variant and r['seed']==42]
        if len(matches)!=1: raise ValueError('Need one verified seed42 result for '+variant)
        row=matches[0]
        if row['epochs']!=100 or not 1<=row['best_epoch']<=100:
            raise ValueError('Only completed100-epoch runs are eligible')
        weights=Path(row['best_weights'])
        if not weights.is_absolute(): weights=core.PROJECT/weights
        weights=weights.resolve()
        if not weights.is_relative_to(core.PROJECT): raise ValueError('Weights outside project')
        if core.digest(weights)!=row['best_weights_sha256']: raise ValueError('Selected weights changed')
        selected.append((row,weights))
    out.mkdir(parents=True)
    sources=mixed.sources()
    for path in (Path(__file__),core.HERE/'deployment_equivalence.py',core.HERE/'compact_candidate.py'):
        sources[str(path.relative_to(core.PROJECT))]=core.digest(path)
    try:
        dataset=core.SidewalkDataset(fast.protocol(),'val')
        manifest=dataset.manifest()
        image,_=dataset[0]
        tensor=image.unsqueeze(0).contiguous()
        entries=[]
        for row,weights in selected:
            variant=row['variant']; name=variant+'_seed42'
            model=build_compact(variant,42) if variant=='L0' else build_candidate(variant,42) if variant in ('D','E') else build_c(42) if variant=='C' else core.build_paired_model(variant,42)
            model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
            model.eval()
            state_before=core.state_digest(model.state_dict())
            path=out/(name+'_fp32.onnx')
            with torch.inference_mode():
                reference=model(tensor).numpy()
                torch.onnx.export(model,tensor,str(path),export_params=True,opset_version=17,
                    input_names=['image'],output_names=['logits'],do_constant_folding=not a.preserve_bn)
            onnx.checker.check_model(onnx.load(str(path)))
            options=ort.SessionOptions(); options.intra_op_num_threads=1; options.inter_op_num_threads=1
            if a.preserve_bn:
                assert any(n.op_type=='BatchNormalization' for n in onnx.load(str(path)).graph.node)
                options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'],disabled_optimizers=disabled)
            session.disable_fallback()
            assert session.get_providers()==['CPUExecutionProvider']
            exported=session.run(['logits'],{'image':tensor.numpy()})[0]
            assert reference.shape==exported.shape==(1,2,768,1024)
            assert reference.dtype==exported.dtype==np.float32
            match=bool(np.allclose(exported,reference,rtol=LIMITS['logit_rtol'],atol=LIMITS['logit_atol']))
            mismatch=float(np.mean(reference.argmax(1)!=exported.argmax(1)))
            assert np.isfinite(reference).all() and np.isfinite(exported).all()
            assert match and mismatch<=LIMITS['argmax_disagreement_fraction']
            assert core.state_digest(model.state_dict())==state_before
            assert core.digest(weights)==row['best_weights_sha256']
            entries.append(dict(name=name,variant=variant,seed=42,selected_epoch=row['best_epoch'],
                onnx=str(path),onnx_sha256=core.digest(path),weights=str(weights),weights_sha256=core.digest(weights),
                max_absolute_logit_error_CPU=float(np.max(np.abs(reference-exported))),
                argmax_disagreement_fraction_CPU=mismatch,images_checked=1,backend='CPUExecutionProvider'))
            print(name,'CPU export preflight passed',flush=True)
            del model,session,reference,exported
            gc.collect()
        assert not torch.cuda.is_initialized()
        assert core.digest(a.selected_results)==registry_hash
        assert core.SidewalkDataset(fast.protocol(),'val').manifest()==manifest
        for path,sha in sources.items(): assert core.digest(core.PROJECT/path)==sha
        core.json_write(out/'models.json',dict(models=entries,height=768,width=1024,
            mean=[.485,.456,.406],std=[.229,.224,.225],
            execution=dict(disabled_optimizers=disabled),
            scope='One-image CPU export preflight. Run validate_deployment_fp32 for profiled pilot and full500 acceptance.'))
        core.json_write(out/'verification.json',dict(status='pass',models=len(entries),images_per_model=1,
            source_hashes=sources,selected_results_sha256=registry_hash,data_manifest=manifest,
            preserve_bn=a.preserve_bn,disabled_optimizers=disabled,
            weights_and_model_state_unchanged=True,onnx_checker_pass=True,CPU_logits_checked=True,
            CUDA_initialized=False,GPU_verified=False,full500_metrics_verified=False,latency_measured=False))
    except Exception:
        core.json_write(out/'failure.json',dict(status='failed',traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selected-results',type=Path,default=core.PROJECT/'results/abc_summary.json')
    p.add_argument('--variants',nargs='+',choices=['A','L0'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preserve-bn',action='store_true',help='Keep BatchNormalization explicit and disable ConvBN/ConvActivation fusion')
    main(p.parse_args())
