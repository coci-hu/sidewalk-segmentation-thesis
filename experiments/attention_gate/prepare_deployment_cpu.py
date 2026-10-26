"""CPU-only FP32 export preflight for fixed seed42 A/C; no speed claims."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import onnx
import onnxruntime as ort
import common_runner as core
import common_runner_v2 as fast
import common_runner_c as c
from ca_hswish_control import build_c


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-job', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    assert out.is_relative_to(core.HERE) and not out.exists()
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    pinned = c.sources()
    rows = json.loads((a.source_job / 'c_seed42_results.json').read_text(encoding='utf-8'))['results']
    image, _ = core.SidewalkDataset(fast.protocol(), 'val')[0]
    tensor = image.unsqueeze(0).contiguous()
    entries = []
    for row in rows:
        if row['variant'] not in ('A', 'C'):
            continue
        name = row['variant'] + '_seed42'
        model = build_c(42) if row['variant'] == 'C' else core.build_paired_model('A', 42)
        weight_hash = core.digest(row['best_weights'])
        model.load_state_dict(torch.load(row['best_weights'], map_location='cpu', weights_only=True), strict=True)
        model.eval()
        path = out / (name + '_fp32.onnx')
        with torch.inference_mode():
            reference = model(tensor).numpy()
            torch.onnx.export(model, tensor, str(path), export_params=True, opset_version=17,
                input_names=['image'], output_names=['logits'], do_constant_folding=True)
        onnx.checker.check_model(onnx.load(str(path)))
        options = ort.SessionOptions(); options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        exported = session.run(['logits'], {'image': tensor.numpy()})[0]
        max_abs = float(np.max(np.abs(reference - exported)))
        mismatch = float(np.mean(reference.argmax(1) != exported.argmax(1)))
        match = bool(np.allclose(reference, exported, rtol=1e-3, atol=1e-4))
        assert np.isfinite(exported).all() and match and mismatch <= 0.001
        assert core.digest(row['best_weights']) == weight_hash
        entries.append({'name': name, 'variant': row['variant'], 'seed': 42,
            'selected_epoch': row['best_epoch'], 'onnx': str(path), 'onnx_sha256': core.digest(path),
            'weights': row['best_weights'], 'weights_sha256': weight_hash,
            'max_absolute_logit_error_CPU': max_abs, 'argmax_disagreement_fraction_CPU': mismatch,
            'allclose_rtol1e3_atol1e4': match, 'images_checked': 1, 'backend': 'CPUExecutionProvider'})
        print(name, 'CPU export pass', max_abs, mismatch, flush=True)
    assert c.sources() == pinned
    core.json_write(out / 'models.json', {'models': entries, 'height': 768, 'width': 1024,
        'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
        'scope': 'CPU export preflight only; GPU and500-image metric equivalence and latency still pending'})
    core.json_write(out / 'verification.json', {'status': 'pass', 'models': 2, 'images_per_model': 1,
        'onnx_checker_pass': True, 'CPU_logit_equivalence_pass': True, 'source_hashes': pinned,
        'script_sha256': core.digest(__file__), 'GPU_verified': False, 'full500_metrics_verified': False,
        'latency_measured': False, 'training_started': False})


if __name__ == '__main__':
    main()
