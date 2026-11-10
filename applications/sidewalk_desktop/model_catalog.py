"""Bind algorithm presentation to frozen evidence, independently of current inputs."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def algorithm(variant):
    effect=json.loads((ROOT/'results/compact_final_effect_20261107.json').read_text(encoding='utf-8'))
    profile=json.loads((ROOT/'results/lightweight_profile_20261101_v2.json').read_text(encoding='utf-8'))['results'][variant]
    acc=effect['accuracy_summary'][variant];speed=effect['timing_summary'][variant]
    return dict(name='L0 通道缩减 ENet' if variant=='L0' else 'A 精简 ENet 基线',
        channels='16 → 48 → 96 → 48 → 16' if variant=='L0' else '16 → 64 → 128 → 64 → 16',
        params=profile['params'],gflops=profile['gflops_conv_linear'],
        iou=acc['sidewalk']['mean_pct'],iou_sd=acc['sidewalk']['sample_std_pct'],
        boundary=acc['boundary']['mean_pct'],latency=speed['model_only']['mean_ms'],
        full_latency=speed['end_to_end']['mean_ms'])

def resolve_models(source,output):
    """Resolve relative deployment assets without rewriting the shipped manifest."""
    manifest=json.loads(source.read_text(encoding='utf-8'))
    for item in manifest['models']:
        for key in ('onnx','weights'):
            path=Path(item[key])
            if not path.is_absolute():path=(source.parent/path).resolve()
            if not path.is_file():raise FileNotFoundError('缺少模型文件，请按 README 准备模型包：'+path.name)
            item[key]=str(path)
    target=output/'resolved_models.json'
    with target.open('x',encoding='utf-8') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
    return target

def default_models():
    packaged=Path(__file__).resolve().parent/'models/models.json'
    if packaged.exists():return packaged
    return ROOT/'experiments/attention_gate/step_13_2026-11-07/A_L0_deployment_001/export/models.json'
