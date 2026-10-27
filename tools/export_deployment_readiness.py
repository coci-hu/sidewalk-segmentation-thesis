"""Small, explicit pilot evidence; never label a CPU pilot as full GPU deployment."""
from export_repository_summary import ROOT, EXP, digest, dump
from export_followup_summary import read, sanitize


def main():
    pilot=EXP/'step_10_2026-10-27/fp32_cpu_pilot_002'
    synthetic=EXP/'step_10_2026-10-27/equivalence_checks_001/verification.json'
    guard=EXP/'step_10_2026-10-27/gpu_guard_check_002/verification.json'
    progress=EXP/'step_07_2026-10-27/progress_audit_001/verification.json'
    verification=read(pilot/'verification.json')
    assert verification['status']=='pass' and verification['images_per_model']==3
    assert not verification['full500_metrics_verified'] and not verification['GPU_verified']
    for relative,expected in verification['source_hashes'].items(): assert digest(ROOT/relative)==expected
    assert read(synthetic)['status']==read(guard)['status']=='pass'
    assert read(guard)['validator_sha256']==digest(EXP/'validate_deployment_fp32.py')
    report=dict(snapshot_date='2026-10-27',pilot=read(pilot/'results.json'),verification=verification,
        synthetic_checks=read(synthetic),GPU_concurrency_guard=read(guard),
        sources={str(p.relative_to(ROOT)):digest(p) for p in (pilot/'results.json',pilot/'verification.json',synthetic,guard)},
        next_steps=['After training: CUDA pilot and full500 equivalence','Only after FP32 acceptance: FP16 and isolated latency/memory benchmark'],
        documentation_checked=['https://onnxruntime.ai/docs/api/python/api_summary.html',
            'https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html',
            'https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#use_tf32'],
        limitations='Two frozen A/C models, three existing validation images each, CPU only. No500-image acceptance, GPU inference, FPS, or physical embedded-device claim.')
    dump(ROOT/'results/deployment_readiness.json',sanitize(report))
    audit=read(progress)
    assert audit['status']=='pass'
    audit['source']=dict(path=progress.relative_to(ROOT).as_posix(),sha256=digest(progress))
    dump(ROOT/'results/de_checkpoint_audit.json',sanitize(audit))
    print('Exported CPU pilot, synthetic/guard checks and immutable-checkpoint audit.')


if __name__=='__main__': main()
