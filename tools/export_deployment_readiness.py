"""Small, explicit pilot evidence; never label a CPU pilot as full GPU deployment."""
import argparse
import datetime
from pathlib import Path
from export_repository_summary import ROOT, EXP, digest, dump
from export_followup_summary import read, sanitize


def main(a):
    pilot=EXP/'step_10_2026-10-27/fp32_cpu_pilot_002'
    synthetic=EXP/'step_10_2026-10-27/equivalence_checks_001/verification.json'
    guard=EXP/'step_10_2026-10-27/gpu_guard_check_002/verification.json'
    # Preserve the current audit by default; never silently rewind to an old snapshot.
    progress=a.progress_audit
    if progress is None:
        current=ROOT/'results/de_checkpoint_audit.json'
        progress=ROOT/read(current)['source']['path'] if current.exists() else EXP/'step_07_2026-10-27/progress_audit_001/verification.json'
    verification=read(pilot/'verification.json')
    assert verification['status']=='pass' and verification['images_per_model']==3
    assert not verification['full500_metrics_verified'] and not verification['GPU_verified']
    for relative,expected in verification['source_hashes'].items(): assert digest(ROOT/relative)==expected
    assert read(synthetic)['status']==read(guard)['status']=='pass'
    assert read(guard)['validator_sha256']==digest(EXP/'validate_deployment_fp32.py')
    report=dict(snapshot_date=datetime.date.today().isoformat(),pilot=read(pilot/'results.json'),verification=verification,
        synthetic_checks=read(synthetic),GPU_concurrency_guard=read(guard),
        sources={str(p.relative_to(ROOT)):digest(p) for p in (pilot/'results.json',pilot/'verification.json',synthetic,guard)},
        next_steps=['After training: CUDA pilot and full500 equivalence','Only after FP32 acceptance: FP16 and isolated latency/memory benchmark'],
        documentation_checked=['https://onnxruntime.ai/docs/api/python/api_summary.html',
            'https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html',
            'https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#use_tf32'],
        limitations='Two frozen A/C models, three existing validation images each, CPU only. No500-image acceptance, GPU inference, FPS, or physical embedded-device claim.')
    additional=[]
    for folder in a.additional_pilot:
        proof=read(folder/'verification.json')
        assert proof['status']=='pass' and proof['mode']=='pilot' and proof['backend']=='cpu'
        assert proof['images_per_model']==3 and not proof['full500_metrics_verified'] and not proof['GPU_verified']
        for relative,expected in proof['source_hashes'].items(): assert digest(ROOT/relative)==expected
        additional.append(dict(pilot=read(folder/'results.json'),verification=proof,
            sources={str(p.relative_to(ROOT)):digest(p) for p in (folder/'results.json',folder/'verification.json')}))
    if additional:
        report['additional_pilots']=additional
        report['limitations']='Frozen selected models, three existing validation images each, CPU only. No500-image acceptance, GPU inference, FPS, or physical embedded-device claim.'
    else:
        previous=ROOT/'results/deployment_readiness.json'
        if previous.exists():
            old=read(previous)
            if old.get('additional_pilots'):
                # Keep previously exported evidence unless explicitly replaced; reject stale sources.
                for item in old['additional_pilots']:
                    for relative,expected in item['verification']['source_hashes'].items(): assert digest(ROOT/relative)==expected
                report['additional_pilots']=old['additional_pilots']
                report['limitations']=old['limitations']
    # CPU pilot evidence stays intact; expose the latest GPU acceptance separately.
    for relative in ('results/deployment_cuda_fp32_preserved_bn.json','results/deployment_cuda_fp32.json'):
        path=ROOT/relative
        if not path.exists(): continue
        gpu=read(path)
        report['latest_cuda_full_validation']=dict(path=relative,sha256=digest(path),status=gpu['status'],
            full500_acceptance_passed=gpu['full500_acceptance_passed'],execution=gpu['execution'])
        report['next_steps']=(['FP16 equivalence under a separately declared precision protocol','Isolated latency/memory benchmark and application validation']
            if gpu['full500_acceptance_passed'] else ['Resolve FP32 numerical gate failures without silent tolerance relaxation','Repeat full500 acceptance before FP16 or latency conclusions'])
        report['limitations']='CPU pilot evidence is historical; latest CUDA full-validation state is reported separately. NoFPS or physical embedded-device claim.'
        break
    fp16_path=ROOT/'results/deployment_cuda_fp16.json'
    if fp16_path.exists():
        fp16=read(fp16_path)
        report['latest_fp16_validation']=dict(path='results/deployment_cuda_fp16.json',sha256=digest(fp16_path),
            status=fp16['status'],mode=fp16['mode'],decision=fp16['decision'],
            full500_acceptance_passed=fp16['full500_acceptance_passed'])
        if fp16['decision']=='retain_accepted_FP32_no_FP16_search':
            report['next_steps']=['Use accepted FP32 configuration in prototype',
                'Isolated FP32 latency/memory comparison under fixed measured power conditions',
                'Application stability and human-reviewed local image validation']
    dump(ROOT/'results/deployment_readiness.json',sanitize(report))
    audit=read(progress)
    assert audit['status']=='pass'
    audit['source']=dict(path=progress.relative_to(ROOT).as_posix(),sha256=digest(progress))
    dump(ROOT/'results/de_checkpoint_audit.json',sanitize(audit))
    print('Exported CPU pilot, synthetic/guard checks and immutable-checkpoint audit.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--progress-audit',type=lambda s:Path(s).resolve())
    p.add_argument('--additional-pilot',type=lambda s:Path(s).resolve(),action='append',default=[])
    main(p.parse_args())
