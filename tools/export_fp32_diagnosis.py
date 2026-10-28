"""Compact trace of failed gates, bounded probes and an unchanged-threshold repair."""
from export_repository_summary import ROOT, EXP, digest, dump
from export_followup_summary import read, sanitize


def main():
    base=EXP/'step_10_2026-10-28'
    full=base/'ABC_fp32_cuda_full_001'
    diagnostic=base/'fp32_numerical_diagnosis_001'
    repair=base/'preserved_bn_probe_001'
    exported=base/'preserved_bn_reexport_cpu_001'
    assert read(full/'verification.json')['status']=='fail'
    observed=read(diagnostic/'results.json'); corrected=read(repair/'results.json')
    assert observed['status']=='diagnosis_complete' and corrected['status']=='probe_complete'
    initial=[]
    for r in read(full/'results.json')['results']:
        per=read(full/(r['name']+'_per_image.json'))
        initial.append(dict(name=r['name'],failing_images=[int(x['image_id']) for x in per if not x['logits_allclose']],
            changed_prediction_pixels=sum(x['argmax_disagreement_pixels'] for x in per),
            total_pixels=sum(x['pixels'] for x in per),maximum_absolute_logit_error=r['maximum_absolute_logit_error'],
            absolute_metric_difference_pp=r['absolute_metric_difference_pp'],checks=r['checks']))
    probes=[]
    for r in observed['results']:
        conditions=[]
        for c in r['conditions']:
            info=dict(name=c['name'],status=c['status'])
            if c['status']=='evaluated':
                info.update(subset_allclose=c['subset_allclose'],provider=c['provider_evidence']['provider'],
                    failed_images=[x['image_id'] for x in c['records'] if not x['vs_pytorch_cuda']['allclose']],
                    maximum_tolerance_ratio=max(x['vs_pytorch_cuda']['maximum_error_to_tolerance_ratio'] for x in c['records']))
            else: info['error']=c['error']
            conditions.append(info)
        cpu={key:dict(failed_images=[x['image_id'] for x in r['CPU_reference_comparisons'] if not x[key]['allclose']],
            maximum_tolerance_ratio=max(x[key]['maximum_error_to_tolerance_ratio'] for x in r['CPU_reference_comparisons']))
            for key in ('pytorch_cuda_vs_cpu','onnx_cpu_vs_pytorch_cpu')}
        probes.append(dict(name=r['name'],CPU_comparisons=cpu,conditions=conditions))
    repaired=[]
    for r in corrected['results']:
        assert r['subset_allclose'] and r['provider_evidence']['provider']=='CUDAExecutionProvider'
        repaired.append(dict(name=r['name'],indices=[x['image_id'] for x in r['records']],
            maximum_tolerance_ratio=max(x['agreement']['maximum_error_to_tolerance_ratio'] for x in r['records']),
            original_BN_nodes=r['graph_node_counts']['BatchNormalization'],optimized_BN_nodes=r['optimized_node_counts']['BatchNormalization'],
            weights_sha256=r['weights_sha256'],onnx_sha256=r['onnx_sha256']))
    export_proof=read(exported/'verification.json'); reproduction=read(exported/'reproduction_verification.json')
    guard=read(base/'preserved_bn_guard_001/verification.json')
    assert export_proof['status']==reproduction['status']==guard['status']=='pass'
    assert export_proof['preserve_bn'] and reproduction['all_three_graph_hashes_equal_probe']
    paths=[full/'verification.json',full/'results.json',diagnostic/'plan.json',diagnostic/'results.json',
        repair/'plan.json',repair/'results.json',exported/'verification.json',exported/'reproduction_verification.json',
        base/'preserved_bn_guard_001/verification.json']
    report=dict(status='diagnosis_complete',initial_full_validation=initial,backend_probes=probes,
        preserved_BN_probe=repaired,reexport=dict(status='pass',all_three_graph_hashes_equal_probe=True,
            source_hashes=export_proof['source_hashes']),GPU_guard=guard,
        tolerance_or_weights_changed=False,
        finding='Selected failing inputs pass when exporting with explicit BN and disabling ConvBN/ConvActivation fusion. Algorithm change alone did not resolve the gates; disabling all graph optimizations required CPU nodes and was rejected.',
        limitation='This is a combined export/execution change, not proof of a single faulty optimizer or general ONNX inaccuracy. Subset repair does not establish full500 acceptance; no speed benefit is claimed.',
        official_references=['https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html',
            'https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html'],
        sources={str(p.relative_to(ROOT)):digest(p) for p in paths})
    full_repair=ROOT/'results/deployment_cuda_fp32_preserved_bn.json'
    if full_repair.exists():
        final=read(full_repair)
        report['repaired_full500']=dict(status=final['status'],passed=final['full500_acceptance_passed'],source_sha256=digest(full_repair))
    else: report['repaired_full500']=dict(status='pending',passed=False)
    dump(ROOT/'results/fp32_numerical_diagnosis.json',sanitize(report))
    print('Exported original failure, bounded diagnostics, reproducible repair and current acceptance status.')


if __name__=='__main__': main()
