"""Read-only checks for a source checkout; needs only the Python standard library."""
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def main():
    hashes = read('results/frozen_source_hashes.json')
    for relative, expected in hashes.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    proof = read('experiments/attention_gate/step_05_2026-10-14/loader_optimization_001/verification.json')
    assert proof['status'] == 'pass' and proof['selected_workers'] == 4
    for relative, expected in proof['core_source_hashes'].items():
        assert hashes[relative.replace('\\', '/')] == expected, relative
    assert hashes['experiments/attention_gate/check_loader_optimization.py'] == proof['test_source_sha256']
    data = read('results/abc_summary.json')
    rows = data['results']
    assert len(rows) == 9 and {(r['variant'], r['seed']) for r in rows} == {(v, s) for v in 'ABC' for s in (42, 43, 44)}
    metrics = ('best_global_miou_pct', 'best_sidewalk_iou_pct', 'best_boundary_iou_pct')
    for row in rows:
        assert row['epochs'] == 100 and 1 <= row['best_epoch'] <= 100
        for key in ('best_weights', 'final_checkpoint'):
            assert not Path(row[key]).is_absolute() and '..' not in Path(row[key]).parts
        for metric in metrics:
            assert 0 <= row[metric] <= 100
    for variant in 'ABC':
        for metric in metrics:
            values = [r[metric] for r in rows if r['variant'] == variant]
            summary = data['summary'][variant][metric]
            assert abs(statistics.mean(values) - summary['mean']) < 1e-10
            assert abs(statistics.stdev(values) - summary['sample_std_ddof1']) < 1e-10
    for comparison in ('C_minus_A', 'C_minus_B'):
        other = comparison[-1]
        for metric in metrics:
            pairs = data['paired_percentage_points'][comparison][metric]
            differences = []
            for seed in (42, 43, 44):
                indexed = {r['variant']: r for r in rows if r['seed'] == seed}
                value = indexed['C'][metric] - indexed[other][metric]
                assert abs(value - pairs['per_seed'][str(seed)]) < 1e-10
                differences.append(value)
            assert abs(statistics.mean(differences) - pairs['mean']) < 1e-10
            assert abs(statistics.stdev(differences) - pairs['sample_std_ddof1']) < 1e-10
    if data['status']['full_sensitivity_complete']:
        sensitivity=read('results/sensitivity_summary.json')
        assert sensitivity['verification']['status']=='pass'
        assert len(sensitivity['results'])==9 and sensitivity['verification']['images_per_model']==500
        by_run={(r['variant'],r['seed']):r for r in rows}
        for r in sensitivity['results']:
            old=by_run[(r['variant'],r['seed'])]
            assert r['weights_sha256']==old['best_weights_sha256'] and r['selected_epoch']==old['best_epoch']
            assert abs(r['policies']['core']['sidewalk_iou_pct']-old['best_sidewalk_iou_pct'])<1e-7
            assert abs(r['core_boundary']['26']['mean_pct']-old['best_boundary_iou_pct'])<1e-7
        diagnosis=read('results/bc_diagnosis_summary.json')
        assert diagnosis['verification']['status']=='pass' and len(diagnosis['results'])==6
        assert diagnosis['verification']['images_per_model']==54
        for r in diagnosis['results']:
            assert r['weights_sha256']==by_run[(r['variant'],r['seed'])]['best_weights_sha256']
            assert len(r['conditions'])==10 and all(c['images']==54 for c in r['conditions'].values())
        acceptance_path=ROOT/'results/de_acceptance.json'
        if acceptance_path.exists():
            proof=read('results/de_acceptance.json')
            assert proof['status']=='pass' and proof['gpu_fullsize_D_E'] and proof['fresh_process_full_state_resume_D_E']
            assert proof['sensitivity_and_diagnosis_reviewed'] and proof['diagnostic_raw_aggregates_reconstructed']
            for relative,expected in proof['source_hashes'].items():
                assert hashlib.sha256((ROOT/relative.replace('\\','/')).read_bytes()).hexdigest()==expected
        print('PASS: same-weight sensitivity and diagnosis linkage; D/E proof source identity when present.')
    if (ROOT/'results/deployment_readiness.json').exists():
        readiness=read('results/deployment_readiness.json')
        verification=readiness['verification']
        assert verification['status']=='pass' and verification['mode']=='pilot' and verification['backend']=='cpu'
        assert verification['models']==2 and verification['images_per_model']==3
        assert not verification['full500_metrics_verified'] and not verification['GPU_verified'] and not verification['latency_measured']
        for relative,expected in verification['source_hashes'].items():
            assert hashlib.sha256((ROOT/relative.replace('\\','/')).read_bytes()).hexdigest()==expected
        assert readiness['GPU_concurrency_guard']['occupied_training_lock_rejected_before_model_or_ORT_session']
        by_run={(r['variant'],r['seed']):r for r in rows}
        for r in readiness['pilot']['results']:
            assert r['status']=='pass' and r['images']==3 and r['profiled_images']==1
            assert r['provider_evidence']['provider']=='CPUExecutionProvider'
            assert r['weights_sha256']==by_run[(r['variant'],r['seed'])]['best_weights_sha256']
        seen={(r['variant'],r['seed']) for r in readiness['pilot']['results']}
        for additional in readiness.get('additional_pilots',[]):
            proof=additional['verification']
            assert proof['status']=='pass' and proof['mode']=='pilot' and proof['backend']=='cpu'
            assert proof['images_per_model']==3 and proof['models']==len(additional['pilot']['results'])
            assert not proof['GPU_verified'] and not proof['full500_metrics_verified'] and not proof['latency_measured']
            for relative,expected in proof['source_hashes'].items():
                assert hashlib.sha256((ROOT/relative.replace('\\','/')).read_bytes()).hexdigest()==expected
            for r in additional['pilot']['results']:
                identity=(r['variant'],r['seed'])
                assert identity not in seen
                seen.add(identity)
                assert r['status']=='pass' and r['images']==3 and r['profiled_images']==1
                assert r['provider_evidence']['provider']=='CPUExecutionProvider'
                assert r['weights_sha256']==by_run[identity]['best_weights_sha256']
                assert r['selected_epoch']==by_run[identity]['best_epoch']
        print('PASS: CPU-only deployment pilot scope, selected weights and source identity.')
    if (ROOT/'results/de_review.json').exists():
        review=read('results/de_review.json')
        assert review['status']=='pass' and review['exploratory_seed_only']
        indexed={r['variant']:r for r in review['results']}
        assert len(review['results'])==5 and set(indexed)==set('ABCDE')
        assert all(r['seed']==42 and r['epochs']==100 for r in indexed.values())
        for old in rows:
            if old['seed']==42:
                assert all(indexed[old['variant']][k]==old[k] for k in metrics+('best_epoch','best_weights_sha256'))
        passing=[v for v in 'DE' if all(indexed[v][k]>=indexed['C'][k] for k in metrics[1:])
            and any(indexed[v][k]>indexed['C'][k] for k in metrics[1:])]
        assert passing==review['candidates_passing_screen']
        assert review['decision']==('review_and_repeat_both_candidates' if passing else 'stop_DE_branch_and_continue_baseline_deployment')
        for name,difference in review['paired_percentage_points'].items():
            first,second=name.split('_minus_')
            for metric in metrics: assert abs(difference[metric]-(indexed[first][metric]-indexed[second][metric]))<1e-12
        for relative,expected in review['sources'].items():
            relative=relative.replace('\\','/')
            if relative.startswith(('tools/','results/')):
                assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected
        print('PASS: D/E full results, paired differences and frozen screening decision.')
    for cuda_path in ('results/deployment_cuda_fp32.json','results/deployment_cuda_fp32_preserved_bn.json'):
        if not (ROOT/cuda_path).exists(): continue
        cuda=read(cuda_path)
        assert cuda['raw_aggregate_reconstruction_passed'] and cuda['actual_cuda_nodes_verified']
        assert cuda['precision']==dict(dtype='float32',pytorch_tf32=False,onnx_cuda_use_tf32=0)
        assert len(cuda['results'])==3 and {r['variant'] for r in cuda['results']}==set('ABC')
        passed=all(r['status']=='pass' for r in cuda['results'])
        assert (cuda['status']=='pass')==cuda['full500_acceptance_passed']==cuda['FP16_may_start']==passed
        assert not cuda['latency_measured']
        for relative,expected in cuda['verification']['source_hashes'].items():
            assert hashlib.sha256((ROOT/relative.replace('\\','/')).read_bytes()).hexdigest()==expected
        by_run={(r['variant'],r['seed']):r for r in rows}
        for r in cuda['results']:
            original=by_run[(r['variant'],42)]
            assert r['images']==500 and r['seed']==42 and r['selected_epoch']==original['best_epoch']
            assert r['weights_sha256']==original['best_weights_sha256']
            assert r['provider_evidence']['provider']=='CUDAExecutionProvider'
            assert set(r['provider_evidence']['node_executions'])=={'CUDAExecutionProvider'}
            assert r['provider_options']['CUDAExecutionProvider']['use_tf32']=='0'
            assert r['limits']==dict(logit_rtol=.001,logit_atol=.0001,argmax_disagreement_fraction=.001,metric_difference_pp=.05)
            assert (r['status']=='pass')==all(r['checks'].values())
            for backend in ('pytorch','onnx'):
                matrix=r['metrics'][backend]['confusion']
                assert sum(map(sum,matrix))==500*768*1024
        if cuda.get('execution',{}).get('export_variant')=='preserved_BN':
            assert cuda['execution']['disabled_optimizers']==['ConvBNFusion','ConvActivationFusion']
            for r in cuda['results']:
                assert r['disabled_optimizers']==cuda['execution']['disabled_optimizers']
                assert r['original_BN_nodes']==r['optimized_BN_nodes']>0
        print('PASS: CUDA full500 evidence scope, source identity, provider and acceptance linkage.')
    if (ROOT/'results/de_exploration_plot.json').exists():
        plot=read('results/de_exploration_plot.json')
        for field,relative in [('review_sha256','results/de_review.json'),('script_sha256','tools/plot_de_exploration.py'),
                ('png_sha256','results/de_exploration.png'),('svg_sha256','results/de_exploration.svg')]:
            assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==plot[field]
    if (ROOT/'results/fp32_numerical_diagnosis.json').exists():
        diagnosis=read('results/fp32_numerical_diagnosis.json')
        assert diagnosis['status']=='diagnosis_complete' and not diagnosis['tolerance_or_weights_changed']
        assert diagnosis['reexport']['all_three_graph_hashes_equal_probe']
        for relative,expected in diagnosis['reexport']['source_hashes'].items():
            assert hashlib.sha256((ROOT/relative.replace('\\','/')).read_bytes()).hexdigest()==expected
        guard=diagnosis['GPU_guard']
        assert guard['status']=='pass' and guard['GPU_validation_lock_rejected'] and not guard['model_or_ORT_session_started']
        assert hashlib.sha256((ROOT/'experiments/attention_gate/validate_deployment_fp32_preserved_bn.py').read_bytes()).hexdigest()==guard['validator_sha256']
        repaired=diagnosis['repaired_full500']
        if repaired['status']!='pending':
            path=ROOT/'results/deployment_cuda_fp32_preserved_bn.json'
            assert hashlib.sha256(path.read_bytes()).hexdigest()==repaired['source_sha256']
            assert repaired['passed']==read(path)['full500_acceptance_passed']
        print('PASS: failed-gate history, reproducible BN export and actual GPU lock guard.')
    if (ROOT/'results/deployment_cuda_fp16.json').exists():
        fp16=read('results/deployment_cuda_fp16.json')
        protocol=read('experiments/attention_gate/deployment_fp16_protocol.json')
        assert fp16['protocol']==protocol and fp16['aggregate_reconstruction_passed']
        assert fp16['actual_cuda_nodes_verified'] and fp16['data_manifest_matches_FP32']
        for relative,expected in fp16['verification']['source_hashes'].items():
            assert hashlib.sha256((ROOT/relative.replace('\\','/')).read_bytes()).hexdigest()==expected
        assert hashlib.sha256((ROOT/'experiments/attention_gate/deployment_fp16_protocol.json').read_bytes()).hexdigest()==fp16['protocol_sha256']
        assert len(fp16['results'])==3 and {r['variant'] for r in fp16['results']}==set('ABC')
        count=3 if fp16['mode']=='pilot' else 500
        for r in fp16['results']:
            original=next(x for x in rows if x['variant']==r['variant'] and x['seed']==42)
            assert r['weights_sha256']==original['best_weights_sha256'] and r['selected_epoch']==original['best_epoch']
            assert r['images']==count and r['limits']==protocol['acceptance_limits']
            assert (r['status']=='pass')==all(r['checks'].values())
            assert r['provider_evidence']['provider']=='CUDAExecutionProvider'
            assert set(r['provider_evidence']['node_executions'])=={'CUDAExecutionProvider'}
            for key in ('pytorch','onnx'): assert sum(map(sum,r['metrics'][key]['confusion']))==count*768*1024
        passed=all(r['status']=='pass' for r in fp16['results'])
        assert fp16['full500_acceptance_passed']==(passed and count==500)
        if not passed: assert fp16['decision']=='retain_accepted_FP32_no_FP16_search'
        print('PASS: distinct FP16 protocol, pilot scope and honest failed-candidate decision.')
    print(f'PASS: {len(hashes)} exact-byte source hashes, loader proof, 9 results and paired statistics.')


if __name__ == '__main__':
    main()
