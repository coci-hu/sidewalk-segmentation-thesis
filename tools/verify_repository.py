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
    print(f'PASS: {len(hashes)} exact-byte source hashes, loader proof, 9 results and paired statistics.')


if __name__ == '__main__':
    main()
