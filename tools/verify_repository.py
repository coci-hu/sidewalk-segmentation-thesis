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
    print(f'PASS: {len(hashes)} exact-byte source hashes, loader proof, 9 results and paired statistics.')


if __name__ == '__main__':
    main()
