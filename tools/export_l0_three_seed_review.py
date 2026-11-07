"""Verify completed L0 repeats and export all paired results without reselection."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / 'experiments/attention_gate/step_12_2026-11-04/L0_repeats_001'
OUTPUT = ROOT / 'results/l0_three_seed_review_20261107.json'
METRICS = {'miou': 'val_miou_global', 'sidewalk': 'val_iou_sidewalk_global',
           'boundary': 'val_boundary_iou_author_fg_mean'}
ABC_KEYS = {'miou': 'best_global_miou_pct', 'sidewalk': 'best_sidewalk_iou_pct',
            'boundary': 'best_boundary_iou_pct'}


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(v): return dict(mean_pct=statistics.mean(v), sample_std_pct=statistics.stdev(v))


def main():
    assert not OUTPUT.exists(), 'Never overwrite published results'
    job = read(JOB / 'job_status.json')
    assert job['status'] == 'complete'
    for p, h in job['source_hashes'].items(): assert sha(ROOT / p) == h
    original = read(JOB / 'three_seed_summary.json')
    abc_path = ROOT / 'results/abc_summary.json'
    abc = read(abc_path)
    results = []
    evidence = {str(abc_path.relative_to(ROOT)): sha(abc_path)}
    for seed in (42, 43, 44):
        proof_path = JOB / f'L0_seed{seed}_reevaluation/verification.json'
        v = read(proof_path)
        assert v == original['selected_reevaluations'][str(seed)]
        assert v['status'] == 'pass' and v['actual']['val_images'] == 500
        assert v['confusion_exact'] and v['model_unchanged'] and v['dataset_unchanged']
        assert all(abs(x) <= 1e-6 for x in v['metric_differences'].values())
        assert sha(Path(v['weights'])) == v['weights_sha256']
        run = Path(v['source_run'])
        status = read(run / 'status.json')
        assert status['status'] == 'complete' and status['completed_epoch'] == 100
        if seed != 42:
            audit_path = run / 'completion_audit.json'
            audit = read(audit_path)
            assert audit['status'] == 'pass' and audit['all100_epochs_verified']
            assert audit['paired_A_sample_orders_verified']
            assert sha(run / 'checkpoints/epoch_100.pth') == audit['final_checkpoint_sha256']
            evidence[str(audit_path.relative_to(ROOT))] = sha(audit_path)
        values = {k: 100 * v['actual'][m] for k, m in METRICS.items()}
        row = dict(variant='L0', seed=seed, epochs=100, best_epoch=v['best_epoch'],
                   best_weights=Path(v['weights']).relative_to(ROOT).as_posix(),
                   best_weights_sha256=v['weights_sha256'], metrics_pct=values,
                   manifest_sha256=v['manifest_sha256'], reevaluation_passed=True,
                   paired_changes_pp={})
        for variant in ('A', 'B'):
            base = next(r for r in abc['results'] if r['variant'] == variant and r['seed'] == seed)
            row['paired_changes_pp'][variant] = {k: values[k] - base[ABC_KEYS[k]] for k in METRICS}
        for k in METRICS: row[ABC_KEYS[k]] = values[k]
        results.append(row)
        evidence[str(proof_path.relative_to(ROOT))] = sha(proof_path)
    assert len({r['manifest_sha256'] for r in results}) == 1
    summary = {'L0': {k: stats([r['metrics_pct'][k] for r in results]) for k in METRICS}}
    for k, m in METRICS.items():
        for stat, value in summary['L0'][k].items(): assert abs(value-original['summary'][m][stat]) < 1e-10
    for variant in ('A', 'B'):
        rows = [r for r in abc['results'] if r['variant'] == variant]
        assert sorted(r['seed'] for r in rows) == [42, 43, 44]
        summary[variant] = {k: stats([r[ABC_KEYS[k]] for r in rows]) for k in METRICS}
    differences = {v: {k: summary['L0'][k]['mean_pct'] - summary[v][k]['mean_pct'] for k in METRICS}
                   for v in ('A', 'B')}
    out = dict(status='pass', results=results, summary=summary, mean_changes_pp=differences,
        precision_loss_target_pp=1.0,
        descriptive_mean_within_target_vs_A={k: differences['A'][k] >= -1 for k in ('sidewalk','boundary')},
        each_seed_within_target_vs_A={str(r['seed']): {k: r['paired_changes_pp']['A'][k] >= -1
            for k in ('sidewalk','boundary')} for r in results},
        interpretation='All seeds retained. Mean precision loss is within the working 1pp target, '
            'but individual seeds fail and variability increases. This is not an every-seed guarantee, '
            'statistical significance, independent test evaluation, or accuracy improvement. '
            'The original seed42 screening gate is preserved; mean comparison is descriptive.',
        L1_scope='Seed42 only, no repeat or stable attention-effect claim.',
        deployment_seed=42, deployment_selection='Fixed before repeats, not the best seed chosen afterwards.',
        evidence_sha256=evidence, source_hashes=job['source_hashes'], exporter_sha256=sha(Path(__file__)))
    OUTPUT.write_bytes((json.dumps(out, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))
    print(json.dumps(dict(summary=summary,mean_changes_pp=differences,per_seed=out['each_seed_within_target_vs_A']),indent=2))


if __name__ == '__main__': main()
