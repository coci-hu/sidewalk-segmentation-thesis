"""Reconstruct nine formal sessions from raw samples and execution evidence."""
import argparse
import datetime
import math
from pathlib import Path
import statistics
import sys
import numpy as np
from export_repository_summary import ROOT, EXP, digest, dump
from export_followup_summary import read, sanitize
sys.path.insert(0, str(EXP))
from deployment_equivalence import provider_evidence


def close(a, b): assert math.isclose(a, b, abs_tol=1e-9, rel_tol=1e-10), (a, b)


def main(job):
    job = job.resolve()
    status, plan, result = (read(job / name) for name in ('status.json', 'plan.json', 'results.json'))
    assert status == dict(status='complete', sessions=9, checks_passed=True)
    assert plan['mode'] == result['mode'] == 'full' and result['formal_sessions'] == 9
    assert not (job / 'failure.json').exists() and not (job / 'STOP_REQUESTED').exists()
    protocol_path = EXP / 'benchmark_fp32_protocol.json'
    protocol = read(protocol_path); assert plan['protocol'] == protocol
    expected = [(i, v) for i, order in enumerate(protocol['round_orders'], 1) for v in order]
    assert [(r['round'], r['variant']) for r in result['results']] == expected
    accepted = read(ROOT / 'results/deployment_cuda_fp32_preserved_bn.json')
    sources = [Path(__file__), protocol_path, job / 'results.json', job / 'plan.json', job / 'status.json']
    pooled = {v: {'model_only': [], 'end_to_end': []} for v in 'ABC'}
    reports = []
    code_hashes = {}
    for row in result['results']:
        folder = job / f'round{row["round"]}_{row["variant"]}'
        assert read(folder / 'result.json') == row
        stage_plan = read(folder / 'plan.json')
        assert stage_plan['warmup'] == 50 and stage_plan['count'] == 500 and stage_plan['protocol'] == protocol
        assert not (folder / 'failure.json').exists()
        assert row['status'] == 'pass' and row['formal_timing'] and row['io_binding_argmax_matches_ordinary']
        for path, value in row['source_hashes'].items():
            assert digest(Path(path)) == value
            if Path(path).is_relative_to(ROOT): code_hashes[str(Path(path).relative_to(ROOT))] = value
        item = row['model_identity']
        original = next(r for r in accepted['results'] if r['variant'] == row['variant'])
        assert item['seed'] == 42 and item['selected_epoch'] == original['selected_epoch']
        assert digest(Path(item['onnx'])) == item['onnx_sha256'] == original['onnx_sha256']
        assert digest(Path(item['weights'])) == item['weights_sha256'] == original['weights_sha256']
        for p in (row['power_before'], row['power_after']):
            assert p['AC'] == 1 and p['scheme'] == 'a1841308-3541-4fab-bc81-f71556f20b4a'
        samples_path = folder / 'samples.json'
        assert digest(samples_path) == row['samples_sha256']
        samples = read(samples_path)
        for key in ('model_only', 'end_to_end'):
            values = samples[key + '_ms']
            assert len(values) == row[key]['frames'] == 500 and all(math.isfinite(v) and v > 0 for v in values)
            close(float(np.mean(values)), row[key]['mean_ms'])
            close(float(np.median(values)), row[key]['median_ms'])
            close(float(np.percentile(values, 95)), row[key]['p95_ms'])
            close(500 / row[key]['loop_wall_seconds'], row[key]['actual_fps'])
            assert row[key]['loop_wall_seconds'] >= sum(values) / 1000
            pooled[row['variant']][key].extend(values)
        profiles = list(folder.glob('provider*.json'))
        assert len(profiles) == 1 and digest(profiles[0]) == row['profile_sha256']
        assert provider_evidence(read(profiles[0]), 'CUDAExecutionProvider') == row['provider']
        for snapshot in row['memory_snapshots'].values():
            assert snapshot['status'] == 'available' and snapshot['samples']
            for metric, suffix in [('dedicated_bytes', 'dedicated usage'), ('shared_bytes', 'shared usage')]:
                actual = sum(x['bytes'] for x in snapshot['samples'] if x['path'].lower().endswith(suffix))
                close(actual, snapshot[metric])
        reports.append({k: row[k] for k in ('variant', 'round', 'model_only', 'end_to_end',
            'memory_snapshots', 'power_before', 'power_after', 'provider', 'model_identity')})
        sources.extend([folder / 'result.json', folder / 'plan.json', samples_path, profiles[0]])
    summary = {}
    for variant in 'ABC':
        rows = [r for r in reports if r['variant'] == variant]
        entry = {}
        for key in ('model_only', 'end_to_end'):
            values = pooled[variant][key]
            entry[key] = dict(frames=len(values), mean_ms=float(np.mean(values)), median_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values, 95)), actual_fps=len(values) / sum(r[key]['loop_wall_seconds'] for r in rows),
                session_mean_ms_sd=statistics.stdev(r[key]['mean_ms'] for r in rows))
        item = rows[0]['model_identity']
        entry['onnx_bytes'] = Path(item['onnx']).stat().st_size
        entry['weights_file_bytes'] = Path(item['weights']).stat().st_size
        entry['max_observed_dedicated_MiB'] = max(s['dedicated_bytes'] for r in rows for s in r['memory_snapshots'].values()) / 1024**2
        entry['max_observed_shared_MiB'] = max(s['shared_bytes'] for r in rows for s in r['memory_snapshots'].values()) / 1024**2
        summary[variant] = entry
    output = dict(status='pass', snapshot_date=datetime.date.today().isoformat(), sessions=9,
        samples_per_pipeline=4500, summary=summary, session_results=reports, protocol=protocol,
        source_hashes=code_hashes, sources={str(p.relative_to(ROOT)): digest(p) for p in sources},
        raw_statistics_and_provider_reconstructed=True, training_or_weights_changed=False,
        limitations=['Windows power-saver mode on AC; no maximum-performance claim',
          'Model-only uses one resident fixed tensor; PNG pipeline includes decoding all500 full-size images',
          'No camera/video end-to-end or embedded-device performance claim',
          'Process GPU memory snapshots across adapters, not a true peak; device totals are not process memory'])
    dump(ROOT / 'results/deployment_benchmark.json', sanitize(output))
    print(json_summary(summary))


def json_summary(summary):
    import json
    return json.dumps({v: {k: x[k] for k in ('model_only', 'end_to_end', 'max_observed_dedicated_MiB')} for v, x in summary.items()}, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    main(parser.parse_args().job)
