"""Reconstruct pilot/full FP16 evidence without promoting a failed configuration."""
import argparse
import datetime
import math
from pathlib import Path
import sys
import numpy as np
from export_repository_summary import ROOT, EXP, digest, dump
from export_followup_summary import read, sanitize
from export_cuda_validation import same
sys.path.insert(0, str(EXP))
from deployment_equivalence_fp16 import Agreement, PROTOCOL, LIMITS
from deployment_equivalence import provider_evidence


def main(a):
    a.models = a.models.resolve()
    folder = a.job.resolve()
    proof, plan, results = (read(folder / p) for p in ('verification.json', 'plan.json', 'results.json'))
    assert proof['mode'] == plan['mode'] == results['mode']
    assert proof['backend'] == plan['backend'] == results['backend'] == 'cuda'
    ids = [0, 1, 2] if plan['mode'] == 'pilot' else list(range(500))
    assert proof['models'] == 3 and proof['images_per_model'] == len(ids) and plan['indices'] == ids
    assert digest(PROTOCOL) == plan['protocol_sha256']
    assert plan['acceptance_limits'] == LIMITS
    manifest = read(a.models)
    assert digest(a.models) == plan['models_manifest_sha256']
    assert digest(PROTOCOL) == manifest['protocol_sha256']
    conversion = read(a.models.parent / 'verification.json')
    for path, sha in conversion['source_hashes'].items(): assert digest(Path(path)) == sha
    for path, sha in proof['source_hashes'].items(): assert digest(ROOT / path) == sha
    registry = ROOT / 'results/abc_summary.json'
    assert digest(registry) == plan['selected_results_sha256']
    frozen = {(r['variant'], r['seed']): r for r in read(registry)['results']}
    previous = read(EXP / 'step_10_2026-10-28/ABC_fp32_cuda_preserved_bn_full_001/plan.json')
    assert plan['data_manifest'] == previous['data_manifest']
    sources = [PROTOCOL, Path(__file__), folder / 'verification.json', folder / 'results.json', folder / 'plan.json', a.models]
    reports = []
    for row in results['results']:
        original = frozen[(row['variant'], row['seed'])]
        item = next(x for x in manifest['models'] if x['variant'] == row['variant'])
        assert row['seed'] == 42 and row['selected_epoch'] == original['best_epoch']
        assert digest(Path(item['weights'])) == row['weights_sha256'] == original['best_weights_sha256']
        assert digest(Path(item['onnx'])) == row['onnx_sha256'] == item['onnx_sha256']
        assert digest(Path(item['fp32_onnx'])) == row['fp32_onnx_sha256']
        path = folder / (row['name'] + '_per_image.json')
        records = read(path)
        assert [x['image_id'] for x in records] == [str(i) for i in ids]
        accumulator = Agreement(); accumulator.records = records
        for rec in records:
            assert rec['pixels'] == 768 * 1024
            assert 0 <= rec['argmax_disagreement_pixels'] <= rec['pixels']
            assert math.isfinite(rec['maximum_absolute_logit_error'])
            for backend in ('pytorch', 'onnx'):
                matrix = np.asarray(rec[backend]['confusion'], dtype=np.int64)
                assert matrix.shape == (2, 2) and (matrix >= 0).all() and matrix.sum() == rec['pixels']
                accumulator.confusion[backend] += matrix
                value = rec[backend]['boundary_iou']
                if value is not None:
                    assert math.isfinite(value) and 0 <= value <= 1
                    accumulator.boundary[backend].append(value)
        for key, value in accumulator.summary().items(): same(value, row[key])
        profiles = list(folder.glob(row['name'] + '_provider*.json'))
        assert len(profiles) == 1 and digest(profiles[0]) == row['profile_sha256']
        same(provider_evidence(read(profiles[0]), 'CUDAExecutionProvider'), row['provider_evidence'])
        assert row['provider_options']['CUDAExecutionProvider']['use_tf32'] == '0'
        assert row['original_BN_nodes'] == row['optimized_BN_nodes'] > 0
        reports.append(dict(row, graph_bytes=item['graph_bytes'], fp32_graph_bytes=item['fp32_graph_bytes'],
                            initializer_types=item['initializer_types']))
        sources.extend([path, profiles[0]])
    assert len(reports) == 3 and {r['variant'] for r in reports} == set('ABC')
    passed = all(r['status'] == 'pass' for r in reports)
    assert (proof['status'] == 'pass') == passed
    full = passed and plan['mode'] == 'full'
    report = dict(status='pass' if passed else 'fail', snapshot_date=datetime.date.today().isoformat(),
        mode=plan['mode'], images_per_model=len(ids), results=reports, precision=plan['precision'],
        protocol=read(PROTOCOL), protocol_sha256=digest(PROTOCOL), verification=proof,
        aggregate_reconstruction_passed=True, data_manifest_matches_FP32=True,
        actual_cuda_nodes_verified=True, full500_acceptance_passed=full, latency_measured=False,
        decision='retain_accepted_FP32_no_FP16_search' if not passed else 'benchmark_accepted_configuration' if full else 'run_full500',
        sources={str(p.relative_to(ROOT)): digest(p) for p in sources},
        limitation='Pilot failures reject this conversion only. No full500 FP16 accuracy estimate, FPS, retraining or universal FP16-failure claim.')
    dump(ROOT / 'results/deployment_cuda_fp16.json', sanitize(report))
    print('FP16 reconstructed:', report['status'], report['decision'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    main(parser.parse_args())
