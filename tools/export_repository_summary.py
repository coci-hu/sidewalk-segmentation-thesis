"""Export small, traceable summaries; never modify experimental source/results."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import platform
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments/attention_gate'
SOURCE = EXP / 'step_06_2026-10-24/C_seeds43_44_training_001'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main(snapshot_date=None):
    verification = json.loads((SOURCE / 'final_verification.json').read_text(encoding='utf-8'))
    assert verification['status'] == 'pass' and verification['all_9_runs_100epochs']
    raw = json.loads((SOURCE / 'three_seed_ABC_results.json').read_text(encoding='utf-8'))
    assert {(r['variant'], r['seed']) for r in raw['results']} == {(v, s) for v in 'ABC' for s in (42, 43, 44)}
    metrics = ('best_global_miou_pct', 'best_sidewalk_iou_pct', 'best_boundary_iou_pct')
    for variant in 'ABC':
        for metric in metrics:
            values = [r[metric] for r in raw['results'] if r['variant'] == variant]
            assert abs(statistics.mean(values) - raw['summary'][variant][metric]['mean']) < 1e-10
            assert abs(statistics.stdev(values) - raw['summary'][variant][metric]['sample_std_ddof1']) < 1e-10
    for row in raw['results']:
        assert row['epochs'] == 100
        for key in ('best_weights', 'final_checkpoint'):
            path = Path(row[key])
            assert path.is_file(), path
            row[key] = path.relative_to(ROOT).as_posix()
        row['best_weights_sha256'] = digest(ROOT / row['best_weights'])
        assert digest(ROOT / row['final_checkpoint']) == row['final_checkpoint_sha256']
    raw['source'] = {
        'path': (SOURCE / 'three_seed_ABC_results.json').relative_to(ROOT).as_posix(),
        'sha256': digest(SOURCE / 'three_seed_ABC_results.json'),
        'verification_sha256': digest(SOURCE / 'final_verification.json'),
        'local_only_full_artifacts': True,
    }
    raw['snapshot_date'] = snapshot_date or datetime.date.today().isoformat()
    sensitivity = EXP / 'step_06_2026-10-26/ABC_sensitivity_001/final_verification.json'
    training = EXP / 'step_07_2026-10-26/DE_seed42_training_001'
    raw['status'] = {'ABC_training_complete': True,
                     'full_sensitivity_complete': sensitivity.exists() and json.loads(sensitivity.read_text(encoding='utf-8')).get('status') == 'pass',
                     'DE_formal_training_started': (training / 'D_seed42_epoch001/training_progress.json').exists(),
                     'target_edge_device_verified': False}
    dest = ROOT / 'results'
    dest.mkdir(exist_ok=True)
    dump(dest / 'abc_summary.json', raw)
    packages = {}
    for name in ('torch', 'torchvision', 'numpy', 'Pillow', 'opencv-python', 'PyYAML', 'tqdm', 'matplotlib', 'onnx', 'onnxruntime-gpu'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    dump(dest / 'environment.json', {'python': platform.python_version(), 'system': platform.system(),
        'packages': packages, 'fresh_install_validated': False, 'scope': 'Observed local environment; not a dependency lockfile'})
    frozen = json.loads((EXP / 'step_07_2026-10-24/runner_cpu_checks_002/verification.json').read_text(encoding='utf-8'))
    assert frozen['status'] == 'pass'
    hashes = {k.replace('\\', '/'): v for k, v in frozen['source_hashes'].items()}
    for path, expected in hashes.items():
        assert digest(ROOT / path) == expected, path
    hashes['experiments/attention_gate/check_loader_optimization.py'] = digest(EXP / 'check_loader_optimization.py')
    dump(dest / 'frozen_source_hashes.json', hashes)
    print('Export passed: 9 runs; final/best weights hashed; frozen sources unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-date', type=lambda s: datetime.date.fromisoformat(s).isoformat())
    main(parser.parse_args().snapshot_date)
