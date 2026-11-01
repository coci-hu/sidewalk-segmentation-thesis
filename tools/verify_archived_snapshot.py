"""Verify frozen Git objects at their original revision and unchanged local artifacts.

Unlike freeze_experiments.py --check, this permits later worktree documentation
and plan changes without rewriting any historical hash or result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REVISION = '19ee2c32a6880fa8067d9555b66bfe91ef0fb638'
SNAPSHOT = 'results/experiment_freeze_20261101.json'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--local', action='store_true')
    a = p.parse_args()
    def blob(path):
        return subprocess.check_output(['git', 'show', REVISION + ':' + path], cwd=ROOT)
    snapshot = blob(SNAPSHOT)
    assert snapshot == (ROOT/SNAPSHOT).read_bytes(), 'Historical manifest changed'
    data = json.loads(snapshot)
    for item in data['repository_files']:
        content = blob(item['path'])
        assert len(content) == item['bytes'] and hashlib.sha256(content).hexdigest() == item['sha256'], item['path']
    if a.local:
        for item in data['local_artifacts']:
            h = hashlib.sha256()
            path = (ROOT/item['path']).resolve()
            assert path.is_relative_to(ROOT)
            with path.open('rb') as f:
                for chunk in iter(lambda: f.read(1024*1024), b''):
                    h.update(chunk)
            assert path.stat().st_size == item['bytes'] and h.hexdigest() == item['sha256'], item['path']
    print('PASS: original revision', REVISION, 'repository files', len(data['repository_files']),
          'local artifacts', len(data['local_artifacts']) if a.local else 0)


if __name__ == '__main__':
    main()
