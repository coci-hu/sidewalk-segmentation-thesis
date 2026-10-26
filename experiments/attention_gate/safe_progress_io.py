"""Operational JSON I/O adapter; no tensor, RNG, optimizer or data operations.

Frozen scientific sources and checkpoint settings are unchanged. Only disposable
progress/timing snapshots may be skipped after bounded sharing-error retries.
Authoritative status writes and all serialization/disk errors remain fatal.
"""
import errno
import itertools
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OPTIONAL_SNAPSHOTS = frozenset(('training_progress.json', 'epoch_timing.json'))
_COUNTER = itertools.count()
STATS = {'writes': 0, 'sharing_retries': 0, 'skipped_optional_snapshots': 0}


def sharing_error(error):
    return isinstance(error, PermissionError) or getattr(error, 'winerror', None) in (5, 32, 33)


def atomic_json(path, value, *, attempts=12, replace=None, sleep=time.sleep):
    path = Path(path).resolve()
    if path == ROOT or not path.is_relative_to(ROOT):
        raise ValueError('Operational JSON outside attention_gate')
    # Serialize before writing, rejecting NaN and malformed values even for progress.
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{next(_COUNTER)}.tmp')
    operation = os.replace if replace is None else replace
    last_error = None
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            stream.write(payload)
        for attempt in range(attempts):
            try:
                operation(temporary, path)
                STATS['writes'] += 1
                return True
            except OSError as error:
                if not sharing_error(error):
                    raise
                last_error = error
                STATS['sharing_retries'] += 1
                if attempt + 1 < attempts:
                    sleep(min(0.05 * (attempt + 1), 0.4))
        if path.name not in OPTIONAL_SNAPSHOTS:
            raise last_error
        STATS['skipped_optional_snapshots'] += 1
        print(f'[IO] Snapshot skipped after {attempts} sharing-error attempts: {path.name}; '
              'training/checkpoint operations continue', file=sys.stderr, flush=True)
        return False
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except PermissionError:
            # Only our own telemetry temporary file; never remove experiment outputs.
            print(f'[IO] Temporary snapshot remains locked: {temporary.name}', file=sys.stderr, flush=True)


def install(fast):
    fast.atomic_json = atomic_json
