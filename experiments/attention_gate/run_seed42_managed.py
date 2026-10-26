"""One-off system-managed host; unchanged A/B loop, single Word journal.

Run this with pythonw from the current user's on-demand scheduled task.
No boot trigger, recurring trigger, new seeds, or algorithm changes.
"""
import argparse
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parent
JOURNAL = Path(r'C:\Users\13699\Desktop\毕设\工作记录.docx')
WRITER = ROOT / 'update_work_record.py'
DOC_PYTHON = Path(r'C:\Users\13699\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--migration', required=True, type=Path)
    parser.add_argument('--resume-a', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output == ROOT or output.exists():
        raise ValueError('Use a new output directory under attention_gate')
    stdout_path = Path(str(output) + '.stdout.log')
    stderr_path = Path(str(output) + '.stderr.log')
    with stdout_path.open('x', encoding='utf-8', buffering=1) as out, stderr_path.open('x', encoding='utf-8', buffering=1) as err:
        sys.stdout, sys.stderr = out, err
        import common_runner as core
        import common_runner_v2 as fast
        import run_seed42_training as original
        import verify_seed42_training as verification
        original_finish = verification.finish
        def finish_without_extra_document(job):
            return original_finish(job, write_markdown=False)
        verification.finish = finish_without_extra_document
        lock_path = ROOT / 'seed42_training.lock'
        with lock_path.open('a+b') as lock:
            lock.seek(0)
            if not lock.read(1):
                lock.write(b'1')
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                with fast.keep_awake():
                    try:
                        original.main()
                    except BaseException:
                        traceback.print_exc()
                        raise
                    state_file = output / 'job_status.json'
                    state = json.loads(state_file.read_text(encoding='utf-8'))
                    if state.get('status') == 'complete' and state.get('verified'):
                        process = subprocess.run([str(DOC_PYTHON), str(WRITER), '--mode', 'completion', '--job', str(output)],
                            cwd=str(ROOT), stdout=out, stderr=err, check=False)
                        state['report'] = str(JOURNAL)
                        state['work_record_updated'] = process.returncode == 0
                        state['work_record_update_pending'] = process.returncode != 0
                        fast.atomic_json(state_file, state)
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    main()
