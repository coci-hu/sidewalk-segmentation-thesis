"""A/L0 export -> pilot -> full500 acceptance -> smoke -> balanced timing.

No training, automatic retry, threshold change, or historical overwrite.
"""
import argparse
import ctypes
import datetime
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import traceback
import safe_progress_io as io

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v): io.atomic_json(p,v)


def main(a):
    out=a.output.resolve()
    assert out.is_relative_to(HERE) and not out.exists()
    registry=ROOT/'results/compact_deployment_selection_20261107.json'
    review=read(ROOT/'results/l0_three_seed_review_20261107.json')
    assert review['status']=='pass' and review['deployment_seed']==42
    # Numeric target applies to descriptive means; all failed individual seeds remain reported.
    assert all(review['descriptive_mean_within_target_vs_A'].values())
    source_paths=[Path(__file__),registry,HERE/'export_compact_fp32.py',HERE/'validate_compact_fp32.py',
        HERE/'benchmark_compact_fp32.py',HERE/'benchmark_compact_protocol.json',
        HERE/'deployment_equivalence.py',ROOT/'results/l0_three_seed_review_20261107.json']
    source_paths += [ROOT/p for p in review['source_hashes']]
    sources={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    with (HERE/'compact_deployment_queue.lock').open('a+b') as lock:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            out.mkdir(parents=True)
            dump(out/'plan.json',dict(started_at=datetime.datetime.now().astimezone().isoformat(),
                pid=os.getpid(),source_hashes=sources,scope='A/L0 seed42 local FP32 only',
                source_review=review['mean_changes_pp'],stop_marker=str(out/'STOP_REQUESTED'),
                no_training=True,no_automatic_retry=True,old_evidence_preserved=True))
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)

            def verify():
                if (out/'STOP_REQUESTED').exists():raise InterruptedError('Stop requested between stages')
                for p,h in sources.items():assert sha(ROOT/p)==h,p

            def stage(name,script,args,proof='verification.json'):
                verify();folder=out/name
                command=[sys.executable,str(HERE/script),*map(str,args),'--output',str(folder)]
                with (out/(name+'.stdout.log')).open('wb') as stdout,(out/(name+'.stderr.log')).open('wb') as stderr:
                    child=subprocess.Popen(command,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
                    dump(out/'status.json',dict(status='running',phase=name,pid=os.getpid(),
                        child_pid=child.pid,active_output=str(folder),started_at=datetime.datetime.now().astimezone().isoformat()))
                    code=child.wait()
                if code:raise RuntimeError(f'{name} exited {code}; evidence retained, no retry')
                result=read(folder/proof)
                assert result['status']=='pass',name
                verify();print(name,'passed',flush=True)
                return result

            stage('export','export_compact_fp32.py',['--selected-results',registry,'--variants','A','L0','--preserve-bn'])
            models=out/'export/models.json'
            common=['--models',models,'--selected-results',registry,'--backend','cuda']
            stage('pilot','validate_compact_fp32.py',[*common,'--mode','pilot'])
            full=stage('full','validate_compact_fp32.py',[*common,'--mode','full'])
            assert full['full500_metrics_verified'] and full['GPU_verified']
            reports=read(out/'full/results.json')['results']
            assert {r['variant'] for r in reports}=={'A','L0'} and all(r['status']=='pass' for r in reports)
            accepted=out/'accepted.json'
            dump(accepted,dict(status='pass',full500_acceptance_passed=True,results=reports,
                data_manifest=read(out/'full/plan.json')['data_manifest'],
                proof_sha256=sha(out/'full/verification.json'),results_sha256=sha(out/'full/results.json')))
            for mode in ('smoke','full'):
                stage('timing_'+mode,'benchmark_compact_fp32.py',
                    ['--models',models,'--accepted',accepted,'--mode',mode],proof='results.json')
            verify()
            measured=read(out/'timing_full/results.json')
            assert measured['formal_sessions']==8
            dump(out/'verification.json',dict(status='pass',export_and_full500_acceptance=True,
                timing_sessions=8,source_hashes=sources,training_modified=False,
                next='Independently recompute timing samples; integrate existing demo; archive results.'))
            dump(out/'status.json',dict(status='complete',pid=os.getpid(),phase='await_independent_timing_review',
                completed_at=datetime.datetime.now().astimezone().isoformat()))
        except Exception:
            if out.exists():
                dump(out/'failure.json',dict(status='failed',traceback=traceback.format_exc(),automatic_retry=False))
                dump(out/'status.json',dict(status='stopped',pid=os.getpid(),reason='See failure.json; no automatic retry'))
            raise
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
