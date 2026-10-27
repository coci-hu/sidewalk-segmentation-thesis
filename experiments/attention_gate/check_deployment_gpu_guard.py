"""During active training, ensure the CUDA validation CLI refuses the occupied lock."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main(a):
    here=Path(__file__).resolve().parent
    assert a.output.resolve().is_relative_to(here) and not a.output.exists()
    runner=here/'validate_deployment_fp32.py'
    a.output.mkdir(parents=True)
    forbidden=a.output/'must_not_start'
    command=[sys.executable,str(runner),'--models',str(a.models),'--mode','pilot','--backend','cuda','--output',str(forbidden)]
    r=subprocess.run(command,capture_output=True,timeout=120,
        env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8'))
    (a.output/'probe.stderr.log').write_bytes(r.stderr)
    diagnostic=r.stderr.decode('utf-8',errors='replace')
    assert r.returncode!=0 and 'msvcrt.locking' in diagnostic and '[Errno 13]' in diagnostic, 'Expected the active training lock to reject validation'
    assert not forbidden.exists(), 'GPU validation output unexpectedly started'
    result=dict(status='pass',occupied_training_lock_rejected_before_model_or_ORT_session=True,
        validation_output_not_created=True,training_stopped=False,GPU_inference_executed=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(runner.read_bytes()).hexdigest())
    (a.output/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('PASS: actual CUDA validation CLI refused the active training lock before inference.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
