"""Publish compact application acceptance evidence without media or local personal paths."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'experiments/attention_gate/step_14_2026-11-07/desktop_acceptance_002'
OUTPUT=ROOT/'results/desktop_application_20261107.json'
sys.path.insert(0,str(ROOT/'tools'))
from export_followup_summary import sanitize

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert not OUTPUT.exists()
    proof=read(SOURCE/'verification.json');assert proof['status']=='pass'
    for k in ('actual_Tk_UI_start_and_cancel_tested','UI_recovers_after_cancel','missing_input_handled',
              'GPU_lock_refused','existing_result_overwrite_refused','original_input_unchanged'):assert proof[k]
    assert proof['full_constructed_video_frames']==32
    for p,h in proof['source_hashes'].items():assert sha(ROOT/p)==h
    assert sha(ROOT/proof['screenshot'])==proof['screenshot_sha256']
    artifacts={str((SOURCE/'verification.json').relative_to(ROOT)):sha(SOURCE/'verification.json')}
    for folder in proof['media_jobs']:
        job=ROOT/folder;report=read(job/'result.json');assert report['state']=='complete'
        artifacts[str((job/'result.json').relative_to(ROOT))]=sha(job/'result.json')
        for key in ('output','mask'):
            if report.get(key):
                p=Path(report[key]);assert p.is_relative_to(ROOT);artifacts[str(p.relative_to(ROOT))]=sha(p)
    report=dict(status='pass',application='Windows local sidewalk segmentation desktop application',
        evidence=proof,local_artifacts_sha256=artifacts,
        entry='applications/sidewalk_desktop/启动人行道识别.cmd',
        experimental_snapshot_commit='39f89ffaaaea77832ec6a5a945e5f2329a2223fa',
        screenshot_review='Window-owned GDI screenshot visually inspected; direct screen capture unavailable in this session.',
        preliminary_attempt='desktop_acceptance_001 stopped at screen-capture failure after successful image inference; retained;002 uses GDI fallback and completes full suite.',
        scope='Configured existing Windows machine only; not a standalone EXE installer, new dataset or long-duration stability experiment.',
        scientific_models_unchanged=True,software_cost_cny=0,exporter_sha256=sha(Path(__file__)))
    OUTPUT.write_bytes((json.dumps(sanitize(report),ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print('PASS desktop release: UI actions, both image models, video, cancel, invalid input, busy lock, no-overwrite; scientific sources unchanged.')

if __name__=='__main__':main()
