"""Create or verify an immutable experimental snapshot; never train or overwrite one."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'results/experiment_freeze_20261101.json'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT)
def checked(p):
    p=Path(p); p=p if p.is_absolute() else ROOT/p
    p=p.resolve(); assert p.is_relative_to(ROOT) and p.is_file(),p
    return p
def entry(p,expected=None):
    p=checked(p); value=sha(p)
    if expected: assert value==expected,p
    return dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=value)

def main(a):
    if a.check:
        snapshot=read(TARGET)
        for x in snapshot['repository_files']: assert entry(x['path'])==x
        if a.local:
            for x in snapshot['local_artifacts']: assert entry(x['path'])==x
        assert snapshot['status']=='frozen' and snapshot['remaining_required_experiments']==[]
        print(f"PASS: frozen repository files={len(snapshot['repository_files'])}; local artifacts checked={len(snapshot['local_artifacts']) if a.local else 0}")
        return
    assert not TARGET.exists(),'Frozen snapshots must not be overwritten'
    subprocess.run([sys.executable,str(ROOT/'tools/verify_repository.py')],cwd=ROOT,check=True)
    abc=read(ROOT/'results/abc_summary.json'); de=read(ROOT/'results/de_review.json')
    timing=read(ROOT/'results/deployment_benchmark.json'); stability=read(ROOT/'results/deployment_stability.json')
    cuda=read(ROOT/'results/deployment_cuda_fp32_preserved_bn.json')
    candidate=read(ROOT/'results/pipeline_comparison_review.json'); application=read(ROOT/'results/reference_application.json')
    assert stability['status']=='pass' and stability['full30_minutes'] and stability['exact_repeat_masks']
    assert cuda['full500_acceptance_passed'] and timing['sessions']==9 and application['status']=='pass'
    assert de['decision']=='stop_DE_branch_and_continue_baseline_deployment' and not candidate['candidate_accepted']
    runs=abc['results']+[r for r in de['results'] if r['variant'] in 'DE']
    assert len(runs)==11 and all(r['epochs']==100 for r in runs)
    local={}
    def keep(path,expected=None):
        x=entry(path,expected); local[x['path']]=x
    for r in runs:
        keep(r['best_weights'],r.get('best_weights_sha256'))
        keep(r['final_checkpoint'],r['final_checkpoint_sha256'])
    manifest=ROOT/'experiments/attention_gate/step_10_2026-10-28/preserved_bn_reexport_cpu_001/models.json'
    keep(manifest)
    for r in read(manifest)['models']: keep(r['onnx'],r['onnx_sha256'])
    evidence=['abc_summary','sensitivity_summary','bc_diagnosis_summary','de_review','de_checkpoint_audit',
        'deployment_cuda_fp32_preserved_bn','deployment_cuda_fp16','deployment_benchmark','pipeline_comparison_review',
        'reference_application','deployment_stability']
    for name in evidence:
        proof=read(ROOT/f'results/{name}.json')
        for p,h in proof.get('sources',{}).items(): keep(p,h)
    case=application['case_selection']; keep(case['figure'],case['figure_sha256'])
    for media in application['reference_demo']['media']: keep(media['path'],media['sha256'])
    tracked=git('ls-files','-z').decode('utf-8').split('\0')
    repository=[entry(p) for p in sorted(set(tracked)-{'',TARGET.relative_to(ROOT).as_posix()})]
    # Local raw artifacts and binary models are integrity-checked, not uploaded.
    snapshot=dict(status='frozen',snapshot_id='experiment-freeze-2026-11-01',frozen_on='2026-11-01',
        parent_commit=git('rev-parse','HEAD').decode().strip(),scope='Cityscapes sidewalk binary segmentation and local RTX4060 Laptop deployment validation',
        completed_controlled_100epoch_runs=11,remaining_required_experiments=[],
        main_accuracy_three_seed_summary=abc['summary'],deployment_seed42_timing=timing['summary'],
        stability=dict(seconds=stability['duration_seconds'],frames=stability['frames'],repeated_images=8,changed_frames=0),
        branch_decisions=dict(DE='closed_negative_screen',FP16='closed_failed_pilot',pipeline='closed_failed_exact_output_gate'),
        deployment_choice='B seed42 epoch84; preserved-BN FP32 CUDA; reference preprocessing and postprocessing',
        next_plan=[dict(dates='2026-11-01',task='Complete verification, freeze and synchronize existing evidence',state='complete_in_this_snapshot'),
            dict(dates='2026-11-02 to 2026-11-03',task='User reviews result table, selected examples and demo; arrange findings into thesis chapters',estimated_hours='2-4',new_experiment=False),
            dict(dates='2026-11-04 to 2026-11-06',task='Buffer only for concrete supervisor-required corrections or reproducible defects',estimated_days='1-2 if triggered',new_experiment=False)],
        reopening_rule='Only a reproducible defect or explicit supervisor requirement reopens experimental scope; create a new snapshot and preserve this one',
        excluded=['new datasets or self-collected annotation','new networks, training seeds or precision searches','new hardware or rental'],
        claim_limits=['Validation set reused for checkpoint selection; not independent test',
          'Laptop GPU verification, not physical embedded-board validation','Semantic sidewalk, not proven safety or path planning',
          'Model-only FPS differs from complete PNG throughput','Repeated-still stability, not natural-video temporal accuracy',
          'Negative branches are completed investigations, not accepted improvements'],
        backup_scope='Git stores code and compact evidence only; local binaries/data need their existing local storage. This is an integrity snapshot, not a full data backup.',
        repository_files=repository,local_artifacts=sorted(local.values(),key=lambda x:x['path']))
    TARGET.write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Frozen {len(repository)} repository files and {len(local)} local artifacts;11 completed runs; no remaining required experiment.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--check',action='store_true')
    p.add_argument('--local',action='store_true'); main(p.parse_args())
