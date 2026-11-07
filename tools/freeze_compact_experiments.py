"""Freeze the approved L0-only experimental scope; preserve the original snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'results/compact_experiment_freeze_20261107.json'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def entry(p,h=None):
    p=Path(p);p=(ROOT/p).resolve() if not p.is_absolute() else p.resolve()
    assert p.is_relative_to(ROOT) and p.is_file(),p
    value=sha(p)
    if h:assert value==h,p
    return dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=value)


def main(a):
    if a.check:
        snapshot=read(TARGET);assert snapshot['status']=='frozen'
        for r in snapshot['repository_files']:assert entry(r['path'])==r
        if a.local:
            for r in snapshot['local_artifacts']:assert entry(r['path'])==r
        print('PASS compact snapshot:',len(snapshot['repository_files']),'repository files;',
              len(snapshot['local_artifacts']) if a.local else 0,'local artifacts')
        return
    assert not TARGET.exists(),'Never overwrite a freeze'
    names=['l0_three_seed_review_20261107','compact_deployment_validation_20261107',
        'compact_benchmark_20261107','compact_application_20261107','compact_final_effect_20261107']
    proofs={name:read(ROOT/f'results/{name}.json') for name in names}
    assert all(v['status']=='pass' for v in proofs.values())
    accuracy=proofs[names[0]];timing=proofs[names[2]];app=proofs[names[3]]
    assert len(accuracy['results'])==3 and all(r['epochs']==100 for r in accuracy['results'])
    assert timing['sessions']==8 and timing['raw_statistics_and_provider_reconstructed']
    assert proofs[names[1]]['full500_acceptance_passed']
    assert app['verification']['GPU_lock_rejected_before_inference']
    local={}
    def keep(p,h=None):
        r=entry(p,h);local[r['path']]=r
    for proof in proofs.values():
        for field in ('sources','evidence_sha256'):
            for p,h in proof.get(field,{}).items():keep(p,h)
        for p,h in proof.get('source_hashes',{}).items():entry(p,h)
    for row in accuracy['results']:
        keep(row['best_weights'],row['best_weights_sha256'])
        keep(Path(row['best_weights']).parent.parent/'checkpoints/epoch_100.pth')
    seed42=read(ROOT/'results/compact_seed42_audit_20261103.json')
    l1=seed42['results']['L1'];assert l1['epochs']==100
    keep(l1['best_weights'],l1['best_weights_sha256']);keep(l1['final_checkpoint'],l1['final_checkpoint_sha256'])
    job=ROOT/'experiments/attention_gate/step_13_2026-11-07/A_L0_deployment_001'
    keep(job/'verification.json');keep(job/'export/models.json');keep(job/'accepted.json')
    for item in read(job/'export/models.json')['models']:
        keep(item['onnx'],item['onnx_sha256']);keep(item['weights'],item['weights_sha256'])
    for media in app['media']:keep(media['path'],media['sha256'])
    keep(app['verification']['constructed_video']['source'])
    # Independent public figures are versioned; validation photographs and media stay local.
    for p,h in proofs[names[-1]]['figures'].items():entry(p,h)
    tracked=set(git('ls-files','-z').decode('utf-8').split('\0'))-{'',TARGET.relative_to(ROOT).as_posix()}
    repository=[entry(p) for p in sorted(tracked)]
    report=dict(status='frozen',snapshot_id='compact-experiments-2026-11-07',
        parent_commit=git('rev-parse','HEAD').decode().strip(),
        scope='Approved L0-only supplement:3 L0 seeds,1 exploratory L1 seed, export correctness, local A/L0 timing and existing demo integration.',
        completed_new_100epoch_runs=4,total_completed_controlled_100epoch_runs_including_original=15,
        remaining_required_experiments=[],optional_GUI_not_required=True,
        previous_snapshot='results/experiment_freeze_20261101.json',previous_snapshot_sha256=sha(ROOT/'results/experiment_freeze_20261101.json'),
        effect=proofs[names[-1]]['changes_vs_A'],
        conclusions=proofs[names[-1]]['conclusions'],claim_limits=proofs[names[-1]]['limits'],
        next_steps=['Use frozen figures and tables in thesis methods/results/discussion.',
            'Prepare demonstration screenshots and defense slides; do not treat this as another dataset experiment.',
            'Reopen experiments only for an explicit supervisor requirement or reproducible defect; preserve this snapshot.'],
        backup_scope='Integrity manifest only. Git includes code/compact results/aggregate figures; local weights, datasets and media are not uploaded or backed up by Git.',
        repository_files=repository,local_artifacts=sorted(local.values(),key=lambda r:r['path']))
    TARGET.write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print('FROZEN:',len(repository),'repository files;',len(local),'local artifacts')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');p.add_argument('--local',action='store_true')
    main(p.parse_args())
