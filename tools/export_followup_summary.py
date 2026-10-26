"""Export verified, compact follow-up evidence without datasets or checkpoints."""
import argparse
import datetime
import json
from pathlib import Path
from export_repository_summary import ROOT, EXP, digest, dump


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sanitize(value):
    if isinstance(value, dict):
        return {str(sanitize(k)):sanitize(v) for k,v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        normalized=value.replace('\\','/')
        prefix=ROOT.as_posix()+'/'
        if normalized.startswith(prefix):
            return normalized[len(prefix):]
    return value


def main(snapshot_date=None):
    definitions=[('sensitivity_summary','step_06_2026-10-26/sensitivity_analysis_001','verification.json'),
        ('bc_diagnosis_summary','step_07_2026-10-26/BC_diagnosis_001','final_verification.json')]
    for name, directory, proofname in definitions:
        folder=EXP/directory
        proof=read(folder/proofname)
        assert proof['status']=='pass'
        data=read(folder/'results.json')
        data['source']=dict(results_path=(folder/'results.json').relative_to(ROOT).as_posix(),
            results_sha256=digest(folder/'results.json'), verification_sha256=digest(folder/proofname),
            full_raw_artifacts_local_only=True)
        data['verification']=proof
        dump(ROOT/'results'/f'{name}.json',sanitize(data))
    acceptance=EXP/'step_07_2026-10-26/evidence_review_001/acceptance.json'
    if acceptance.exists():
        proof=read(acceptance)
        assert proof['status']=='pass' and proof['sensitivity_and_diagnosis_reviewed']
        dump(ROOT/'results/de_acceptance.json',sanitize(proof))
    folder=EXP/'step_07_2026-10-26/DE_seed42_training_001'
    snapshot=dict(snapshot_date=snapshot_date or datetime.date.today().isoformat(), status='not_started', results_available=False,
        planned=['D42: fixed alpha=.1,100 epochs','E42: learned alpha from .1,100 epochs'],
        automatic_promotion=False, scope='Progress snapshot, not live telemetry or final accuracy')
    if (folder/'job_status.json').exists():
        actual=read(folder/'job_status.json')
        snapshot.update(status=actual['status'],variant=actual.get('variant'),stage=actual.get('stage'),
            completed_runs=actual.get('completed_runs',0),saved_completed_epoch=actual.get('saved_completed_epoch',0),
            verified_first_epochs={v:(folder/f'{v}_first_epoch_verification.json').exists() for v in 'DE'},
            verified_resumed_second_epochs={v:(folder/f'{v}_resume_verification.json').exists() for v in 'DE'},
            training=actual.get('training',{}))
    if (folder/'final_verification.json').exists():
        assert read(folder/'final_verification.json')['status']=='pass'
        final=read(folder/'seed42_ABCDE_results.json')
        assert {r['variant'] for r in final['results']}==set('ABCDE')
        final['source']=dict(path=(folder/'seed42_ABCDE_results.json').relative_to(ROOT).as_posix(),
            sha256=digest(folder/'seed42_ABCDE_results.json'),verification_sha256=digest(folder/'final_verification.json'))
        dump(ROOT/'results/de_seed42_summary.json',sanitize(final))
        snapshot['results_available']=True
    dump(ROOT/'results/de_progress_snapshot.json',snapshot)
    print('Exported verified sensitivity, diagnosis and actual D/E progress snapshot.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-date',type=lambda s:datetime.date.fromisoformat(s).isoformat())
    main(parser.parse_args().snapshot_date)
