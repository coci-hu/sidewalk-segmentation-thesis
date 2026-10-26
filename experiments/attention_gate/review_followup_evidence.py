"""Verify diagnostic aggregates and bind reviewed evidence to D/E acceptance."""
import argparse
import csv
import statistics
from pathlib import Path

import common_runner as core
import common_runner_mixed as mixed
from analyze_abc_sensitivity import read


def main(a):
    assert a.output.resolve().is_relative_to(core.HERE) and not a.output.exists()
    gpu = read(a.gpu / 'verification.json')
    supervision = read(a.supervisor / 'verification.json')
    sensitivity = read(a.sensitivity / 'verification.json')
    diagnosis = read(a.diagnosis / 'final_verification.json')
    assert all(p['status'] == 'pass' for p in (gpu, supervision, sensitivity, diagnosis))
    assert gpu['source_hashes'] == supervision['source_hashes'] == mixed.sources()
    assert gpu['gpu_fullsize_D_E'] and gpu['fresh_process_full_state_resume_D_E']
    assert sensitivity['models'] == 9 and sensitivity['images_per_model'] == 500
    assert diagnosis['models'] == 6 and diagnosis['images_per_model'] == 54 and diagnosis['conditions_per_model'] == 10
    for relative, sha in diagnosis['source_hashes'].items():
        assert core.digest(core.PROJECT / relative) == sha
    rows = read(a.diagnosis / 'results.json')['results']
    inputs = [a.gpu/'verification.json', a.supervisor/'verification.json',
        a.sensitivity/'verification.json', a.sensitivity/'results.json',
        a.diagnosis/'final_verification.json', a.diagnosis/'results.json']
    for r in rows:
        name = f"{r['variant']}_seed{r['seed']}"
        path = a.diagnosis / (name+'_interventions.csv')
        with path.open(encoding='utf-8-sig', newline='') as f:
            raw = list(csv.DictReader(f))
        assert len(raw) == 540
        inputs.append(path)
        for condition, summary in r['conditions'].items():
            selected = [x for x in raw if x['condition'] == condition]
            assert len(selected) == len({x['sample_id'] for x in selected}) == 54
            matrix = [[sum(int(x[f'conf_{i}{j}']) for x in selected) for j in range(2)] for i in range(2)]
            assert matrix == summary['confusion'] and sum(map(sum,matrix)) == 54*768*1024
            tn,fp,fn,tp = matrix[0]+matrix[1]
            assert abs(100*tp/(tp+fp+fn)-summary['sidewalk_iou_pct']) < 1e-10
            boundary = [float(x['boundary_iou_image']) for x in selected if x['boundary_iou_image']]
            assert len(boundary) == summary['boundary_valid_images']
            assert abs(100*statistics.mean(boundary)-summary['boundary_iou_pct']) < 1e-10
        path = a.diagnosis / (name+'_gate_statistics.csv')
        with path.open(encoding='utf-8-sig', newline='') as f:
            raw = list(csv.DictReader(f))
        inputs.append(path)
        assert len(raw) == len({x['sample_id'] for x in raw}) == 54
        for key, summary in r['gate_statistics'].items():
            values = [float(x[key]) for x in raw if x[key]]
            assert len(values) == summary['images']
            if values:
                assert abs(statistics.mean(values)-summary['mean']) < 1e-9
    findings = [
        'B exceeds A for all three seeds under core/valid-pixel foreground IoU and 6/13/26px boundary metrics. Do not claim attention generally fails.',
        'C vs B average foreground IoU is lower; boundary ranking changes at 6px. Keep frozen 26px primary reporting and publish all widths.',
        'On 54 fixed TRAIN images, six trained B/C gates retain about 24-26% feature L2 norm; this alone does not prove harmful suppression.',
        'Foreground-region retention exceeds background in each model; near-zero gates are uncommon. Evidence does not establish selective sidewalk suppression or gate collapse.',
        'Energy-matched shuffling and uniform gates reduce foreground IoU in all six models; learned structure contributes under these interventions, subject to decoder/BN distribution shift.',
        'Inference-only mixing into trained weights also reduces IoU. It is not evidence that freshly trained D/E will improve or fail.',
        'Proceed with the already-declared bounded D/E training test as a hypothesis, not a confirmed repair. Do not change alpha, optimizer, labels, checkpoint selection or promotion rule after observing results.',
        'Both D42/E42 finish 100 epochs. Only after review of the original rule may BOTH D/E get seeds43/44; compare against A/B/C and E against D; preserve negative results.'
    ]
    a.output.mkdir(parents=True)
    core.json_write(a.output/'acceptance.json', dict(status='pass', source_hashes=mixed.sources(),
        gpu_fullsize_D_E=True, fresh_process_full_state_resume_D_E=True,
        sensitivity_and_diagnosis_reviewed=True, formal_training_ready=True, formal_training_started=False,
        diagnostic_raw_aggregates_reconstructed=True, findings=findings,
        evidence_hashes={str(p.resolve()):core.digest(p) for p in inputs},
        review_script_sha256=core.digest(__file__), data_scope='Cityscapes validation sensitivity and fixed TRAIN diagnostic subset; not independent testing'))
    print('PASS: raw diagnostic aggregates reconstructed; reviewed D/E acceptance bound to evidence.')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('gpu','supervisor','sensitivity','diagnosis','output'):
        p.add_argument('--'+name, type=Path, required=True)
    main(p.parse_args())
