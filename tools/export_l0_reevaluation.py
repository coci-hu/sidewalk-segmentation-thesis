"""Export small, source-bound evidence of the approved L0 seed42 recheck."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'experiments/attention_gate/step_12_2026-11-04/L0_repeats_001/L0_seed42_reevaluation'
OUTPUT=ROOT/'results/l0_seed42_reevaluation_20261104.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not OUTPUT.exists()
    evidence=json.loads((SOURCE/'verification.json').read_text(encoding='utf-8'))
    assert evidence['status']=='pass' and evidence['variant']=='L0' and evidence['seed']==42
    assert evidence['actual']['val_images']==500 and evidence['dataset_unchanged'] and evidence['model_unchanged']
    assert evidence['absolute_tolerance_fraction']==1e-6
    assert all(abs(x)<=1e-6 for x in evidence['metric_differences'].values())
    assert all(sha(ROOT/p)==h for p,h in evidence['source_hashes'].items())
    weights=Path(evidence['weights']);assert sha(weights)==evidence['weights_sha256']
    gate=SOURCE.parent/'repeat_gate.json';decision=json.loads(gate.read_text(encoding='utf-8'))
    assert decision['status']=='pass' and decision['repeat_seeds']==[43,44] and not decision['L1_repeats']
    result={k:evidence[k] for k in ('status','variant','seed','best_epoch','weights_sha256','source_hashes',
        'absolute_tolerance_fraction','manifest_sha256','actual','expected','metric_differences',
        'confusion_exact','model_unchanged','dataset_unchanged','wall_seconds')}
    result.update(weights=weights.relative_to(ROOT).as_posix(),
        raw_verification=(SOURCE/'verification.json').relative_to(ROOT).as_posix(),
        raw_verification_sha256=sha(SOURCE/'verification.json'),pre_inference_plan_sha256=sha(SOURCE/'plan.json'),
        repeat_gate=decision,repeat_gate_sha256=sha(gate),exporter_sha256=sha(Path(__file__)),
        scope='Fresh-process selected-weight reevaluation on same500 validation images; not an independent test set. Exact aggregate metrics/confusion do not assert pixelwise identical masks.')
    OUTPUT.write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print('PASS L0 seed42 full500 recheck exported; repeat43/44 gate verified.')


if __name__=='__main__':main()
