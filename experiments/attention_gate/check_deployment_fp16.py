"""Reject false precision acceptance without changing the established FP32 rules."""
import argparse
import json
from pathlib import Path
import numpy as np
import deployment_equivalence as fp32
import deployment_equivalence_fp16 as fp16


def main(out):
    assert not out.exists()
    before = dict(fp32.LIMITS)
    target = np.array([[0, 1], [1, 0]], dtype=np.int64)
    reference = np.stack([target == 0, target == 1])[None].astype(np.float32)
    checks = []
    for amount, expected32, expected16 in ((0., 'pass', 'pass'), (.005, 'fail', 'pass'), (.1, 'fail', 'fail')):
        old, new = fp32.Agreement(), fp16.Agreement()
        old.add(reference, reference + amount, target, '0')
        new.add(reference, reference + amount, target, '0')
        assert old.summary()['status'] == expected32 and new.summary()['status'] == expected16
    checks.append('precision_specific_logit_rules_do_not_change_FP32')
    bad = fp16.Agreement()
    bad.add(reference, reference[:, ::-1].copy(), target, '0')
    assert bad.summary()['status'] == 'fail'
    assert not bad.summary()['checks']['regional_metrics']
    checks.append('flipped_semantics_rejected')
    for value in (np.nan, np.inf):
        broken = reference.copy(); broken[0, 0, 0, 0] = value
        try: fp16.Agreement().add(reference, broken, target, '0')
        except ValueError: pass
        else: raise AssertionError('Nonfinite output accepted')
    checks.append('nonfinite_outputs_rejected')
    zero = np.zeros((2, 2), dtype=np.int64)
    empty = np.stack([zero == 0, zero == 1])[None].astype(np.float32)
    paired = fp16.Agreement()
    paired.add(empty, empty, zero, 'empty')
    paired.add(reference, reference, target, 'mixed')
    assert paired.summary()['metrics']['onnx']['boundary_excluded_images'] == 1
    assert paired.summary()['metrics']['onnx']['sidewalk_iou_pct'] == 100.
    checks.append('empty_boundary_exclusion_and_global_metrics')
    try: paired.add(reference, reference, target, 'mixed')
    except ValueError: pass
    else: raise AssertionError('Duplicate image accepted')
    checks.append('duplicate_image_rejected')
    assert fp32.LIMITS == before
    out.mkdir(parents=True)
    (out / 'verification.json').write_text(json.dumps(dict(status='pass', checks=checks,
        FP32_limits_unchanged=True, actual_model_validation=False), indent=2) + '\n', encoding='utf-8')
    print('PASS:', len(checks), 'meaningful precision acceptance checks')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
