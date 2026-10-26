"""Exercise checkpoint audits on genuine CUDA smoke outputs and corrupt copies."""
import argparse
import copy
from pathlib import Path

import common_runner as core
import common_runner_mixed as mixed
import run_mixed_pair as supervisor


def main(a):
    proof = supervisor.read(a.gpu / 'verification.json')
    assert proof['status'] == 'pass' and proof['source_hashes'] == mixed.sources()
    assert proof['gpu_fullsize_D_E'] and proof['fresh_process_full_state_resume_D_E']
    assert not a.output.exists()
    checks = []
    cps = {v: core.torch.load(a.gpu / f'{v}_continuous/checkpoints/epoch_002.pth',
        map_location='cpu', weights_only=False) for v in ('D', 'E')}
    for variant in ('D', 'E'):
        cp, reference = cps[variant], cps['E' if variant == 'D' else 'D']
        expected = mixed.settings(variant, 42, 'smoke')
        supervisor.verify_checkpoint(cp, expected, reference)
        checks.append(variant + '_genuine_two_epoch_checkpoint_accepted')
        def reject(name, change):
            bad = copy.deepcopy(cp)
            change(bad)
            try:
                supervisor.verify_checkpoint(bad, expected, reference)
            except (AssertionError, RuntimeError, KeyError, ValueError):
                checks.append(variant + '_' + name + '_rejected')
            else:
                raise AssertionError('Corruption accepted: ' + name)
        reject('wrong_variant', lambda x: x['settings'].update(variant='C'))
        reject('missing_tail', lambda x: x['records'][0].update(train_batch_sizes=[4]))
        reject('duplicate_sample', lambda x: x['records'][0]['train_order'].__setitem__(0, x['records'][0]['train_order'][1]))
        reject('wrong_confusion', lambda x: x['records'][0]['val_confusion'][0].__setitem__(0, 0))
        reject('wrong_metric', lambda x: x['records'][0].update(val_miou_global=0.0))
        reject('wrong_best', lambda x: x.update(best_miou=x['best_miou']+.1))
        reject('wrong_decay', lambda x: x['optimizer_state']['param_groups'][-1].update(weight_decay=.2))
        reject('wrong_rng', lambda x: x['rng']['generators']['sample_order'].zero_())
    try:
        mixed.require_acceptance(a.gpu / 'verification.json')
    except RuntimeError:
        checks.append('GPU_only_proof_cannot_start_formal_training')
    else:
        raise AssertionError('Unreviewed proof accepted')
    a.output.mkdir(parents=True)
    core.json_write(a.output / 'verification.json', dict(status='pass', checks=checks,
        supervisor_sha256=core.digest(supervisor.__file__), check_script=str(Path(__file__).resolve()),
        check_script_sha256=core.digest(__file__), source_hashes=mixed.sources(),
        genuine_gpu_proof_sha256=core.digest(a.gpu / 'verification.json'),
        scope='Checkpoint validation and fail-closed corruption checks; formal dataset audited after epoch1'))
    print('PASS: ' + str(len(checks)) + ' acceptance/rejection checks; no training.')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    core.torch.set_num_threads(2)
    main(p.parse_args())
