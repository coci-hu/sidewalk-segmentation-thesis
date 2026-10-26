"""Regression and rejection checks on real C42 checkpoints; no training."""
import argparse
import copy
from pathlib import Path
import run_c_repeats as r


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-job', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = r.ab.local(a.output)
    assert not out.exists()
    plan = r.read(a.source_job / 'job_plan.json')
    r.core.torch.set_num_threads(2)
    expected = r.c.settings(42, 'train')
    ref = r.core.torch.load(plan['B_seed42_final_checkpoint'], map_location='cpu', weights_only=False)
    checks = []
    for epoch, stage in ((1, 'stage1'), (2, 'stage2'), (100, 'stage2')):
        folder = Path(plan[stage])
        cp = r.core.torch.load(folder / f'checkpoints/epoch_{epoch:03d}.pth', map_location='cpu', weights_only=False)
        r.verify_checkpoint(cp, epoch, 42, expected, ref)
        old_cp = r.old.audit(folder, epoch, plan)
        assert r.equal_tree(cp, old_cp)
        checks.append(f'real_C42_epoch{epoch}_matches_original_audit')
    for fault in ('wrong_seed', 'wrong_variant', 'missing_sample', 'wrong_optimizer_step'):
        bad = copy.deepcopy(cp)
        if fault == 'wrong_seed':
            bad['settings']['seed'] = 43
        elif fault == 'wrong_variant':
            bad['settings']['variant'] = 'B'
        elif fault == 'missing_sample':
            bad['records'][0]['train_order'][0] = bad['records'][0]['train_order'][1]
        else:
            next(iter(bad['optimizer_state']['state'].values()))['step'].sub_(1)
        rejected = False
        try:
            r.verify_checkpoint(bad, 100, 42, expected, ref)
        except AssertionError:
            rejected = True
        assert rejected, fault
        checks.append(fault + '_rejected')
    for seed in (43, 44):
        assert r.equal_tree(r.build_c(seed).state_dict(), r.core.build_paired_model('B', seed).state_dict())
        checks.append(f'C{seed}_initial_tensors_paired')
    r.old.integrity(a.source_job, plan)
    out.mkdir(parents=True)
    r.core.json_write(out / 'verification.json', {'status': 'pass', 'checks': checks,
        'source_hashes': r.c.sources(), 'supervisor_sha256': r.core.digest(r.__file__),
        'check_script': str(Path(__file__).resolve()), 'check_script_sha256': r.core.digest(__file__),
        'training_started': False, 'GPU_training_checks': 'existing C43/44 initialization and C full-size/resume proof retained; new seeds full-data resume audited during formal runs'})
    print({'status': 'pass', 'checks': len(checks)}, flush=True)


if __name__ == '__main__':
    main()
