"""Real 768x1024 batch4+3 training and fresh-process exact resume check.

Seven training and three validation images; no formal accuracy claims.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
import common_runner as core
import common_runner_compact as runner
from compact_candidate import build_compact
from check_step03 import equal_tree


def main(output):
    output = output.resolve()
    assert output.is_relative_to(core.HERE) and output != core.HERE and not output.exists()
    output.mkdir(parents=True)
    core.torch.set_num_threads(2)
    pinned, began, checks, first = runner.sources(), time.perf_counter(), [], {}
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    core.configure(42)
    before_cpu=core.torch.get_rng_state()
    before_cuda=core.torch.cuda.get_rng_state_all()
    left, right = build_compact('L0'), build_compact('L1')
    assert core.torch.equal(before_cpu,core.torch.get_rng_state())
    assert equal_tree(before_cuda,core.torch.cuda.get_rng_state_all())
    assert all(core.torch.equal(t, right.state_dict()[k]) for k,t in left.state_dict().items())
    del left, right
    for variant in ('L0', 'L1'):
        for name, extra in [('continuous', []), ('first', ['--stop-after-epoch', '1']),
            ('resumed', ['--resume', str(output / f'{variant}_first/checkpoints/epoch_001.pth')])]:
            command = [sys.executable, '-u', str(core.HERE / 'common_runner_compact.py'),
                '--variant', variant, '--mode', 'smoke', '--seed', '42', '--output', str(output / f'{variant}_{name}')] + extra
            with (output / f'{variant}_{name}.log').open('x', encoding='utf-8') as log:
                subprocess.run(command, cwd=core.PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
        direct = core.torch.load(output / f'{variant}_continuous/checkpoints/epoch_002.pth', map_location='cpu', weights_only=False)
        restored = core.torch.load(output / f'{variant}_resumed/checkpoints/epoch_002.pth', map_location='cpu', weights_only=False)
        assert equal_tree(direct, restored), variant + ' full-state resume mismatch'
        expected = runner.settings(variant, 42, 'smoke')
        runner.validate_resume(direct, expected)
        assert all(r['train_images']==7 and r['val_images']==3 and r['train_batch_sizes']==[4,3] for r in direct['records'])
        initial = build_compact(variant,42).state_dict()
        assert any(not core.torch.equal(t, initial[k]) for k,t in direct['model_state'].items() if k.endswith('weight'))
        if variant == 'L1':
            assert any(not core.torch.equal(t, initial[k]) for k,t in direct['model_state'].items() if k.startswith('ca.conv'))
        assert all(core.torch.isfinite(t).all() for t in direct['model_state'].values())
        assert all(int(v['step'])==4 for v in direct['optimizer_state']['state'].values())
        # Wrong seed/variant and changed scheduler must be rejected before resuming.
        for broken in [dict(expected, seed=43), dict(expected, variant='A')]:
            try:
                runner.validate_resume(direct,broken)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Resume accepted wrong identity')
        first[variant] = core.torch.load(output / f'{variant}_first/checkpoints/epoch_001.pth',map_location='cpu',weights_only=False)
        checks.append(variant+'_real_fullsize_loss_optimizer_scheduler_rng_best_records_exact_resume')
        print('PASS '+variant+' CUDA training and exact fresh-process resume',flush=True)
    assert first['L0']['records'][0]['train_order']==first['L1']['records'][0]['train_order']
    assert equal_tree(first['L0']['rng'],first['L1']['rng'])
    assert pinned == runner.sources()
    core.json_write(output / 'verification.json',dict(status='pass',source_hashes=pinned,
        test_source_sha256=core.digest(__file__),gpu_fullsize_L0_L1=True,
        fresh_process_full_state_resume_L0_L1=True,checks=checks,
        scope='Real 7/3-image full-resolution CUDA mechanics only; not full dataset accuracy',
        wall_seconds=time.perf_counter()-began,formal_training_started=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    main(parser.parse_args().output)
