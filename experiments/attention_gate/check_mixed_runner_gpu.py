"""D/E full-resolution CUDA acceptance, including exact fresh-process recovery.

This is a mechanics check on 7 train / 3 val images, not a formal result.
It does not unlock formal training without a separate diagnosis review.
"""
import argparse
import gc
import json
import msvcrt
import os
import subprocess
import sys
import time
from pathlib import Path

import common_runner as core
import common_runner_v2 as fast
import common_runner_mixed as mixed
from mixed_ca_candidate import build_candidate
from check_step03 import equal_tree


def main(a):
    output = a.output.resolve()
    assert output.is_relative_to(core.HERE) and output != core.HERE and not output.exists()
    assert core.torch.cuda.is_available()
    output.mkdir(parents=True)
    core.torch.set_num_threads(2)
    pinned, checks, began = mixed.sources(), [], time.perf_counter()
    core.json_write(output / 'plan.json', dict(source_hashes=pinned,
        test_source_sha256=core.digest(__file__), mode='full-size CUDA mechanics only',
        train_images=7, val_images=3, epochs=2, seed=42, training_started=False))
    # Only this synthetic comparison owns the lock here; child runners own it later.
    with (core.HERE / 'seed42_training.lock').open('a+b') as lock:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            core.configure(42)
            image = core.torch.randn(4, 3, 768, 1024, device='cuda')
            target = core.torch.zeros(4, 768, 1024, dtype=core.torch.long, device='cuda')
            cuda_rng = core.torch.cuda.get_rng_state_all()
            reference = None
            for variant in ('D', 'E'):
                model = build_candidate(variant, 42).cuda().train()
                core.torch.cuda.set_rng_state_all(cuda_rng)
                logits = model(image)
                loss = core.combined_loss(logits, target, fast.protocol())
                loss.backward()
                gradients = {k: p.grad.detach().cpu().clone() for k, p in model.named_parameters() if k != 'ca.beta'}
                assert core.torch.isfinite(logits).all() and all(core.torch.isfinite(v).all() for v in gradients.values())
                state = dict(logits=logits.detach().cpu(), loss=loss.item(), gradients=gradients,
                    shared_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k not in ('ca.beta', 'ca.fixed_alpha')})
                if reference is None:
                    reference = state
                else:
                    assert equal_tree(reference, state), 'D/E initial common forward/backward differs'
                    assert core.torch.isfinite(model.ca.beta.grad).all() and model.ca.beta.grad.abs().item() > 0
                del model, logits, loss, gradients
            del state, reference, image, target
            gc.collect(); core.torch.cuda.empty_cache()
        finally:
            lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
    checks.append('D_E_fullsize_batch4_initial_logits_loss_common_gradients_BN_state_exact_beta_gradient_nonzero')
    print('PASS full-size paired forward/backward', flush=True)
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    first_runs = {}
    for variant in ('D', 'E'):
        for name, extra in [('continuous', []), ('first', ['--stop-after-epoch', '1']),
                ('resumed', ['--resume', str(output / f'{variant}_first/checkpoints/epoch_001.pth')])]:
            command = [sys.executable, '-u', str(core.HERE / 'common_runner_mixed.py'), '--variant', variant,
                '--seed', '42', '--mode', 'smoke', '--output', str(output / f'{variant}_{name}')] + extra
            with (output / f'{variant}_{name}.log').open('x', encoding='utf-8') as log:
                subprocess.run(command, cwd=core.PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT,
                    check=True, timeout=900)
        direct = core.torch.load(output / f'{variant}_continuous/checkpoints/epoch_002.pth', map_location='cpu', weights_only=False)
        restored = core.torch.load(output / f'{variant}_resumed/checkpoints/epoch_002.pth', map_location='cpu', weights_only=False)
        assert equal_tree(direct, restored), variant + ' full state differs after fresh-process resume'
        mixed.validate_resume(direct, mixed.settings(variant, 42, 'smoke'))
        assert all(r['train_images'] == 7 and r['val_images'] == 3 and r['train_batch_sizes'] == [4, 3] for r in direct['records'])
        assert all(int(v['step']) == 4 for v in direct['optimizer_state']['state'].values())
        initial = build_candidate(variant, 42).state_dict()
        assert any(not core.torch.equal(v, initial[k]) for k, v in direct['model_state'].items() if k.startswith('ca.conv'))
        if variant == 'E':
            assert direct['records'][0]['alpha_before'] != direct['records'][-1]['alpha_after']
            assert direct['optimizer_state']['param_groups'][1]['weight_decay'] == 0
        else:
            assert all(r['alpha_before'] == r['alpha_after'] for r in direct['records'])
        first_runs[variant] = core.torch.load(output / f'{variant}_first/checkpoints/epoch_001.pth', map_location='cpu', weights_only=False)
        checks.append(variant + '_fullsize_real_input_production_loop_fresh_process_model_optimizer_scheduler_rng_records_best_exact')
        print('PASS ' + variant + ' fresh-process recovery', flush=True)
        del direct, restored, initial
    assert first_runs['D']['records'][0]['train_order'] == first_runs['E']['records'][0]['train_order']
    assert equal_tree(first_runs['D']['rng'], first_runs['E']['rng'])
    checks.append('D_E_first_epoch_sample_order_and_post_epoch_random_states_paired')
    assert mixed.sources() == pinned
    core.json_write(output / 'verification.json', dict(status='pass', source_hashes=pinned,
        test_source_sha256=core.digest(__file__), checks=checks, gpu_fullsize_D_E=True,
        fresh_process_full_state_resume_D_E=True, sensitivity_and_diagnosis_reviewed=False,
        formal_training_ready=False, formal_training_started=False,
        scope='Full 768x1024 GPU batch4+3, real 7/3 image production-loop smoke; not full-dataset accuracy',
        wall_seconds=time.perf_counter()-began))
    print('GPU mechanics passed; formal launch still requires the separate evidence review.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
