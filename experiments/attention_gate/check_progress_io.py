"""Reproduce Windows file locks and check numerical/RNG equivalence on CUDA."""
import argparse
import copy
import ctypes
import json
from pathlib import Path
import common_runner as core
import common_runner_v2 as fast
import safe_progress_io as io
from check_step03 import equal_tree


def locked(path):
    kernel = ctypes.windll.kernel32
    kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.CreateFileW(str(path), 0x80000000, 1, None, 3, 0x80, None)
    if handle in (None, ctypes.c_void_p(-1).value):
        raise ctypes.WinError()
    return lambda: kernel.CloseHandle(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    assert output.is_relative_to(core.HERE) and not output.exists()
    output.mkdir(parents=True)
    core.torch.set_num_threads(2)
    checks = []
    def passed(name, detail):
        checks.append({'check': name, 'status': 'pass', 'detail': detail})
        print('PASS ' + name, flush=True)
    frozen = fast.sources()
    p = output / 'training_progress.json'
    core.json_write(p, {'old': True})
    release = locked(p)
    try:
        reproduced = False
        try:
            fast.atomic_json(p, {'new': True})
        except PermissionError:
            reproduced = True
        assert reproduced
        assert io.atomic_json(p, {'new': True}, attempts=2, sleep=lambda _: None) is False
        assert json.loads(p.read_text()) == {'old': True}
    finally:
        release()
    passed('real_Windows_lock_and_optional_fallback', 'Original replace fails; locked progress is safely skipped without overwriting old JSON')
    core.json_write(output / 'job_status.json', {'old': True})
    release = locked(output / 'job_status.json')
    try:
        rejected = False
        try:
            io.atomic_json(output / 'job_status.json', {}, attempts=2, sleep=lambda _: None)
        except PermissionError:
            rejected = True
        assert rejected
    finally:
        release()
    for value in ({'bad': float('nan')},):
        rejected = False
        try:
            io.atomic_json(p, value)
        except ValueError:
            rejected = True
        assert rejected
    passed('critical_and_serialization_errors_not_hidden', 'Authoritative status lock and nonfinite JSON are rejected')
    calls = []
    def transient(source, destination):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError('simulated brief sharing violation')
        io.os.replace(source, destination)
    assert io.atomic_json(p, {'retry_success': True}, replace=transient, sleep=lambda _: None)
    assert len(calls) == 3 and json.loads(p.read_text()) == {'retry_success': True}
    passed('bounded_retry_recovers', 'Two replace failures then successful identical JSON')
    checkpoint = core.torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    assert checkpoint['completed_epoch'] == 93
    assert checkpoint['settings'] == fast.expected_settings('A', 43)
    protocol = fast.protocol()
    train = core.SidewalkDataset(protocol, 'train')
    batch = [train[i] for i in range(4)]
    images = core.torch.stack([x[0] for x in batch])
    target = core.torch.stack([x[1] for x in batch])
    reference = None
    for profile in ('original', 'safe_with_locked_snapshot'):
        core.configure(43)
        model = core.build_paired_model('A', 43).cuda()
        optimizer, scheduler = core.optimizer_scheduler(model, protocol)
        model.load_state_dict(checkpoint['model_state'], strict=True)
        optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer_state']))
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        gens = core.generators(43)
        core.restore_rng(checkpoint['rng'], gens)
        release = locked(p) if profile != 'original' else None
        def batches():
            yield images, target
            if profile == 'original':
                fast.atomic_json(output / 'baseline_progress.json', {'test': 1})
            else:
                assert not io.atomic_json(p, {'test': 1}, attempts=2, sleep=lambda _: None)
        try:
            result = fast.BASE_TRAIN(model, batches(), optimizer, protocol, core.torch.device('cuda:0'))
            core.torch.cuda.synchronize()
            state = {'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                'optimizer': copy.deepcopy(optimizer.state_dict()), 'scheduler': scheduler.state_dict(),
                'rng': core.capture_rng(gens), 'loss': result}
            if reference is None:
                reference = state
            else:
                assert equal_tree(reference, state)
        finally:
            if release:
                release()
            del model, optimizer, scheduler
    passed('CUDA_fullsize_resumed_update_bitwise_equal', 'Checkpoint93, 768x1024 batch4: loss, weights, BN, optimizer, scheduler and all RNG exact despite skipped locked snapshot')
    assert fast.sources() == frozen
    assert not list(output.glob('.*.tmp'))
    passed('frozen_sources_and_temporary_cleanup', 'No scientific source changes; own temporary files cleaned')
    core.json_write(output / 'verification.json', {'status': 'pass', 'checks': checks,
        'frozen_source_hashes': frozen,
        'adapter_sha256': core.digest(Path(io.__file__)),
        'test_source_sha256': core.digest(__file__),
        'checkpoint_sha256': core.digest(args.checkpoint),
        'test_only_no_formal_training': True})


if __name__ == '__main__':
    main()
