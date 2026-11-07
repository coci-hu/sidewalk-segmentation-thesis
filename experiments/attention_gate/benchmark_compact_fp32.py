# Versioned A/L0 adapter; timing loops unchanged; fresh paired baseline measurements.
"""Controlled CUDA FP32 latency for already accepted graphs; fresh process per session."""
import argparse
import ctypes
from ctypes import wintypes
import datetime
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = HERE / 'benchmark_compact_protocol.json'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path, obj): Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def external(args):
    # Windows PowerShell5 cannot import PowerShell7 modules inherited from the
    # host's PSModulePath. Let this child resolve its own system module path.
    environment = {k: v for k, v in os.environ.items() if k.lower() != 'psmodulepath'}
    return subprocess.check_output(args, env=environment, creationflags=subprocess.CREATE_NO_WINDOW,
                                   timeout=30).decode('utf-8', errors='replace').strip()


def power():
    class Status(ctypes.Structure):
        _fields_ = [('ACLineStatus', wintypes.BYTE), ('BatteryFlag', wintypes.BYTE),
                    ('BatteryLifePercent', wintypes.BYTE), ('SystemStatusFlag', wintypes.BYTE),
                    ('BatteryLifeTime', wintypes.DWORD), ('BatteryFullLifeTime', wintypes.DWORD)]
    value = Status()
    assert ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(value))
    scheme = re.search(r'[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}', external(['powercfg', '/getactivescheme'])).group().lower()
    return dict(AC=int(value.ACLineStatus), battery_percent=int(value.BatteryLifePercent), scheme=scheme)


def memory_snapshot():
    # Counter query runs outside timed loops. WDDM nvidia-smi process usage is N/A here.
    command = "$ErrorActionPreference='Stop'; $v = Get-Counter -Counter '\\GPU Process Memory(*)\\Dedicated Usage','\\GPU Process Memory(*)\\Shared Usage'; @($v.CounterSamples | Where-Object { $_.InstanceName -like 'pid_" + str(os.getpid()) + "_*' -and $_.Status -eq 0 } | ForEach-Object { @{path=$_.Path; bytes=$_.CookedValue; status=$_.Status} }) | ConvertTo-Json -Compress"
    try:
        values = json.loads(external(['powershell.exe', '-NoProfile', '-Command', command]))
        if isinstance(values, dict): values = [values]
        return dict(status='available' if values else 'unavailable', samples=values,
                    dedicated_bytes=sum(x['bytes'] for x in values if x['path'].lower().endswith('dedicated usage')) if values else None,
                    shared_bytes=sum(x['bytes'] for x in values if x['path'].lower().endswith('shared usage')) if values else None,
                    scope='process across reported adapters; snapshot only, not true peak')
    except Exception as error:
        return dict(status='unavailable', reason=str(error))


def gpu_context():
    return dict(gpu=external(['nvidia-smi', '--query-gpu=name,driver_version,pstate,temperature.gpu,power.draw,power.limit,memory.used,clocks.sm', '--format=csv,noheader']),
                processes=external(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader']))


def worker(a):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    import numpy as np
    import torch  # Also load the installed CUDA DLLs before ORT session initialization.
    import onnxruntime as ort
    import cv2
    from PIL import Image
    import common_runner as core
    import common_runner_v2 as fast
    from deployment_equivalence import provider_evidence
    torch.set_num_threads(1); cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    manifest = read(a.models); protocol = read(PROTOCOL)
    item = next(m for m in manifest['models'] if m['variant'] == a.variant)
    accepted_path = a.accepted.resolve()
    accepted = read(accepted_path)
    assert accepted['status'] == 'pass' and accepted['full500_acceptance_passed']
    known = next(r for r in accepted['results'] if r['variant'] == a.variant)
    assert sha(item['onnx']) == item['onnx_sha256'] == known['onnx_sha256']
    assert sha(item['weights']) == item['weights_sha256'] == known['weights_sha256']
    assert item['seed'] == 42 and item['selected_epoch'] == known['selected_epoch']
    assert manifest['execution']['disabled_optimizers'] == ['ConvBNFusion', 'ConvActivationFusion']
    assert not a.output.exists() and a.output.resolve().is_relative_to(HERE)
    with (HERE / 'seed42_training.lock').open('a+b') as lock:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            a.output.mkdir(parents=True)
            before_power = power()
            assert before_power['AC'] == 1 and before_power['scheme'] == 'a1841308-3541-4fab-bc81-f71556f20b4a'
            sources = {str(p): sha(p) for p in (Path(__file__), PROTOCOL, a.models, accepted_path,
                         Path(core.__file__), Path(fast.__file__), HERE / 'deployment_equivalence.py')}
            dataset = core.SidewalkDataset(fast.protocol(), 'val')
            data_manifest = dataset.manifest()
            assert data_manifest == accepted['data_manifest']
            torch.cuda.init()
            memory = {'before_session': memory_snapshot()}
            context = {'before': gpu_context()}
            options = ort.SessionOptions(); options.intra_op_num_threads = 1; options.inter_op_num_threads = 1
            options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
            options.enable_profiling = True; options.profile_file_prefix = str(a.output / 'provider')
            session = ort.InferenceSession(item['onnx'], sess_options=options,
                providers=[('CUDAExecutionProvider', {'use_tf32': 0})], disabled_optimizers=manifest['execution']['disabled_optimizers'])
            session.disable_fallback()
            assert session.get_providers()[0] == 'CUDAExecutionProvider'
            assert session.get_provider_options()['CUDAExecutionProvider']['use_tf32'] == '0'
            tensor, _ = dataset[0]
            host = tensor.unsqueeze(0).contiguous().numpy()
            input_gpu = ort.OrtValue.ortvalue_from_numpy(host, 'cuda', 0)
            output_gpu = ort.OrtValue.ortvalue_from_shape_and_type([1, 2, 768, 1024], np.float32, 'cuda', 0)
            binding = session.io_binding()
            binding.bind_ortvalue_input('image', input_gpu); binding.bind_ortvalue_output('logits', output_gpu)
            binding.synchronize_inputs(); session.run_with_iobinding(binding); binding.synchronize_outputs()
            bound = output_gpu.numpy()
            ordinary = session.run(['logits'], {'image': host})[0]
            assert np.isfinite(bound).all() and np.allclose(bound, ordinary, rtol=.001, atol=.0001)
            assert np.array_equal(bound.argmax(1), ordinary.argmax(1))
            profile = Path(session.end_profiling())
            provider = provider_evidence(read(profile), 'CUDAExecutionProvider')
            warmup = 2 if a.mode == 'smoke' else protocol['warmup_each_pipeline']
            count = 3 if a.mode == 'smoke' else protocol['measured_frames_each_pipeline']
            dump(a.output / 'plan.json', dict(mode=a.mode, model=item['name'], round=a.round,
                protocol=protocol, source_hashes=sources, power=before_power, created_at=datetime.datetime.now().astimezone().isoformat(),
                warmup=warmup, count=count, model_identity=item, profile_sha256=sha(profile)))
            for _ in range(warmup): session.run_with_iobinding(binding)
            torch.cuda.synchronize()
            memory['after_model_warmup'] = memory_snapshot()
            def progress(stage, completed):
                if (a.output.parent / 'STOP_REQUESTED').exists():
                    raise InterruptedError('User stop marker; partial session is not a benchmark result')
                dump(a.output / 'status.json', dict(stage=stage, completed_frames=completed, target=count))
            raw = []
            wall = time.perf_counter()
            for i in range(count):
                if i % 25 == 0: progress('model_only', i)
                torch.cuda.synchronize(); start = time.perf_counter()
                session.run_with_iobinding(binding); torch.cuda.synchronize()
                raw.append((time.perf_counter() - start) * 1000)
            model_wall = time.perf_counter() - wall
            model_samples = raw

            def pipeline(index):
                with Image.open(dataset.pairs[index][0]) as original:
                    rgb = original.convert('RGB')
                    frame = np.array(rgb)
                    value = dataset.transform(rgb).unsqueeze(0).contiguous().numpy()
                logits = session.run(['logits'], {'image': value})[0]
                mask = logits.argmax(1)[0].astype(np.uint8)
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                selected = mask == 1
                overlay = frame.copy()
                overlay[selected] = (.6 * overlay[selected] + .4 * np.array([40, 210, 40])).astype(np.uint8)
                return overlay, logits

            for i in range(warmup): pipeline(i)
            torch.cuda.synchronize()
            memory['after_pipeline_warmup'] = memory_snapshot()
            raw = []; fingerprints = []
            wall = time.perf_counter()
            for i in range(count):
                if i % 25 == 0: progress('end_to_end', i)
                torch.cuda.synchronize(); start = time.perf_counter()
                overlay, logits = pipeline(i); torch.cuda.synchronize()
                raw.append((time.perf_counter() - start) * 1000)
                # Output guards are outside each timed sample but included in actual loop FPS.
                assert np.isfinite(logits).all()
                fingerprints.append(int(overlay[0, 0].sum()))
            end_wall = time.perf_counter() - wall
            memory['after_measurement'] = memory_snapshot()
            context['after'] = gpu_context()
            after_power = power()
            assert after_power['AC'] == 1 and after_power['scheme'] == before_power['scheme']
            assert dataset.manifest() == data_manifest
            for path, value in sources.items(): assert sha(path) == value
            assert sha(item['onnx']) == item['onnx_sha256'] and sha(item['weights']) == item['weights_sha256']

            def stats(samples, seconds):
                assert len(samples) == count and all(np.isfinite(x) and x > 0 for x in samples)
                return dict(mean_ms=float(np.mean(samples)), median_ms=float(np.median(samples)),
                    p95_ms=float(np.percentile(samples, 95)), actual_fps=count / seconds,
                    loop_wall_seconds=seconds, frames=count)

            dump(a.output / 'samples.json', dict(model_only_ms=model_samples, end_to_end_ms=raw, output_sentinels=fingerprints))
            dump(a.output / 'result.json', dict(status='pass', mode=a.mode, model=item['name'], variant=a.variant, round=a.round,
                model_only=stats(model_samples, model_wall), end_to_end=stats(raw, end_wall),
                memory_snapshots=memory, gpu_context=context, power_before=before_power, power_after=after_power,
                provider=provider, source_hashes=sources, model_identity=item, samples_sha256=sha(a.output / 'samples.json'),
                profile_sha256=sha(profile), io_binding_argmax_matches_ordinary=True,
                formal_timing=a.mode == 'full', accuracy_remeasured=False, true_peak_memory_measured=False))
            progress('complete', count)
            print(item['name'], a.mode, 'completed', flush=True)
        except Exception:
            if a.output.exists(): dump(a.output / 'failure.json', dict(status='failed', traceback=traceback.format_exc()))
            raise
        finally:
            lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def main(a):
    a.output = a.output.resolve(); a.models = a.models.resolve()
    if a.variant:
        worker(a); return
    assert not a.output.exists() and a.output.is_relative_to(HERE)
    a.output.mkdir(parents=True)
    protocol = read(PROTOCOL)
    orders = [['A','L0']] if a.mode == 'smoke' else protocol['round_orders']
    sources = {str(p): sha(p) for p in (Path(__file__), PROTOCOL, a.models)}
    dump(a.output / 'plan.json', dict(protocol=protocol, mode=a.mode, sources=sources,
         created_at=datetime.datetime.now().astimezone().isoformat(), power=power()))
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        results = []
        for number, order in enumerate(orders, 1):
            for variant in order:
                folder = a.output / f'round{number}_{variant}'
                dump(a.output / 'status.json', dict(status='running', round=number, variant=variant, completed=len(results)))
                command = [sys.executable, str(Path(__file__).resolve()), '--models', str(a.models), '--output', str(folder),
                           '--mode', a.mode, '--variant', variant, '--round', str(number), '--accepted', str(a.accepted)]
                with (a.output / f'round{number}_{variant}.stdout.log').open('wb') as stdout, (a.output / f'round{number}_{variant}.stderr.log').open('wb') as stderr:
                    subprocess.run(command, stdout=stdout, stderr=stderr, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
                results.append(read(folder / 'result.json'))
                print('Completed', number, variant, flush=True)
        for path, value in sources.items(): assert sha(path) == value
        dump(a.output / 'results.json', dict(status='pass', mode=a.mode, results=results,
            formal_sessions=len(results) if a.mode == 'full' else 0))
        dump(a.output / 'status.json', dict(status='complete', sessions=len(results), checks_passed=True))
    except Exception:
        dump(a.output / 'failure.json', dict(status='failed', traceback=traceback.format_exc())); raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['smoke', 'full'], required=True)
    parser.add_argument('--variant', choices=['A','L0'])
    parser.add_argument('--accepted', type=Path, required=True)
    parser.add_argument('--round', type=int, default=1)
    main(parser.parse_args())
