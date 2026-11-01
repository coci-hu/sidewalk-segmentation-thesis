"""CPU architecture audit, not accuracy or runtime benchmarking.

Report exact parameter counts and Conv/ConvTranspose/Linear FLOPs (2 per MAC).
Other operators are explicitly excluded; do not label this a full operator count.
Cross-check module hooks with PyTorch 2.5's operator-level FlopCounterMode.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'experiments/attention_gate')]
from model_enet import ENet
from model_enet_ca import ENetCA
from compact_candidate import build_compact


def count(model, x):
    layers, handles = [], []
    def hook(name):
        def observe(module, args, output):
            if isinstance(module, nn.ConvTranspose2d):
                macs = args[0].numel() * (module.out_channels // module.groups) * math.prod(module.kernel_size)
            elif isinstance(module, nn.Conv2d):
                macs = output.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)
            else:
                macs = output.numel() * module.in_features
            layers.append(dict(name=name, kind=type(module).__name__, macs=macs,
                               input_shape=list(args[0].shape), output_shape=list(output.shape)))
        return observe
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            handles.append(module.register_forward_hook(hook(name)))
    model.eval()
    try:
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            output = model(x)
    finally:
        for handle in handles:
            handle.remove()
    macs = sum(r['macs'] for r in layers)
    assert counter.get_total_flops() == 2 * macs, (counter.get_total_flops(), macs)
    assert torch.isfinite(output).all()
    return dict(params=sum(p.numel() for p in model.parameters()),
                trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
                conv_linear_macs=macs, conv_linear_flops=2 * macs,
                gflops_conv_linear=2 * macs / 1e9,
                pytorch_operator_counter_flops=counter.get_total_flops(),
                output_shape=list(output.shape), layers=layers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite a completed profile: ' + str(args.output))
    torch.set_num_threads(1)
    torch.manual_seed(42)
    # Independent closed-form sanity cases include groups and transposed convolution.
    cases = [(nn.Conv2d(4, 6, 3, groups=2), (1, 4, 7, 7), 2700),
             (nn.ConvTranspose2d(4, 6, 2, stride=2, groups=2), (1, 4, 5, 5), 1200),
             (nn.Linear(4, 3), (2, 4), 24)]
    for model, shape, expected in cases:
        assert count(model, torch.zeros(shape))['conv_linear_macs'] == expected
    l0, l1 = build_compact('L0'), build_compact('L1')
    assert all(torch.equal(t, l1.state_dict()[k]) for k, t in l0.state_dict().items())
    models = {'A': ENet(2), 'B': ENetCA(2), 'L0': l0, 'L1': l1}
    expected = {'A': 263640, 'B': 266992}
    x = torch.zeros(1, 3, 768, 1024)
    results = {}
    for name, model in models.items():
        row = count(model, x)
        assert row['output_shape'] == [1, 2, 768, 1024]
        if name in expected:
            assert row['params'] == expected[name]
        row['trained_in_this_audit'] = False
        row['accuracy_evaluated_in_this_audit'] = False
        results[name] = row
        print(name, row['params'], row['gflops_conv_linear'], flush=True)
    for name in ('B', 'L0', 'L1'):
        for base in ('A', 'B'):
            results[name]['reduction_vs_' + base] = {
                metric: 100 * (1 - results[name][metric] / results[base][metric])
                for metric in ('params', 'conv_linear_flops')}
    timing_file = ROOT / 'experiments/attention_gate/step_05_2026-10-14/seed42_training_003/A_seed42_epochs002_100/epoch_timing.json'
    timing = json.loads(timing_file.read_text(encoding='utf-8'))
    seconds = [r['train_seconds'] + r['validation_seconds'] for r in timing[-10:]]
    paths = ['code/model_enet.py', 'code/model_enet_ca.py', 'code/components/ca_attention.py',
             'experiments/attention_gate/compact_candidate.py', 'tools/profile_lightweight.py']
    evidence = dict(status='architecture_precheck_passed', torch_version=torch.__version__,
                    input_shape=list(x.shape), dtype='float32', device='cpu', batch_size=1,
                    convention='Conv2d/ConvTranspose2d/Linear only; 1 MAC = 2 FLOPs; dense execution, no bias adds',
                    excluded_operations=['batch normalization', 'activations including sigmoid', 'pooling and means',
                                         'residual adds', 'attention multiplies', 'interpolation', 'padding and data movement'],
                    full_operator_flops=False, hardware_latency_measured=False,
                    tests=['grouped_conv_closed_form', 'grouped_transpose_closed_form', 'linear_closed_form',
                           'operator_counter_matches_hooks', 'fullsize_finite_output', 'L0_L1_shared_initialization'],
                    results=results,
                    historical_timing=dict(source=str(timing_file.relative_to(ROOT)).replace('\\', '/'),
                        source_sha256=hashlib.sha256(timing_file.read_bytes()).hexdigest(),
                        last_10_epochs_mean_seconds=sum(seconds)/len(seconds),
                        projected_100_epochs_hours=sum(seconds)/len(seconds)*100/3600,
                        note='Historical A reference only; new-model full-epoch pilot required; not a new speed measurement'),
                    source_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
