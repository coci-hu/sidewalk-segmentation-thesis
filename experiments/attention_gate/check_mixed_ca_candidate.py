"""CPU candidate checks only; GPU/full-trainer/resume acceptance still pending."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import copy
from pathlib import Path
import torch
import common_runner as core
import common_runner_c as c
import mixed_ca_candidate as candidate
from ca_hswish_control import build_c
from check_step03 import equal_tree


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    output = a.output.resolve()
    assert output.is_relative_to(core.HERE) and not output.exists()
    torch.set_num_threads(2)
    torch.manual_seed(70124)
    checks = []
    for seed in (42, 43, 44):
        base, d, e = build_c(seed), candidate.build_candidate('D', seed), candidate.build_candidate('E', seed)
        original = base.state_dict()
        for model in (d, e):
            assert equal_tree(original, {k: v for k, v in model.state_dict().items() if k in original})
        assert sum(p.numel() for p in d.parameters()) == 266992
        assert sum(p.numel() for p in e.parameters()) == 266993
        assert torch.equal(d.ca.alpha(), e.ca.alpha())
        groups = candidate.parameter_groups(e)
        parameter_ids = [id(p) for group in groups for p in group['params']]
        assert len(parameter_ids) == len(set(parameter_ids)) == len(list(e.parameters()))
        assert groups[-1]['params'][0] is e.ca.beta and groups[-1]['weight_decay'] == 0
        checks.append(f'seed{seed}_shared_initial_state_alpha_parameters_and_optimizer_groups')
    base.eval(); d.eval(); e.eval()
    x = torch.randn(2, 128, 96, 128)
    with torch.no_grad():
        assert torch.equal(d.ca(x), e.ca(x))
        saved = d.ca.fixed_alpha.clone()
        d.ca.fixed_alpha.fill_(1)
        assert torch.equal(d.ca(x), base.ca(x))
        d.ca.fixed_alpha.zero_()
        assert torch.equal(d.ca(x), x)
        d.ca.fixed_alpha.copy_(saved)
    checks.append('actual_feature_shape_D_E_initial_forward_exact_and_alpha_endpoints_exact')
    e.train()
    optimizer = torch.optim.AdamW(candidate.parameter_groups(e), lr=1e-4, foreach=False)
    before = e.ca.beta.detach().clone()
    target = x * 0.7
    loss = (e.ca(x) - target).square().mean()
    loss.backward()
    assert e.ca.beta.grad is not None and torch.isfinite(e.ca.beta.grad) and e.ca.beta.grad.abs() > 0
    assert e.ca.conv1.weight.grad is not None and e.ca.conv1.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, e.ca.beta) and 0 < e.ca.alpha().item() < 1
    checks.append('beta_and_CA_gradients_finite_nonzero_and_beta_updated')
    output.mkdir(parents=True)
    core.json_write(output / 'verification.json', {'status': 'pass', 'scope': 'CPU candidate component only',
        'checks': checks, 'source_hashes': c.sources(), 'candidate_sha256': core.digest(candidate.__file__),
        'check_sha256': core.digest(__file__), 'training_started': False,
        'formal_training_ready': False, 'pending': ['explicit D/E checkpoint identity and source-pinned trainer',
            'GPU fullsize training', 'fresh-process full-state resume equivalence', 'B/C diagnostics and sensitivity review'],
        'fixed_and_initial_alpha': 0.1, 'beta_weight_decay': 0, 'endpoint_outputs_bitwise_equal_on_CPU': True})
    print({'status': 'pass', 'checks': len(checks), 'formal_training_ready': False}, flush=True)


if __name__ == '__main__':
    main()
