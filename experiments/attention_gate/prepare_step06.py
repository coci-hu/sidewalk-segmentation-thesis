"""CPU-only correctness checks and a preregistered plan; never formal training.

GPU/full-resolution/production-resume checks remain pending until the live A/B
job finishes. Output must be a new directory. All current source fingerprints
are checked, without rewriting the running job or its immutable protocol.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import argparse
import ast
import copy
import csv
import importlib.util
import json
import math
import statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import common_runner as core
import common_runner_v2 as fast
from ca_hswish_control import PaperHardSwish, author_axis_state, build_c
from components.ca_attention import CoordAtt


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def checkpoint_audit(job, output):
    """Audit a completed immutable checkpoint, never a file being written."""
    plan = read(job / 'job_plan.json')
    assert fast.sources() == plan['source_hashes']
    folder = Path(plan['variant_folders']['A'])
    status = read(folder / 'status.json')
    epoch = status['completed_epoch']
    path = folder / f'checkpoints/epoch_{epoch:03d}.pth'
    cp = torch.load(path, map_location='cpu', weights_only=False)
    assert cp['completed_epoch'] == cp['scheduler_state']['last_epoch'] == epoch
    assert cp['settings']['source_hashes'] == plan['source_hashes']
    assert cp['settings']['protocol'] == fast.protocol()
    assert cp['settings']['variant'] == 'A' and cp['settings']['seed'] == 42
    assert [r['epoch'] for r in cp['records']] == list(range(1, epoch + 1))
    for row in cp['records']:
        assert row['train_images'] == 2975 and row['val_images'] == 500
        assert row['train_batch_sizes'] == [4] * 743 + [3]
        assert sorted(row['train_order']) == list(range(2975))
        assert sum(map(sum, row['val_confusion'])) == 500 * 768 * 1024
        expected_lr = 1e-6 + (1e-4 - 1e-6) * (1 + math.cos(math.pi * row['epoch'] / 100)) / 2
        assert abs(row['lr_next'] - expected_lr) < 1e-14
    assert all(s['step'].item() == epoch * 744 for s in cp['optimizer_state']['state'].values())
    assert all(v.item() == epoch * 744 for k, v in cp['model_state'].items() if k.endswith('num_batches_tracked'))
    assert all(bool(torch.isfinite(v).all()) for v in cp['model_state'].values() if v.is_floating_point())
    old_paths = plan['checkpoint_sources']
    for item in old_paths:
        assert core.digest(item['source']) == item['source_sha256']
        assert core.digest(item['copy']) == item['copy_sha256']
    with (folder / 'metrics.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    # The writer can append a newer row between reads; audit only the captured epoch.
    assert len(rows) >= epoch
    assert abs(float(rows[epoch - 1]['val_miou_global']) - cp['records'][-1]['val_miou_global']) < 1e-12
    times = [r['train_seconds'] + r['validation_seconds'] for r in read(folder / 'epoch_timing.json') if r['epoch'] <= epoch]
    last = times[-5:]
    summary = {'status': 'pass', 'checked_completed_epoch': epoch,
        'checkpoint': str(path), 'checkpoint_sha256': core.digest(path),
        'optimizer_steps_per_parameter': epoch * 744, 'all_completed_epoch_sample_counts_and_learning_rates_correct': True,
        'finite_model_parameters': True, 'pinned_sources_and_original_epoch1_checkpoints_unchanged': True,
        'last_five_epoch_minutes': [x / 60 for x in last], 'recent_mean_epoch_minutes': statistics.mean(last) / 60,
        'A_remaining_hours_extrapolated': (100 - epoch) * statistics.mean(last) / 3600,
        'B_remaining_hours_if_same_rate_only': 99 * statistics.mean(last) / 3600,
        'estimate_caveat': 'B has not been measured with optimized loading; extrapolation excludes interruptions and is not a deadline',
        'live_snapshot': read(job / 'job_status.json'), 'paired_seed42_complete': False,
        'three_seed_study_complete': False}
    core.json_write(output / 'step05_progress_check.json', summary)
    del cp
    return summary


def reference_class(reference):
    source = read(reference / 'source.json')
    path = reference / 'coordatt.py'
    assert source['sha256'] == core.digest(path)
    assert source['commit'] == 'a3f3490ceba779a8443a231d0f792866bd24436d'
    # Inspect the downloaded reference before loading it as local test code.
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in tree.body:
        assert isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef))
    assert {n.name for n in tree.body if isinstance(n, ast.ClassDef)} == {'h_sigmoid', 'h_swish', 'CoordAtt'}
    spec = importlib.util.spec_from_file_location('step06_pinned_author_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, source


def check_models(author, protocol):
    checks = []
    init = []
    for seed in (42, 43, 44):
        torch.manual_seed(9876)
        state_before = torch.get_rng_state().clone()
        a, b, c = core.build_paired_model('A', seed), core.build_paired_model('B', seed), build_c(seed)
        assert torch.equal(state_before, torch.get_rng_state())
        sa, sb, sc = a.state_dict(), b.state_dict(), c.state_dict()
        assert equal_tree(sb, sc)
        assert all(torch.equal(v, sc[k]) for k, v in sa.items())
        assert isinstance(b.ca.act, nn.ReLU) and isinstance(c.ca.act, PaperHardSwish)
        b_modules, c_modules = dict(b.named_modules()), dict(c.named_modules())
        assert set(b_modules) <= set(c_modules)
        assert all(type(b_modules[n]) is type(c_modules[n]) for n in b_modules if n != 'ca.act')
        init.append({'seed': seed, 'B_C_initial_full_state_exact': True, 'A_B_C_shared_backbone_exact': True,
            'construction_preserves_cpu_rng': True, 'parameters_A': sum(p.numel() for p in a.parameters()),
            'parameters_B_C': sum(p.numel() for p in c.parameters()), 'B_C_initial_state_sha256': core.state_digest(sc)})
    checks.append({'check': 'three_seed_initialization_state_and_structure', 'details': init})

    activation = PaperHardSwish()
    values = torch.tensor([-8., -3., -2., -1., 0., 1., 2., 3., 8.], requires_grad=True)
    expected = values * torch.clamp(values + 3, min=0, max=6) / 6
    actual = activation(values)
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
    actual.sum().backward()
    assert bool(torch.isfinite(values.grad).all())
    checks.append({'check': 'paper_activation_values_and_finite_gradient', 'pass': True})

    comparisons = []
    for dtype in (torch.float32, torch.float64):
        for shape in ((2, 16, 7, 11), (1, 16, 5, 13), (2, 16, 9, 3)):
            for training in (False, True):
                torch.manual_seed(123)
                local = CoordAtt(channels=shape[1]).to(dtype=dtype)
                local.act = PaperHardSwish()
                ref = author.CoordAtt(shape[1], shape[1], reduction=32).to(dtype=dtype)
                ref.load_state_dict(author_axis_state(local.state_dict()), strict=True)
                local.train(training)
                ref.train(training)
                x = torch.randn(*shape, dtype=dtype, requires_grad=True)
                xr = x.detach().clone().requires_grad_(True)
                y, yr = local(x), ref(xr)
                atol, rtol = (3e-6, 3e-5) if dtype == torch.float32 else (2e-12, 2e-11)
                torch.testing.assert_close(y, yr, atol=atol, rtol=rtol)
                assert y.shape == x.shape
                weights = torch.randn_like(y)
                (y * weights).sum().backward()
                (yr * weights).sum().backward()
                torch.testing.assert_close(x.grad, xr.grad, atol=atol * 3, rtol=rtol * 3)
                ref_named = dict(ref.named_parameters())
                gradient_max = 0.0
                for name, p in local.named_parameters():
                    mapped_name = name.replace('conv_h.', 'conv_w.', 1) if name.startswith('conv_h.') else (
                        name.replace('conv_w.', 'conv_h.', 1) if name.startswith('conv_w.') else name)
                    other = ref_named[mapped_name]
                    assert p.grad is not None and bool(torch.isfinite(p.grad).all())
                    torch.testing.assert_close(p.grad, other.grad, atol=atol * 10, rtol=rtol * 10)
                    gradient_max = max(gradient_max, (p.grad - other.grad).abs().max().item())
                torch.testing.assert_close(local.bn1.running_mean, ref.bn1.running_mean, atol=atol, rtol=rtol)
                torch.testing.assert_close(local.bn1.running_var, ref.bn1.running_var, atol=atol, rtol=rtol)
                comparisons.append({'dtype': str(dtype), 'shape': list(shape), 'training': training,
                    'output_max_abs_error': (y - yr).abs().max().item(),
                    'input_gradient_max_abs_error': (x.grad - xr.grad).abs().max().item(),
                    'parameter_gradient_max_abs_error': gradient_max, 'atol': atol, 'rtol': rtol})
    checks.append({'check': 'mapped_author_reference_forward_backward_BN_rectangular_shapes', 'details': comparisons})

    b, c = core.build_paired_model('B', 42), build_c(42)
    c.ca.act = nn.ReLU(inplace=True)
    x = torch.randn(2, 3, 64, 96)
    bypass = []
    for training in (False, True):
        b.train(training)
        c.train(training)
        rng = torch.get_rng_state()
        with torch.no_grad():
            yb = b(x)
            torch.set_rng_state(rng)
            yc = c(x)
        assert torch.equal(yb, yc)
        assert equal_tree(b.state_dict(), c.state_dict())
        bypass.append({'training': training, 'output_bitwise_equal': True, 'all_buffers_exact': True})
    checks.append({'check': 'restoring_ReLU_reproduces_B_full_network', 'details': bypass})

    c = build_c(42)
    c.train()
    target = torch.randint(0, 2, (2, 64, 96))
    optimizer, scheduler = core.optimizer_scheduler(c, protocol)
    loss = core.combined_loss(c(x), target, protocol)
    assert torch.isfinite(loss)
    loss.backward()
    gradient = {}
    for name, p in c.named_parameters():
        assert p.grad is not None and bool(torch.isfinite(p.grad).all()), name
        if name.startswith('ca.'):
            norm = p.grad.abs().sum().item()
            assert norm > 0, name
            gradient[name] = norm
    optimizer.step()
    scheduler.step()
    payload = {'model': copy.deepcopy(c.state_dict()), 'optimizer': copy.deepcopy(optimizer.state_dict()),
        'scheduler': copy.deepcopy(scheduler.state_dict()), 'rng': torch.get_rng_state().clone()}
    c2 = build_c(42)
    o2, s2 = core.optimizer_scheduler(c2, protocol)
    c2.load_state_dict(payload['model'], strict=True)
    o2.load_state_dict(payload['optimizer'])
    s2.load_state_dict(payload['scheduler'])
    torch.set_rng_state(payload['rng'])
    c.train()
    optimizer.zero_grad(set_to_none=True)
    loss1 = core.combined_loss(c(x), target, protocol)
    loss1.backward()
    optimizer.step()
    scheduler.step()
    torch.set_rng_state(payload['rng'])
    c2.train()
    o2.zero_grad(set_to_none=True)
    loss2 = core.combined_loss(c2(x), target, protocol)
    loss2.backward()
    o2.step()
    s2.step()
    assert torch.equal(loss1, loss2)
    assert equal_tree(c.state_dict(), c2.state_dict())
    assert equal_tree(optimizer.state_dict(), o2.state_dict())
    assert equal_tree(scheduler.state_dict(), s2.state_dict())
    checks.append({'check': 'CPU_toy_gradient_and_in_memory_restore_next_update', 'ca_gradient_L1': gradient,
        'next_update_state_exact': True, 'scope': 'small CPU synthetic input, not production CUDA resume'})
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--job', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    args = parser.parse_args()
    output, job = args.output.resolve(), args.job.resolve()
    if not output.is_relative_to(core.HERE) or output == core.HERE or output.exists():
        raise ValueError('Output must be an unused child directory')
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_available(), 'Preparation must not access the occupied GPU'
    before = fast.sources()
    start = datetime.now(timezone(timedelta(hours=8))).isoformat()
    author, provenance = reference_class(args.reference.resolve())
    protocol = fast.protocol()
    progress = checkpoint_audit(job, output)
    checks = check_models(author, protocol)
    assert fast.sources() == before == read(job / 'job_plan.json')['source_hashes']
    core.json_write(output / 'cpu_checks.json', {'status': 'pass', 'checks': checks,
        'CUDA_device_hidden': True, 'CPU_threads': 2, 'formal_training_started': False,
        'new_validation_images_evaluated': 0, 'new_C_CUDA_checks_completed': False})
    source_files = [Path(__file__), core.HERE / 'ca_hswish_control.py',
                    args.reference.resolve() / 'coordatt.py', args.reference.resolve() / 'source.json']
    hashes = {str(p.relative_to(core.PROJECT)): core.digest(p) for p in source_files}
    plan = {'status': 'CPU_prepared_GPU_checks_pending', 'started': start,
        'base_protocol': 'protocol_v1_2.json', 'base_protocol_sha256': core.digest(fast.PROTOCOL_FILE),
        'scientific_base': {'data': protocol['data'], 'metrics': protocol['metrics'], 'formal_training': protocol['formal_training']},
        'variant': 'C', 'label': 'ENet + paper-form h-swish CA, legacy axis layout preserved',
        'sole_intended_change_from_B': 'ca.act: nn.ReLU -> x * (ReLU6(x + 3) / 6)',
        'unchanged': ['backbone and stage3 insertion', 'all initial tensors including CA', 'mean-pooling axes and concatenation order',
                      'convolution biases and bottleneck min8/reduction32', 'normalization and training protocol', 'parameter count'],
        'author_reference': provenance, 'axis_mapping': 'local conv_h(width) -> author conv_w; local conv_w(height) -> author conv_h',
        'equivalence_scope': 'matched weights forward/backward/BN within float32 and float64 tolerances; not bitwise equality to official axis layout',
        'initialization': 'construct paired B from seed, replace parameter-free activation only; fresh training from epoch1',
        'never_resume_C_from_B_trained_checkpoint': True,
        'runtime_source_hashes': hashes, 'running_A_B_sources_unchanged': before,
        'preparation_proof': str(output / 'cpu_checks.json'), 'new_C_formal_training_started': False,
        'GPU_and_entry_checks_still_required': ['full 768x1024 batch4 forward/backward and memory',
            'new versioned runner labels C in every settings/checkpoint/report',
            'production CUDA fresh-process epoch-boundary resume equivalence',
            'strict rejection of A/B checkpoints when executing C'],
        'schedule': ['finish and verify A/B seed42', 'review and complete the planned A/B paired seeds43/44',
                     'after GPU checks, run C seed42 as exploration; any formal C claim requires matching seed repeats'],
        'interpretation': {'B_vs_C': 'effect of changing activation within this specific CA implementation',
            'C_vs_A': 'whether paper-form activation attention helps this task under fixed conditions',
            'single_seed': 'exploratory, not stable benefit/failure',
            'C_improves': 'C becomes a necessary comparator for any proposed new gate',
            'no_gain': 'retain negative results, do not search validation for a preferred narrative'},
        'not_prepared_or_started': ['D/E candidate full training', 'other seeds full training', 'second dataset', 'second backbone', 'paid services']}
    core.json_write(output / 'step06_plan.json', plan)
    core.json_write(output / 'verification.json', {'status': 'pass', 'scope': 'CPU preparation only',
        'checks': len(checks), 'running_job_sources_unchanged': True, 'source_hashes': hashes,
        'C_GPU_production_readiness': False, 'C_formal_training_started': False})
    minutes = progress['recent_mean_epoch_minutes']
    report = f'''# 第五步进度与第六步准备

第五步正在运行。本次只新增第六步C的模型构建、CPU预检和实验计划，没有修改后台任务，也没有启动C正式训练。

## 第五步核验

- 核验时ENet已保存第{progress['checked_completed_epoch']}轮；ENet+CA仍等待顺序运行。
- 已保存轮次的2975张训练图、500张验证图、744批次、样本排列、余弦学习率及优化器步数正确；模型参数均为有限值。
- 当前冻结源码、原始epoch1检查点及执行迁移副本哈希未改变。
- 最近5轮平均训练加验证约{minutes:.2f}分钟。A剩余约{progress['A_remaining_hours_extrapolated']:.1f}小时；若B耗时相同，B另外约{progress['B_remaining_hours_if_same_rate_only']:.1f}小时。B尚无优化后整轮实测，该估计受后台负载、温度与中断影响。
- [进度核验快照](step05_progress_check.json)只反映检查时刻；[后台实时状态](../../../step_05_2026-10-14/seed42_training_001/job_status.json)持续更新。

本次只审查运行正确性，不用未完成的A曲线与未运行的B比较。种子42的两组完成也不等于三个配对种子研究完成。

## 第六步具体准备

[C模型构建](../../../ca_hswish_control.py)调用已有配对初始化，仅把CA内部ReLU换成作者形式h-swish，不增加参数。A为263640参数，B和C均为266992参数。新模块不加载已训练B权重。

[作者源码快照](../author_reference/source.json)固定到提交`{provenance['commit']}`。原本地CA先沿宽方向编码并把对应分支命名为conv_h，作者命名恰好相反；这是布局与命名的差异，不能仅据此判断原实现轴向错误。比较作者代码时交换方向分支权重。单因素C保留原布局，避免把激活和轴布局同时改变。

[CPU检查](cpu_checks.json)通过：三个种子的A/B/C共同初值和B/C全部状态完全相同；构造不改变外部CPU随机状态；非方形输入上与作者实现的前向、输入梯度、参数梯度及BN状态在记录的浮点容差内一致；恢复ReLU后整网输出逐位复现B；小型合成输入的梯度与状态恢复后下一次更新一致。

上述不是CUDA全尺寸训练或生产入口断点恢复证明。当前GPU隐藏，仅使用2个CPU计算线程，没有重新读取训练图片、没有新验证集预测，也没有占用CUDA显存。CPU检查仍会短暂使用少量CPU与内存。

## 真正进入第六步的顺序

1. 等当前A/B种子42完成并自动核验，先审阅配对结果；随后补足原定43、44种子，完成第五步的重复证据。
2. GPU空闲后完成C的全尺寸梯度与内存检查，新增可明确记录C身份的训练入口，并验证独立进程恢复及错误检查点拒绝。现有A/B入口限制A/B，不直接拿它把C伪装成B。
3. 按固定协议从epoch1开始C的42种子探索；若写入正式对照或改变研究判断，再补齐对应重复。所有正式C比较需要与A/B匹配的种子，不能只展示最好一次。

## 论文证据安排

| 要回答的问题 | 必要比较或证据 | 可以写到什么程度 |
|---|---|---|
| 注意力收益是否可复现 | A/B三个配对种子，逐种子差值及均值标准差 | 当前协议与任务上的稳定性，不能外推普遍失效 |
| 激活实现是否影响结果 | B/C相同初始化、位置和训练条件 | 激活替换的效应；更好时不代表独创算法 |
| 是否需要混合门控 | C与固定/可学习门控的条件性消融 | 只有后续证据支持时才进入，不提前宣称有效 |
| 是否有工业应用价值 | 导出一致性、端到端延迟、资源占用与场景样例 | 实测平台范围内的精度与成本取舍 |

论文可先整理任务定义、相关工作和统一实验方法。结果表和结论暂留待实测，不把h-swish对照本身列作原创贡献。部署结论限定实际使用的RTX4060平台；工业嵌入式设备尚未验证。

## 你需要做什么

现在继续保持接电，避免手动休眠、关机或另开GPU训练；不用手工运行C。可以随时说“查看第五步进度”。后台状态complete且verified为true后，可以说“检查种子42结果并安排剩余重复实验”。[第六步计划](step06_plan.json)记录了已准备和待检查事项。
'''
    (output / 'REPORT.md').write_text(report, encoding='utf-8')
    print(json.dumps({'status': 'pass', 'checks': len(checks), 'A_checked_epoch': progress['checked_completed_epoch'],
        'C_training_started': False, 'output': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
