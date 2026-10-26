"""Write Step 3 delivery notes only after the completed correctness check."""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to(HERE):
        raise ValueError("Run must be inside attention_gate")
    status = json.loads((run / "step_status.json").read_text(encoding="utf-8"))
    checks = json.loads((run / "checks.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete" and not status["formal_training_started"]
    lookup = {row["check"]: row["details"] for row in checks}
    gradients = lookup["real_768x1024_batch4_backward_and_optimizer"]
    losses = lookup["loss_and_gradients_match_original_CE_and_Dice"]
    integrity = json.loads((run / "integrity_after.json").read_text(encoding="utf-8"))
    memory_rows = "\n".join(f"| {row['variant']} | {row['parameters_m']:.6f} | {row['parameter_tensors_with_finite_gradients']} | {row['cuda_peak_allocated_mib']:.1f} | {row['cuda_peak_allocated_mib']/1024:.2f} |" for row in gradients)
    loss_rows = "\n".join(f"| {row['device']} | {row['loss_absolute_error']:.3g} | {row['gradient_max_absolute_error']:.3g} |" for row in losses)
    report = f"""# 第三步：检查并统一训练入口

第三步已完成，{len(checks)} 项检查通过。本步只进行实现检查和小规模 smoke 训练，未启动全训练集的 100 轮正式训练。成功输出为本目录；前两次检查记录保留在相邻的 `run_001`、`run_002`，不作为完整成功结果。

## 已实现的范围

新增 [common_runner.py](../../common_runner.py) 为 A（现有精简 ENet）和 B（现有 ReLU CA 的 ENet）提供同一入口。C（作者形式 CA）、D/E（固定／可学习混合门控）尚未实现，留待后续证据决定。SBNet-CBR 的消融也未在本步启动。

旧代码、权重、训练日志没有修改。运行前后 {integrity['protected_files_checked']} 个旧文件 SHA-256 一致；第二步正式评估目录的 {integrity['step02_files_checked']} 个文件也全部一致。

## 统一的训练条件

| 项目 | 当前设置 |
|---|---|
| 数据 | Cityscapes train 2975 / val 500；原标签 ID 8 为人行道，其他为背景 |
| 输入 | 768×1024；图像双线性缩放、掩码最近邻；保留旧 ImageNet 标准化 |
| 批次 | 训练 4，保留最后 3 张；验证 1；num_workers=1；不常驻 worker |
| 优化器 | AdamW，lr=1e-4，weight_decay=0.01，betas=(0.9,0.999)，eps=1e-8；显式 foreach=False |
| 学习率 | CosineAnnealingLR，T_max=100，eta_min=1e-6；每完成一轮步进一次 |
| 损失 | 0.4 CE + 0.6 Dice，smooth=1e-6 |
| 精度和策略 | FP32，严格确定性；不启用 AMP、额外增强、早停、预训练或冻结 |
| 种子 | 42、43、44；同种子复制全部共同骨干参数和缓冲量；CA 独立初始化 |
| 样本顺序 | 城市／文件名排序后，由专用 generator 决定每轮顺序；与模型构造随机数分开 |
| 验证指标 | 全图累积混淆矩阵；正确前景 Boundary IoU；按全局 val mIoU 选择最优，平局保留较早轮次 |
| 输出 | 新目录；已有目录拒绝使用；原权重与日志不会作为新训练的输出 |

每轮共 744 个训练批次，最后一批为 3；每张训练图恰好使用一次。0.01 是统一沿用 ENet 基线的条件，不是声称其最优。

同一随机种子不足以保证骨干相同：CA 的构造会占用随机数，使其后解码器初始化发生变化。本入口先构造基线，再复制所有共同状态，单独初始化 CA；CPU 初始化也不会改动共享的 CUDA dropout 随机状态。

## 实测实现检查

三个种子的共同骨干参数和 BN 缓冲量逐张量完全一致，共同子模块结构一致。CA 旁路后，训练模式（包含 dropout）与推理模式输出均逐位一致。矩形特征上的方向池化、拼接、拆分和乘法与直接计算一致；本步验证现有 CA 自身的一致性，没有把它称为作者原版。

2975/500 个图像标签配对与旧入口一致；实际抽查 3 张训练图和 3 张验证图，输入张量与掩码逐位一致。两轮全部训练样本顺序的 A/B 配对检查通过。

全分辨率、batch=4 的前向、损失、反向与更新通过；所有参数张量都有有限梯度，CA 梯度也能传回。检查不证明模型最终收敛或注意力必然有效。

| 组 | 参数 M | 有限梯度参数张量数 | 峰值已分配显存 MiB | 换算 GiB |
|---|---:|---:|---:|---:|
{memory_rows}

这里是 PyTorch 已分配显存，不等于 nvidia-smi 的总占用，也不是所有模型的训练显存。A/B 在本机 RTX 4060 Laptop 8 GB 上通过，不需要为了这两组购买硬件或租服务器。

## 确定性运行的执行修订

本机 Torch 2.5.1+cu121 的二维交叉熵 mean CUDA 算子在严格确定性模式下报错，见 `../run_001/failure.json`。新入口使用 `-log_softmax(logits, dim=1).gather(1, target.unsqueeze(1)).mean()`。当前没有类别权重、标签平滑或被忽略的目标像素，因此它与原无权重逐像素平均 CE 数学等价；FP32 归约顺序可能有微小差异。[PyTorch 交叉熵公式说明](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)

新执行约定单独保存为 [protocol_v1_1.json](../../protocol_v1_1.json)。旧 v1、第二步快照和历史结果保留。数据、损失权重、优化器及候选设计未变。这里引用的文档用于核对公式；CUDA 支持与误差结论来自本机检查。

| 对照设备 | 与旧损失绝对误差 | logits 梯度最大绝对误差 |
|---|---:|---:|
{loss_rows}

新损失通过 CPU/CUDA 数值和梯度对照，也通过严格确定性前向及反向。另一处检查程序的 Windows 子进程中文日志编码问题已修正；原失败记录留在 `../run_002/failure.json`。

## 保存和恢复

A/B 各用 7 张训练图、3 张验证图运行 2 个小轮次；每轮训练批次为 4+3。另在全新 Python 进程中从第 1 轮检查点恢复第 2 轮。整个最终检查点递归比较一致，包括模型、AdamW 状态、调度器、Python/NumPy/CPU/CUDA 随机状态、数据 generator、最优权重、指标和样本顺序。epoch=2 对应 scheduler.last_epoch=2，学习率与解析余弦公式一致。

恢复只承诺在相同代码、协议、数据身份和运行环境下，从已完成轮次的边界继续。中途失败的轮次需要从上一完整检查点重做；不是逐 batch 恢复。恢复时必须提供一个新输出目录，旧检查点和日志不会被覆盖。不同模型组等不兼容的检查点会在开始训练前被拒绝。[PyTorch 对可复现范围的说明](https://docs.pytorch.org/docs/2.14/notes/randomness.html)

smoke 结果只证明入口可运行、能够恢复，不能作为论文精度表、性能结论或正式训练时长估算。

## 本步文件

- [checks.json](checks.json)：全部检查详情。
- [paired_initialization.json](paired_initialization.json)：三个种子的共同骨干／CA 初始校验值。
- [formal_plan_only/settings.json](formal_plan_only/settings.json)：100 轮正式条件的计划；没有训练。
- [A_continuous/status.json](A_continuous/status.json)、[B_continuous/status.json](B_continuous/status.json)：小规模连续运行状态。
- [A_resume_epoch1/status.json](A_resume_epoch1/status.json)、[B_resume_epoch1/status.json](B_resume_epoch1/status.json)：独立进程恢复结果。
- [integrity_after.json](integrity_after.json)：旧文件及第二步产物完整性。
- [step_status.json](step_status.json)：第三步完成状态。

## 后续如何推进

若论文要把 ENet+CA 的变化归因于注意力，需要重新进行相同条件的 A/B 关键对照；不需要把 15 个历史模型全部重跑。历史权重衰减差异只能解释为什么旧比较存在混杂，尚未证明它导致了精度差异。确认训练范围前先完成第四步诊断。

SBNet-CBR 若要保留单模块贡献的论断，需要后续补必要消融；若仅保留为历史整体方案观察，可减少这条支线，把主要精力放到 ENet 的可靠对照。

现在无需用户安装软件、手动运行或购买服务。可以回复“继续第四步，做注意力诊断”。下一步固定训练集样本，检查门控分布、特征变化及旁路／受控干预的现象；不把这些现象单独当作失效原因的证明，也不自动启动正式 100 轮训练。
"""
    with (run / "REPORT.md").open("x", encoding="utf-8") as f:
        f.write(report)
    print("Wrote " + str(run / "REPORT.md"))


if __name__ == "__main__":
    main()
