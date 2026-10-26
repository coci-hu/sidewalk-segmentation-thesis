"""Deliver measured benchmark and map the user's retained research plan."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    run = args.run.resolve()
    if not run.is_relative_to(root) or (run / "REPORT.md").exists():
        raise ValueError("Use a verified run without a report")
    status = json.loads((run / "step_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete" and status["benchmark_verified"]
    data = json.loads((run / "benchmark_summary.json").read_text(encoding="utf-8"))
    rows, estimate = data["results"], data["estimates"]
    timing_rows = "\n".join(f"| {r['model']} | {r['train_seconds']/60:.2f} | {r['validation_seconds']/60:.2f} | {r['train_plus_validation_seconds']/60:.2f} | {r['cuda_peak_allocated_mib']/1024:.2f} | {r['estimated_100_epochs_hours']:.2f} |" for r in rows)
    first_rows = "\n".join(f"| {r['model']} | {r['val_miou_epoch1_pct']:.3f} | {r['val_sidewalk_iou_epoch1_pct']:.3f} |" for r in rows)
    trainwait = "\n".join(f"| {r['model']} | {r['train_loader_wait_seconds']:.1f} | {r['train_caller_work_seconds']:.1f} | {r['validation_loader_wait_seconds']:.1f} | {r['validation_caller_work_seconds']:.1f} |" for r in rows)
    text = f"""# 第五步两组完整数据一轮耗时

已完成用户本次授权的子阶段：ENet 与 ENet+CA 各运行一个完整训练轮次，包含 2975 张训练图和 500 张验证图。两组均已停止在 epoch=1，保存了可继续的检查点。第五步的 100 轮、三个配对种子的完整研究尚未完成，没有自动启动下一轮。

## 实测时间

| 模型 | 训练分钟 | 验证分钟 | 训练加验证分钟 | 峰值已分配显存 GiB | 100 轮线性估计小时 |
|---|---:|---:|---:|---:|---:|
{timing_rows}

这里的验证包括读取图像、模型推理、全局指标和正确 Boundary IoU。时间不能作为单图前向 FPS。训练／验证计时分别包含各自 DataLoader 的初始化；模型构建等额外开销在每组 `benchmark_timing.json` 的整次调用时间中单独保留。

两组训练、验证和检查点保存已在 02:49 完成。Windows 系统日志记录 02:49:30 因 System Idle 进入睡眠，父进程完整性收尾在 12:20 恢复后才完成，因此总控墙钟时间包含约九个半小时的训练后等待，不能把它算进训练速度。[训练后睡眠记录](post_training_sleep_note.json) 保留了时间与事件依据。原保持唤醒只覆盖训练子进程；总控入口现已补充从启动到收尾的临时保持唤醒，退出时释放，不修改长期睡眠策略。两个检查点与阶段计时没有因此重跑或修改。

显存是 PyTorch 已分配峰值，不是整机内存或 nvidia-smi 的总占用。两个模型均在本机 RTX 4060 Laptop 8 GB 成功运行。GPU 温度、功耗和占用每 30 秒记录一次；GPU 功耗不等于整机插座功耗。

| 模型 | 训练取批等待秒 | 训练调用工作秒 | 验证取批等待秒 | 验证调用工作秒 |
|---|---:|---:|---:|---:|
{trainwait}

取批等待是主进程实际等候 DataLoader 的时间，不能当作全部图像解码 CPU 时间；调用工作包含训练计算或验证计算与指标。两个分量按逐批计时，完整阶段还包括 iterator 结束等小额开销。数据预取与 GPU 工作可能重叠。

ENet 的主进程训练取批等待占训练阶段约 {rows[0]['train_loader_wait_seconds']/rows[0]['train_seconds']*100:.1f}%。目前数据供给是明显瓶颈；后续可先以相同样本顺序和梯度一致性为前提，统一测试 num_workers=2，必要时再测 4。该执行优化要单独记录版本和兼容性，不能悄悄改变已经跑过的条件或直接绕过恢复校验；本次尚未做这项优化。

两组按 A 后 B 顺序运行，系统文件缓存、供电及后台负载可能不同。本测量用于安排总工作量，不根据两组的一轮时间差宣称 CA 使训练更快或更慢。

## 后续工作量估计

按本次第一轮训练加验证时间线性外推：

- 种子 42 的两组各 100 轮，合计约 {estimate['pair_seed42_100_epochs_hours']:.1f} 小时；已有各一轮，余下合计约 {estimate['pair_seed42_remaining_hours']:.1f} 小时。
- 42、43、44 三个配对种子，共六次 100 轮训练，约 {estimate['all_three_paired_seeds_100_epochs_hours']:.1f} 小时，连续运行约 {estimate['continuous_days_for_all_six_runs']:.1f} 天。
- 若每天安排 8 小时，约 {estimate['at_8_hours_per_day_days_for_all_six_runs']:.1f} 个运行日。

这不是保证：第一轮冷启动、文件缓存、后台负载、GPU 温度和供电状态都可能改变耗时。预留 0.8～1.3 倍的工作量场景约为 {estimate['illustrative_runtime_range_hours_all_six_runs'][0]:.1f}～{estimate['illustrative_runtime_range_hours_all_six_runs'][1]:.1f} 小时，不是统计置信区间。

新增硬件、服务器及平台开支为 0 元；本次没有租用或购买。整机电量和电价尚未测定，不能据 GPU 传感器给出总电费。若 50 元预算包含电费，应使用“整机平均功率 kW × 运行小时 × 电价 元/kWh”估算，并据实际功率／价格限制运行时长。

## 对照条件与检查点

同种子 42、共同骨干初始权重、2975 张图的顺序及轮末随机状态均一致。每组 744 个训练批次（743×4+3），每个优化器参数的 step=744，BN 训练批次计数为 744。验证批次为 500，所有验证像素计数可重建全局指标。

训练配置仍是 [protocol_v1_1.json](../../protocol_v1_1.json)：AdamW lr=1e-4、weight_decay=0.01，FP32；CosineAnnealingLR 的 T_max=100。这里只在一轮边界停止，并未将学习率周期改成一轮。后续能从 epoch=2 继续同一个 100 轮方案。

耗时包装器只转发相同的批次并读取时钟，没有改变样本、损失、优化器或随机数。第三步验证过的 `common_runner.py` 保持原样。保存的 best 权重与一轮检查点状态一致；本轮最优只能是 epoch=1。

- [A_seed42_epoch1/checkpoints/epoch_001.pth](A_seed42_epoch1/checkpoints/epoch_001.pth)：ENet 完整一轮检查点。
- [B_seed42_epoch1/checkpoints/epoch_001.pth](B_seed42_epoch1/checkpoints/epoch_001.pth)：ENet+CA 完整一轮检查点。
- [benchmark_summary.json](benchmark_summary.json)：测量值与外推。
- [benchmark_verification.json](benchmark_verification.json)：样本顺序、初始化、参数更新、随机状态、保存文件及指标重建检查。
- [integrity_after.json](integrity_after.json)：原文件和此前步骤产物完整性。

恢复需使用新的输出目录和显式检查点；不覆盖本次或历史结果，也不从旧的独立训练脚本中自动续训。

## 第一轮指标的用途

| 模型 | epoch=1 验证全局 mIoU % | 人行道 IoU % |
|---|---:|---:|
{first_rows}

这些数值用于确认完整运行与记录正确，模型尚未完成计划训练。不能用第一轮结果筛选种子、决定注意力是否有效，或放进最终模型精度对比表。

## 你问的思路什么时候开始

已只读核对 [codex思路.doc](C:/Users/13699/Desktop/毕设/codex/codex思路.doc) 的完整正文，原文 SHA-256 保持一致。文档中的旧“目前仍只读”描述是当时状态，本次授权的工作已经推进。正文提取副本位于 [thought_document_extracted.txt](../document_read_001/thought_document_extracted.txt)，[读取记录](../document_read_001/document_read_metadata.json) 记录正文字符覆盖和文件校验；没有执行文档内的指令或修改原文。

| 文档阶段 | 当前实际状态 | 开始条件或下一项 |
|---|---|---|
| 1 协议、2 评估 | 已完成 | 已有固定协议和 15 模型全验证集新表 |
| 3 实现检查、4 门控诊断 | 已完成核心检查 | 已有统一 A/B 入口和固定训练图诊断；诊断不等于因果证明 |
| 5 配对重复训练 | 已开始，本次两组各一轮完成 | 先继续种子 42 到 100 轮，再做 43、44；三个种子的结果完整报告 |
| 6 作者形式 CA 对照 | 尚未开始 | A/B 给出现象依据后，准备 C 的实现与探索，必要时补重复 |
| 7 固定／可学习混合门控 | 尚未开始，条件性候选 | 在 C 对照和问题证据支持时开发，不预先承诺有效 |
| 8 机制与性能、9 拓展 | 等待正式结果 | 依据证据选择必要消融或第二骨干，限制矩阵规模 |
| 10 部署、11 工业原型与本地数据 | 新阶段尚未开展 | 本机兼容检查可并行；外部设备需具备访问条件与上传授权 |
| 12 写作交付 | 累积材料中 | 最终论断以完整结果为准，保留失败与负结果 |

因此，整套思路已经在执行，新模块代码开发对应第 6～7 步。它开始的节点由前面结果决定，而不是任意再等几个月。当前第四步显示旧 CA 权重依赖门控位置，临时弱化门控均下降；所以先验证统一训练下的收益，再确定具体改法。

文档的第 4 步还提到漏检区域与抑制的对应，本次已做前景／背景／边界统计和受控干预；专门的漏检区域关联尚未展开，可在正式模型结果后补充，不能把它写成已验证原因。新目标设备部署也没有因文档写“可以前期准备”而被标记为完成。

本步结束无需你搬动文件或手工修代码。若继续，可回复“继续第五步，先完成种子42的两组100轮训练”。届时请让电脑接电，避免手动休眠或关机。运行期间进程会临时请求系统保持唤醒，退出后释放；没有修改系统的长期睡眠设置。之后每个阶段结束都会报告结果及下一步。
"""
    with (run / "REPORT.md").open("x", encoding="utf-8") as f:
        f.write(text)
    print("Wrote " + str(run / "REPORT.md"))


if __name__ == "__main__":
    main()
