"""Report verified execution optimization and actual continuation state."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    step = root / "step_05_2026-10-14"
    output = step / "loader_optimization_001/REPORT.md"
    if output.exists():
        raise FileExistsError(output)
    proof = read(step / "loader_optimization_001/verification.json")
    quality = read(step / "loader_v2_integration_001/verification.json")
    migration = read(step / "loader_migration_001/migration_verification.json")
    job = args.job.resolve()
    plan, live = read(job / "job_plan.json"), read(job / "job_status.json")
    assert proof["status"] == quality["status"] == migration["status"] == "pass"
    profiles = [p for p in proof["profiles"] if not p.get("skipped")]
    baseline = next(p for p in profiles if p["workers"] == 1)
    rows = "\n".join(f"| {p['workers']} | {p['steady_images_per_second']:.2f} | {p['steady_images_per_second']/baseline['steady_images_per_second']:.2f}× | 一致 |" for p in profiles)
    epoch2_file = job / "A_seed42_epochs002_100/epoch_timing.json"
    measured = "优化后完整轮次仍在运行，暂不将数据供给倍数当作训练加速倍数。"
    if epoch2_file.exists():
        completed = [r for r in read(epoch2_file) if "validation_seconds" in r]
        if completed:
            first = completed[0]
            total = first["train_seconds"] + first["validation_seconds"]
            measured = f"ENet 首个优化完整轮次（epoch={first['epoch']}）实测：训练 {first['train_seconds']/60:.2f} 分钟，验证 {first['validation_seconds']/60:.2f} 分钟，合计 {total/60:.2f} 分钟。旧一轮为 14.64 分钟，本次比值约 {878.2159039/total:.2f}×；缓存和后台负载不同，该比值不作为固定性能保证。ENet+CA 的优化完整轮次尚需其实际运行后确认。"
    text = f"""# 数据读取优化与第五步续训

读取优化、更新一致性和独立进程恢复检查通过，统一选择 {proof['selected_workers']} 个读取进程。种子42的两组正式对照已启动；本报告不将启动写成100轮完成，也不将一个种子写成三种子研究完成。

## 选择依据

| num_workers | 定常数据供给 图/秒 | 相对原1进程 | 输入与训练更新 |
|---|---:|---:|---|
{rows}

同一组192张固定训练图，排除首8个批次后测剩余160张数据供给。选择只看吞吐与可行性，不看模型验证精度；若较小进程数在最高吞吐的5%以内，选较小者。4个进程在已测候选中最快；没有声称是所有硬件和进程数中的全局最优。

这些是DataLoader数据供给速度，不是模型FPS，也不是整轮训练加速倍数。{measured}

## 一致性与恢复

跨18个训练城市的36张图和3个验证城市样本，逐张图像张量与掩码SHA-256一致，样本顺序一致。A/B各从已有一轮检查点出发，在7张训练图和3张验证图上执行2个小轮次，1、2、4进程的模型、AdamW、调度器、全部随机状态、指标及顺序逐项完全一致。

新执行入口再用B做连续小轮次与全新Python进程恢复，整个最终检查点完全一致。未迁移的旧执行检查点被新入口严格拒绝；没有关闭原来的配置／源码／数据／环境匹配检查。[读取检查](verification.json) · [新入口恢复检查](../loader_v2_integration_001/verification.json)

新增[protocol_v1_2.json](../../protocol_v1_2.json)只改变num_workers。输入尺寸、标签、标准化、无额外增强、共享初始化、样本顺序、batch=4、保留尾批3张、AdamW参数、损失、FP32、100轮余弦周期和全局验证选权重均保持原样。prefetch_factor仍为PyTorch默认2，persistent_workers仍为false。

[common_runner_v2.py](../../common_runner_v2.py)是执行适配器，实际训练、损失、验证和保存循环仍调用未修改的common_runner.py。每轮开始核对源码指纹，运行期间改动源码会停止，而不是混用版本。

## 已有一轮如何承接

两份旧epoch=1检查点只读保留；[新执行副本](../loader_migration_001/migration_verification.json)记录原路径、原哈希、原配置及新执行指纹。逐项证明模型、最优权重、优化器、调度器、随机状态和历史指标未改变。两组同样使用首轮1进程、从第二轮起统一{proof['selected_workers']}进程的执行历史。

修改的是新副本的执行元数据，没有篡改原结果或假装旧首轮已使用4进程。迁移不会把历史独立训练权重混入新对照。

## 当前正式任务

- 任务目录：`{job}`。
- 范围：种子42，A=ENet、B=ENet+当前ReLU CA，分别从epoch=2继续到100；A后B顺序运行。
- 生成报告时状态：`{live['status']}`，当前组别`{live.get('variant', '见实时状态')}`。
- [实时状态](../{job.name}/job_status.json)每15秒更新；每个模型另有training_progress.json和逐轮metrics.csv。
- 每完成一轮保存新检查点；失败不自动覆盖重启。恢复需使用新目录和显式检查点。
- 根目录出现STOP_AFTER_EPOCH文件时，当前完整轮次保存后停止；需要停止时可以直接告诉我。
- 两组完成后自动核验100轮、74400次优化器更新、学习率、最优权重、配对顺序、轮末随机状态和旧文件完整性，并生成REPORT.md及学习曲线，然后停止。
- 没有自动开展43、44种子，没有训练C/D/E或其他模块，没有租服务器或上传数据。

长任务在本机独立后台运行，总控和训练过程都临时保持系统唤醒，退出后释放；没有修改长期睡眠策略。请保持接电，避免手动休眠、关机或运行另一个GPU训练任务。关闭屏幕不影响该请求。

## 下一次何时找我

现在无需手工搬文件或改代码。运行中可以随时说“查看第五步进度”。当实时状态为complete且verified为true时，两组种子42已完成并核验，可以说“检查种子42结果，决定下一阶段”。若出现failed或stopped_at_epoch_boundary，保留目录并发来状态或报错，不要删除检查点。

正式算法思路仍按证据推进：先审阅配对训练，再决定标准CA对照及候选改进。这次读取加速属于执行效率改进，不作为新的注意力算法创新。
"""
    with output.open("x", encoding="utf-8") as f:
        f.write(text)
    print("Wrote " + str(output))


if __name__ == "__main__":
    main()
