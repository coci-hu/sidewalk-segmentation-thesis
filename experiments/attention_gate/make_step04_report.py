"""Evidence-based Step 4 report; never overwrite existing reports or old results."""
import argparse
import json
from pathlib import Path
import common_runner as cr
from evaluate_existing import timestamp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to(cr.HERE) or (run / "REPORT.md").exists():
        raise ValueError("Use a verified new report destination")
    tables = json.loads((run / "analysis_tables.json").read_text(encoding="utf-8"))
    verification = json.loads((run / "analysis_verification.json").read_text(encoding="utf-8"))
    integrity = json.loads((run / "integrity_after.json").read_text(encoding="utf-8"))
    status = json.loads((run / "step_status.json").read_text(encoding="utf-8"))
    assert verification["status"] == "pass" and status["mode"] == "full"
    protocol = json.loads((run / "protocol_snapshot.json").read_text(encoding="utf-8"))
    summary = {r["model"]: r for r in tables["gate_summary"]}
    global_rows = {(r["model"], r["condition"]): r for r in tables["global"]}
    deltas = {(r["model"], r["condition"]): r for r in tables["paired_deltas"]}
    names = protocol["models"]
    gate_table = "\n".join(f"| {name} | {summary[name]['gate_mean_mean']:.4f} | {summary[name]['feature_l2_retention_mean']*100:.2f}% | {summary[name]['gate_fraction_below_005_mean']*100:.3f}% | {summary[name]['gate_spatial_std_per_channel_mean_mean']:.4f} |" for name in names)
    intervention_table = "\n".join(f"| {name} | {deltas[name,'bypass']['delta_global_sidewalk_iou_pp']:+.3f} | {deltas[name,'energy_uniform']['delta_global_sidewalk_iou_pp']:+.3f} | {deltas[name,'energy_shuffle_42']['delta_global_sidewalk_iou_pp']:+.3f} | {deltas[name,'energy_shuffle_43']['delta_global_sidewalk_iou_pp']:+.3f} | {deltas[name,'mixed_0_1']['delta_global_sidewalk_iou_pp']:+.3f} |" for name in names)
    originals = protocol["reference_models"] + names
    original_table = "\n".join(f"| {name} | {global_rows[name, 'original' if name in names else 'historical_reference']['global_miou_pct']:.3f} | {global_rows[name, 'original' if name in names else 'historical_reference']['global_sidewalk_iou_pct']:.3f} | {global_rows[name, 'original' if name in names else 'historical_reference']['author_boundary_mean_pct']:.3f} |" for name in originals)
    step2 = json.loads((cr.HERE / "step_02_2026-10-14/run_002/main_table.json").read_text(encoding="utf-8"))
    old = {r["model"]: r for r in step2}
    train_diff = global_rows["ENet+CA", "original"]["global_miou_pct"] - global_rows["ENet", "historical_reference"]["global_miou_pct"]
    val_diff = old["ENet+CA"]["miou_global_pct"] - old["ENet"]["miou_global_pct"]
    trained_checks = json.loads((run / "model_checks.json").read_text(encoding="utf-8"))
    error = max(r["max_matched_channel_energy_relative_error"] or 0 for r in trained_checks)
    text = f"""# 第四步：已有注意力权重诊断

第四步完成。18 个训练城市各固定抽取 3 张图，共 54 张；4 个含 CA 的历史模型各执行 10 种条件，另有 2 个历史基线参考，共 {verification['prediction_rows']} 条预测记录。未进行训练，未使用验证图调参。完整推理耗时约 {status['wall_seconds']/60:.1f} 分钟，包含数据、指标和校验，不能当作模型 FPS。

当前证据不支持将“CA 没起作用”作为既定结论：已有模型依赖原门控的空间分布；同时历史 ENet 与 ENet+CA 的验证集比较仍有训练条件混杂。下一步优先统一条件的 A/B 对照，暂不默认追加混合门控模块。

## 固定样本与操作范围

诊断规则见 [diagnostic_protocol_v1.json](../../diagnostic_protocol_v1.json)，完整样本身份见 [fixed_train_manifest.json](fixed_train_manifest.json)。城市与文件排序后使用 NumPy default_rng(20261014)，每城无放回抽取 3 张，不使用模型结果或标签决定抽样。54 张图在推理前固定；成功／失败均保留。

训练图用于机制线索检查，不是独立测试。每个城市等量抽样也不等于按整个训练集城市比例抽样；本步指标不可放进第二步验证精度主表。图片和标签保持已有 768×1024、ID 8 人行道、其他背景的规则。

含 CA 的模型为 ENet+CA、ENet-CBR、SBNet+CA、SBNet-CBR；参考为分别训练的 ENet、SBNet。名称指本地自实现模型，不替代官方实现的结果。

所有模型 strict 加载，处于 eval 模式，权重和 BN 运行统计固定；干预只在 CA 输出位置发生，不修改权重文件，也不训练或重新校准 BN。

## 门控没有大面积饱和

表中先逐图统计，再对 54 张图求平均。“近零”为预先固定的 g<0.05，仅是描述阈值。g 为两个方向 sigmoid 门控在每个通道的乘积。

| 模型 | 平均门控 g | CA 后特征 L2 范数保留 | g<0.05 占比 | 每通道空间标准差均值 |
|---|---:|---:|---:|---:|
{gate_table}

门控均值约 0.24，范数保留约 25%～26%，同时门控在空间上存在变化。这不等于信息只剩四分之一，也不能单凭小于 1 的门控判定机制失败。L2 范数与平方能量是不同数量；后面的能量匹配通过匹配每通道 L2 范数实现。

![门控与特征分布](gate_and_feature_distribution.png)

## 改动空间分布后的响应

下面为同一个已训练检查点、同样 54 张图中，相对于原始 CA 的全局人行道 IoU 差值，单位是百分点。

| 模型 | 直接旁路 | 保持每通道能量的均匀缩放 | 保持能量的空间打乱 42 | 保持能量的空间打乱 43 | 临时混合 α=0.1 |
|---|---:|---:|---:|---:|---:|
{intervention_table}

每通道能量匹配的最大相对误差为 {error:.3g}，明显低于预设容许值 2e-5。两个独立种子的打乱保留原门控值分布；能量匹配版本还匹配 CA 原输出每通道 L2 范数。因此下降不能只解释为移除了约 0.24 的整体缩放。

ENet+CA 在能量匹配打乱下，人行道 IoU 分别下降约 32.59、34.07 个百分点；50 张前景 IoU 下降，4 张不变。说明原门控位置分布在该已训练模型的预测中有实际作用，不能把该权重描述为“注意力完全不起作用”。

干预仍会改变特征均值、空间相关性和后续 BN／解码器输入分布；能量匹配没有消除所有分布变化。上述响应不证明 CA 从头训练一定提高泛化，也不证明它具有普遍的语义方向理解能力。不同模型之间下降的大小不能直接用来量化 CA 的独立贡献。

![全部干预差值](intervention_deltas.png)

边界列在 [paired_intervention_deltas.csv](paired_intervention_deltas.csv) 中使用原始／干预都具有非空边界并集的图像做配对差值，并报告有效对数。部分条件有效数为 53 而不是 54；图中边界差值是逐图配对均值，与全局人行道 IoU 的聚合方式不同。完整每种条件的 Boundary IoU 及有效数在 [global_intervention_table.csv](global_intervention_table.csv) 中。

## 混合门控目前的证据

将已训练原 CA 的输出临时改为 x×(0.9+0.1g)，四个模型的人行道 IoU 均下降。这项结果不支持把现有权重直接替换成弱门控作为部署改进，也不足以给 D/E 候选立项为“已经有效的创新”。

D/E 从头训练后是否有效仍未知。本步没有训练它们，也没有据推理干预宣布它们永远无效。为控制工作量，建议先完成 A/B 正式对照；只有出现可复现的问题及可检验的改进依据，才扩展候选。

## 训练子集和验证集的历史现象

| 已有权重 | 本步训练子集全局 mIoU % | 人行道 IoU % | 前景 Boundary IoU % |
|---|---:|---:|---:|
{original_table}

ENet+CA 相对 ENet 在这 54 张训练图上的全局 mIoU 为 {train_diff:+.3f} 个百分点，而第二步 500 张验证图为 {val_diff:+.3f} 个百分点。两个比较方向不同，可作为继续检查训练条件与泛化表现的线索。

训练城市、验证城市和样本构成不同，历史模型的权重衰减也不同。不能据这两个数单独确认过拟合，更不能据此证明权重衰减是下降原因。要验证衰减参数的因果作用，需要额外控制实验；目前优先排除这种混杂，不立即扩展完整因素组合。

## 可视化纠正

新热力图使用 mean_channels(a_vertical×a_horizontal)，并固定色标 0～1。旧脚本计算 mean_channels(a_vertical)×mean_channels(a_horizontal)，一般不等于实际门控的通道均值；相关误差已逐图保存。本步小型反例的两种值分别为 0.41 和 0.25。

旧图片和脚本均保留。注意力图的存在或色彩变化只能描述权重，不能单独证明任务增益；通道平均图看起来接近均匀也不能证明每个通道的门控没有空间变化。

例图按固定 manifest 的第 0、18、36 项保存，每个模型各 3 张，共 12 张，没有按效果挑选。以下为第一张 ENet+CA 例图：

![固定样本 ENet+CA](figures/ENet_plus_CA_00.png)

## 核验与可复用文件

- [analysis_verification.json](analysis_verification.json)：{verification['prediction_rows']} 条计数记录重建 42 行条件主表、36 行配对差值；216 条门控／特征统计。
- [model_checks.json](model_checks.json)：每张图的包装后原始 logits 与未包装模型逐位一致；干预间编码器特征相同；参数及 BN 缓冲量不变。
- [per_model](per_model/)：逐图预测计数与门控特征统计 CSV。
- [global_intervention_table.csv](global_intervention_table.csv)、[paired_intervention_deltas.csv](paired_intervention_deltas.csv)、[gate_feature_summary.csv](gate_feature_summary.csv)：可用于后续分析的汇总。
- [visual_data](visual_data/)：例图的原始预测、标签、门控数组；60 个保存预测的指标已从数组独立重算。
- [figures](figures/)：12 张固定样本例图。
- [diagnostic_source_exact.py](diagnostic_source_exact.py)：与本次执行源码逐字节一致的快照。[快照说明](snapshot_verification_note.json) 记录 Windows 文本模式的 LF/CRLF 转换；原文本快照保留，代码文本一致。当前入口已改为二进制保存快照，不影响推理公式。
- [integrity_after.json](integrity_after.json)：{integrity['original_files']} 个旧文件、{integrity['previous_step_files']} 个前两步文件及 108 个所选原始图像／标签，运行前后 SHA-256 一致。

smoke 的 126 条记录与完整运行中同样三张图的记录完全一致。诊断运算的矩形／零特征、能量匹配、打乱分布和热力图反例检查在 [operation_checks_001.json](../operation_checks_001.json) 中。没有删除或覆盖旧结果，没有外部上传或付费服务。

## 第五步如何安排

先使用第三步统一入口，在本机 RTX 4060 上做相同条件的 A/B 对照。首个种子设为 42，先各完成完整数据的一轮，调度器仍按 T_max=100，保留可继续的 epoch=1 检查点；根据真实一轮耗时估计 100 轮及三种子总时间，再分阶段推进。不会根据第 1 轮精度筛选模型。

正式目标是 A/B 的 42、43、44 三个配对种子，报告逐种子结果与均值／标准差。旧 15 模型的评估保留作历史参考，不全部重跑。作者形式 CA、混合门控和 CBR 消融按后续结果决定，当前不增加主线规模。

若统一条件后 CA 不下降，应把研究表述调整为注意力效果的可复现性、训练条件敏感性与精度／部署取舍；不要为维持“失效”构思而忽略结果。若稳定下降，再针对实现控制等提出可验证的机制假设。该路线与原课题的轻量分割和部署主题一致。

现在无需手动操作。可以回复“继续第五步，先测一轮耗时并启动统一对照”。进入长训练前需要电脑接电，并避免自动睡眠；具体耗时与运行范围在一轮实测后报告。
"""
    with (run / "REPORT.md").open("x", encoding="utf-8") as f:
        f.write(text)
    status.update({"status": "complete", "analysis_verified": True, "analysis_completed": timestamp(),
                   "report": "REPORT.md", "next_step": "Step 5: benchmark one full epoch per matched A/B, then stage paired training; no automatic mixed-gate expansion"})
    cr.json_write(run / "step_status.json", status, exclusive=False)
    cr.json_write(run / "progress.json", status, exclusive=False)
    print("STEP 4 COMPLETE: " + str(run / "REPORT.md"))


if __name__ == "__main__":
    main()
