"""Create the verified Step 2 report and a standalone scientific comparison plot."""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = Path(args.run).resolve()
    if not run.is_relative_to(HERE):
        raise RuntimeError("Run must be inside the experiment directory")
    status = json.loads((run / "step_status.json").read_text(encoding="utf-8"))
    verified = json.loads((run / "analysis_verification.json").read_text(encoding="utf-8"))
    metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    rows = json.loads((run / "main_table.json").read_text(encoding="utf-8"))
    differences = json.loads((run / "comparison_deltas.json").read_text(encoding="utf-8"))
    assert status["mode"] == "full" and status["status"] == "complete" and verified["status"] == "pass"
    report = run / "REPORT.md"
    plot = run / "unified_metrics.png"
    if report.exists() or plot.exists():
        raise RuntimeError("Refusing to overwrite an existing report or plot")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ordered = sorted(rows, key=lambda row: row["iou_sidewalk_global_pct"])
    positions = list(range(len(ordered)))
    names = [row["model"] for row in ordered]
    colors = ["#4e79a7" if name.startswith("ENet") else "#59a14f" if name.startswith("SBNet") else "#92979c" for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(13, 8.5), sharey=True)
    for axis, field, title in zip(axes, ["iou_sidewalk_global_pct", "boundary_iou_author_fg_pct"], ["Sidewalk IoU", "Author Boundary IoU (foreground)"]):
        values = [row[field] for row in ordered]
        axis.barh(positions, values, color=colors, height=0.65)
        axis.set_xlabel("Percent")
        axis.set_title(title)
        axis.set_xlim(0, max(values) + 7)
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
        for y, value in zip(positions, values):
            axis.text(value + 0.35, y, f"{value:.2f}", va="center", fontsize=9)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(names)
    fig.suptitle("Unified evaluation of saved historical best weights", fontsize=14)
    fig.text(0.5, 0.02, "500 Cityscapes val images | 768 x 1024 | single historical runs; training settings differ", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(plot, dpi=180)
    plt.close(fig)

    lines = ["# 第二步现有权重统一评估", "",
             "15 个已完成模型的已保存最佳权重已在全部 500 张 Cityscapes 验证图上评估，共 7,500 次预测。旧权重按历史训练指标选择；本次统一的是评估方式，不是训练条件。U-Net 不进入主表。", "",
             "## 本次运行", "",
             f"完成时间为 {status['finished']}，平台为 {metadata['gpu']}，输入 768×1024、batch=1、PyTorch FP32。正式运行耗时约 {status['wall_seconds'] / 60:.1f} 分钟，包含预处理、指标计算和文件核验，不能转换成模型 FPS。", "",
             "前景为 Cityscapes 标签 ID 8，人行道作为可通行区域的代理标签；其他标签按既有协议并入背景。验证集用于历史模型选择，不称独立测试集。", "",
             "## 统一主表", "",
             "| 模型 | 参数 M | 全局 mIoU % | 全局人行道 IoU % | 作者前景 Boundary IoU % | 边界有效图数 | 历史报告最佳轮次 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in sorted(rows, key=lambda row: -row["iou_sidewalk_global_pct"]):
        lines.append(f"| {row['model']} | {row['params_m']:.4f} | {row['miou_global_pct']:.4f} | {row['iou_sidewalk_global_pct']:.4f} | {row['boundary_iou_author_fg_pct']:.4f} | {row['boundary_valid_images']} | {row['historical_reported_best_epoch']} |")
    lines += ["", "参数量本次直接由模型统计。轮次从历史训练 CSV 推断，纯 state_dict 未嵌入轮次；不是按本次全局指标重新选取的最优轮。", "",
              "![统一指标对比](unified_metrics.png)", "",
              "## 旧权重之间的差异", "",
              "差值单位为百分点。这些是已有权重的观察性比较，尚不能归因为单个模块，也不能用单次结果判断训练波动。", "",
              "| 模型 | 参照 | mIoU 差值 | 人行道 IoU 差值 | Boundary IoU 差值 |",
              "|---|---|---:|---:|---:|"]
    for row in differences:
        lines.append(f"| {row['model']} | {row['reference']} | {row['miou_delta_pp']:+.4f} | {row['sidewalk_iou_delta_pp']:+.4f} | {row['boundary_iou_delta_pp']:+.4f} |")
    lookup = {row["model"]: row for row in rows}
    enet = lookup["ENet"]
    enet_ca = lookup["ENet+CA"]
    laspp = lookup["SBNet+L-ASPP"]
    cbr = lookup["SBNet-CBR"]
    lines += ["", "## 对后续研究的意义", "",
              f"ENet+CA 相比 ENet 的全局 mIoU 差值为 {enet_ca['miou_global_pct'] - enet['miou_global_pct']:+.4f} 个百分点。但第一步已查到旧权重的训练权重衰减分别为 0.0001 和 0.01，不能把这项差异单独归因于 CA。下一步先建立相同条件的训练入口。", "",
              f"SBNet-CBR 相比 SBNet+L-ASPP 的全局 mIoU 差值为 {cbr['miou_global_pct'] - laspp['miou_global_pct']:+.4f} 个百分点，作者前景 Boundary IoU 差值为 {cbr['boundary_iou_author_fg_pct'] - laspp['boundary_iou_author_fg_pct']:+.4f} 个百分点。该结果用于安排必要消融，尚未分离 CA 与 BGFR 的贡献。", "",
              "新的 Boundary IoU 采用两个边界带交并比，边界宽度为对角线的 0.02，即 26 像素，补一像素零边框。逐图计算后平均；双方空边界的图像排除，并报告有效数量。旧表中标为官方的自定义公式不与新值混用。[作者定义](https://gist.github.com/bowenc0221/71f7a02afee92646ca05efeeb14d687d)", "",
              "分城市结果是验证集内部的分组分析，不是新增数据集或独立跨域测试。各城样本量为 frankfurt 267、lindau 59、munster 174；全局主表由全部像素计数计算，不平均三个城市的 IoU。", "",
              "## 完整性与复核", "",
              "181 个原代码、权重和实验产物在运行前后 SHA-256 一致。500 张输入原图和标签的大小、修改时间保持不变。15 个模型都严格加载；每个模型 500 条逐图记录，主表与分城市表已从逐图计数独立重算校验。", "",
              "初始无 RGB 缓存的运行保留在 `../run_001`。优化后正式结果在本目录；ENet 和 ENet+CA 各 500 张的混淆计数及边界计数与原读取方式完全一致。内存缓存只改变读取方式，没有改变输入值或模型。", "",
              "## 结果文件", "",
              "| 文件 | 用途 |", "|---|---|",
              "| [main_table.csv](main_table.csv) | 15 个模型的统一主表 |",
              "| [per_city_table.csv](per_city_table.csv) | 45 行分城市结果 |",
              "| [comparison_deltas.csv](comparison_deltas.csv) | 指定模型与参照的差值 |",
              "| [per_model](per_model/) | 每模型逐图 CSV、汇总和权重来源 |",
              "| [sample_manifest.json](sample_manifest.json) | 500 张图像与标签的文件身份 |",
              "| [run_metadata.json](run_metadata.json) | 协议、软件、硬件和代码校验值 |",
              "| [analysis_verification.json](analysis_verification.json) | 从计数独立重建结果的校验 |",
              "| [integrity_after.json](integrity_after.json) | 原文件完整性检查 |",
              "| [step_status.json](step_status.json) | 第二步完成状态 |", "",
              "## 用户下一步", "",
              "本步无需用户安装软件、购买设备、注册账号或手动处理结果。第二步已完成且校验通过，可以回复“继续第三步，检查并统一训练入口”。第三步先做参数、初始化、梯度、保存恢复和小批次检查，不直接启动 100 轮重复训练。", "",
              "若后续看到路径缺失、权重装配错误或显存报错，保留报错并发回；不要删除或改名已有结果。", ""]
    with report.open("x", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(str(report))


if __name__ == "__main__":
    main()
