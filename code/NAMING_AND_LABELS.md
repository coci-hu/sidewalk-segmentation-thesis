# 命名与标签口径说明（论文写作前必读）

> 本文档是代码、日志、图表与论文之间的**唯一口径依据**。修改任何显示名称或标签解释时，必须同步更新本文件。

---

## 1. 标签口径：本文的"可通行区域"是什么

**结论：本文的类别 1 是 Cityscapes 的 `sidewalk`（人行道，标签 id = 8），不是盲道（触觉铺装）。**

- Cityscapes 的类别体系里**没有**"盲道 / 触觉铺装 / tactile paving"这一类别。
- 本文全部训练、评估、部署实验均以 `label == 8` 生成二分类掩码，前景即人行道区域。
- 论文允许的表述是：**"以人行道区域作为视障人群可通行区域的代理标签"**。
- 论文**不允许**把类别 1 直接称作"盲道"，因为那会声称一个数据集并不包含的类别。

### 代码中的对应位置

| 位置 | 说明 |
|---|---|
| `code/eval_common.py` 的 `SIDEWALK_ID = 8` | 全局标签定义，id 8 |
| 各 `train_*.py` 中的 `SIDEWALK_ID = 8` | 独立训练脚本内的同名常量 |
| CSV/图表的 `iou_fg`、`biou_*_fg` | 指类别 1，即人行道 / 可通行区域 |
| `code_sam/labelme_json_to_mask.py` 的 `TRAVERSABLE_KEYWORDS` | `["blind","sidewalk","tactile","盲道","人行道","可通行区域"]`，兼容多种标注写法 |

### 代码中保留旧措辞的标识符

以下 **Python 标识符**为兼容历史脚本未改名，仅作内部名称，**不进入论文**：

- 类名 `CityscapesBlindDataset`、`HaikouBlindDataset`
- 目录名 `dataset/haikou_blind/`、项目名 `blind_road_project`
- `code/visualize_prediction.py` 中变量名含 `blind`

> 若答辩前需要彻底统一，可再做一次标识符级重命名，但这不影响论文正确性。

---

## 2. 架构命名：SBNet 是什么

**结论：论文中的"SBNet"是自写的精简双分支网络，不是官方 BiSeNetV2。**

| 项目 | 本文 SBNet | 官方 BiSeNetV2 |
|---|---|---|
| 结构 | detail(3 conv) + semantic(4 conv) + concat + head | 含 ARM / CE / BGA 等完整模块 |
| 实测参数量 | **0.9446M**（基线） | M 量级 |
| 来源 | 本项目自写 | Yu et al., *BiSeNet V2*, IJCV 2021 |

论文中的写法建议：

> "本文采用一种自实现的精简双分支网络（Slim Bilateral Network, SBNet）作为容量充足的骨干，
> 其结构由浅层细节分支、深层语义分支与融合头组成，在 Cityscapes 二分类任务下参数量为 0.94M。
> 需说明的是，该网络并非 Yu 等人提出的官方 BiSeNetV2（后者包含 ARM/CE/BGA 等模块，参数量为 M 量级），
> 因此本文结果不宜与文献中的 BiSeNetV2 数值直接比较。"

### 显示名对照

| 论文/图表显示名 | 代码注册名 | 模块文件 | 权重文件 | 实测参数量 |
|---|---|---|---|---|
| SBNet | `bisenetv2` | `model_bisenetv2.py` | `best_bisenet_100epoch_bin.pth` | 0.9446M |
| SBNet+CA | `bisenetv2_ca` | `model_bisenetv2_ca.py` | `best_bisenet_ca_100epoch_bin.pth` | 0.9513M |
| SBNet+L-ASPP | `bisenetv2_laspp` | `model_bisenetv2_laspp.py` | `bisenetv2_laspp_best.pth` | 1.2812M |
| SBNet-CBR | `bisenetv2_cbr` | `model_bisenetv2_cbr.py` | `bisenetv2_cbr_best.pth` | 1.3211M |

> 权重文件名保留历史 `bisenetv2_*` 前缀是为了不破坏已有实验产物，**仅文件名未改，不改变任何网络结构**。

---

## 3. 全部模型的实测参数量（论文表格请以此为准）

由 `blind_road` 环境实测（`sum(p.numel() for p in model.parameters())`），非引用文档：

| 显示名 | 实测参数量 (M) | FPS (RTX 4060, 768×1024) | 权重文件 |
|---|---|---|---|
| ENet | 0.2636 | 120.7 | `enet_best.pth` |
| ENet+CA | 0.2670 | 116.7 | `enet_ca_best.pth` |
| ENet+BA | 0.2720 | 113.6 | `enet_ba_best.pth` |
| ENet-DBA | 0.2754 | 111.8 | `enet_dba_best.pth` |
| ENet+L-ASPP | 0.3500 | 105.3 | `enet_laspp_best.pth` |
| ENet-CBR | 0.3618 | 103.6 | `enet_cbr_best.pth` |
| SBNet | 0.9446 | 91.0 | `best_bisenet_100epoch_bin.pth` |
| SBNet+CA | 0.9513 | 92.4 | `best_bisenet_ca_100epoch_bin.pth` |
| SBNet+L-ASPP | 1.2812 | 87.0 | `bisenetv2_laspp_best.pth` |
| SBNet-CBR | 1.3211 | 86.4 | `bisenetv2_cbr_best.pth` |
| Fast-SCNN | 1.4660 | 279.1 | `fastscnn_best.pth` |
| STDC-Seg | 2.2461 | 93.9 | `best_stdc_seg_100epoch_bin.pth` |
| PP-LiteSeg-T | 2.7143 | 163.5 | `best_ppliteseg_100epoch_bin.pth` |
| SegFormer-B0 | 4.3009 | 25.4 | `segformer_best.pth` |

FPS 与 FLOPs 取自 `train_logs/benchmark_fps.csv`（`benchmark_fps.py`，warmup 50 / test 200 iters，GPU 空闲时测量，2026-09-13 17:40 重跑）。

> ⚠️ **本表的 FPS 列已于 2026-09-13 17:40 全表刷新**。此前本表沿用的是更早一次测量的旧值，
> 与 `benchmark_fps.csv` 在 14 个模型中有 13 个不一致（最大偏差 STDC-Seg 100.6 → 93.9、
> ENet-CBR 99.5 → 103.6）。论文引用请以 `benchmark_fps.csv` 和本表为准。
>
> ⚠️ **不要引用 `deploy_benchmark.csv` 里的 `pt_fps` / `ort_fp32_fps`（GPU 列）做速度对比**：
> 它与 `benchmark_fps.csv` 口径不同（不同测量循环），且 GPU-FP32 与 CPU-INT8 之间不可比。
> 该表现在可用的是**精度列**与派生列 `int8_speedup_vs_cpu_fp32`
> （2026-09-13 18:14 重跑，结果见 §5.1）。

---

## 4. 指标口径

### 4.1 mIoU 的两种统计方式

| 口径 | 实现位置 | 说明 |
|---|---|---|
| 逐图平均 | 各 `train_*.py` 的 `compute_mIoU` | 训练 CSV 的 `val_mIoU` 列；每张图先算 IoU 再对图求平均 |
| 全局混淆矩阵 | `eval_common.py` 的 `mious_from_confusion` | 评估脚本使用；先累积混淆矩阵再算 IoU |

两者相差 1–3 个百分点。**论文所有表格必须统一为其中一种**，推荐用全局混淆矩阵口径（更接近分割领域惯例）。

**执行方式**：运行 `code/eval_main_table.py`，它会**一次遍历同时算出两种口径**，并把 `miou_global`、
`miou_per_image` 与两者的差 `miou_gap` 全部写进 `train_logs/eval_main_table.csv`。
论文主表引用 `miou_global` 列，表注写明"全局混淆矩阵口径 + 最优轮"。
这样既能统一口径，又能在正文里如实说明"两种口径相差 X 个百分点"，反而是加分项。

### 4.2 "最优轮" vs "最后一轮"

差异最大可达 6.5 个百分点（SBNet+L-ASPP：最优轮 ep39 = 76.84%，最后一轮 = 76.36%）。
论文需二选一并全文统一，推荐**最优轮**，但必须同时给出对应轮次。

### 4.3 Boundary IoU 的两种公式

`eval_boundary_iou.py` 会同时输出两种口径：

| CSV 列 | 口径 |
|---|---|
| `biou_official*` | 官方 Cheng et al. CVPR 2021 公式（含 1 像素 border padding，每图先算再平均） |
| `biou_legacy*` | 本项目早期公式（无 padding，全局累积），保留仅作对照 |

论文引用 `biou_official`，并说明边界带宽度 `d = round(0.02 × √(768²+1024²)) = 26 px`。

---

## 5. 已废弃 / 不要引用的产物

| 产物 | 原因 |
|---|---|
| `deploy_benchmark.csv` 的 `pt_fps` / `ort_fp32_fps`（GPU 列） | 与 `benchmark_fps.csv` 口径不同，不能与 CPU-INT8 列直接相比（见 §3 注） |
| `enet_ba` / `enet_cbr` / `bisenetv2_cbr` 日志中的"总训练时长 0.0 分钟" | 最后一次运行仅用于补导出 CSV，主循环为空 |
| `archive/generated_configs_legacy/enet_ca_config.yaml` | Trainer 自动生成的产物，含 `!!python/tuple`，safe_load 不可读（`config.py` 已修复） |
| `figures/` 下的 4 张 PNG | 早期产物，与 `train_logs/` 下的图口径不一致，论文统一用 `train_logs/`。现归档位置为 `archive/figures_legacy/` |
| `train_logs/attention_BiSeNetV2-CBR.png` | 旧命名重复品，现行版本是同内容的 `attention_SBNet-CBR.png`。现归档位置为 `archive/figures_legacy/` |

### 5.1 部署量化的正确结论（2026-09-13 18:14 重跑后确立）

修复"FP32 用 GPU、INT8 用 CPU"的错误对比口径后，用新列 `int8_speedup_vs_cpu_fp32` 得到：

| 结论 | 数据 | 可否写进论文 |
|---|---|---|
| 精度损失小 | 最大 −0.82 pp（SBNet+L-ASPP）；ENet 系列 ≤0.14 pp；ENet-CBR 反而 +0.02 pp | ✅ 可写 |
| 体积压缩明显 | ONNX 压缩 50.6%（ENet）～ 74.3%（SBNet） | ✅ 可写 |
| **动态量化提速** | **6 个模型全部 < 1（0.51–0.74），即 CPU 上反而更慢** | ❌ **不可写"提速"** |

原因：`quantize_dynamic` + 仅量化 `Conv`/`MatMul` 对以卷积为主的 CNN 并非最优路径（官方建议 CNN 用 `quantize_static` + 校准集）。
若论文需要"量化提速"这个卖点，必须改用静态量化重做（约 1–2 小时），否则只写精度与体积。
