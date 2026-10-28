# 基于轻量化语义分割的城市可通行区域识别与边缘部署研究

本科毕业设计，研究 Cityscapes 人行道二分类中轻量网络的注意力行为、受控改进与本机部署。语义人行道预测不等同于安全通行判断，目前不提供路径规划或避障能力。

本仓库同步代码、冻结协议、必要验证凭据和精简结果。数据集、权重、断点、逐轮日志、视频、ONNX 及个人论文材料保留在本地，不随 Git 上传；GitHub 不是这些大文件的备份。

## 当前进度

截至 2026-10-28，A/B/C 各三个配对种子（42、43、44），共 9 次 100 轮训练已完成且通过完整性核验。以下使用各自最早达到最高验证集 global mIoU 的同一权重，报告均值 ± 样本标准差，单位为百分数。

| 方案 | global mIoU | 人行道 IoU | 人行道 Boundary IoU |
|---|---:|---:|---:|
| A：ENet | 81.022 ± 0.010 | 64.197 ± 0.024 | 52.922 ± 0.168 |
| B：ENet + 现有 ReLU CA | 81.606 ± 0.106 | 65.301 ± 0.166 | 53.493 ± 0.165 |
| C：仅将 CA 激活改为 h-swish | 81.502 ± 0.247 | 65.109 ± 0.467 | 53.423 ± 0.560 |

B 相对 A 在三个种子上均提升；C 相对 B 的平均人行道 IoU 下降 0.193 个百分点，Boundary IoU 下降 0.070 个百分点，种子波动较大。现有结果不支持“CA 普遍失效”或“恢复原论文激活必然提升”。三次种子实验不构成显著性证明；验证集参与模型选择，不能称为独立测试集。

逐种子数值、配对差值与来源见 [results/abc_summary.json](results/abc_summary.json)。历史 15 模型比较存在训练条件差异，只作背景资料，不与此受控表混排。

![三个配对种子的完整结果](results/abc_three_seeds.png)

9 组冻结权重各 500 张的敏感性评价已经完成，并复现原主指标。B 相对 A 在有效标签像素口径和 6、13、26 像素边界下均保留三种子正收益。C 相对 B 的平均 Boundary IoU 差值在 6 像素时为 +0.053 个百分点，13 像素为 -0.006，26 像素为 -0.070；保留原主口径并报告全部宽度。逐城市与前景比例分组见 [敏感性结果](results/sensitivity_summary.json)。

B/C 六组权重在原固定 54 张训练图上各完成 10 种诊断干预。能量匹配后打乱门控仍使表现下降，但人行道特征保留比例没有低于背景；现有证据不支持普遍失效、门控塌缩或选择性压制人行道。推理干预可能改变解码器和 BN 输入分布，不能代替训练对照。见 [诊断结果](results/bc_diagnosis_summary.json)。

D/E 各完成种子 42 的 100 轮探索，并通过完整检查点、配对样本顺序、最终随机状态及历史文件保护核验。D 固定 α=0.1；E 学习 α，最终为 0.208797。以下为同一种子、按验证集 global mIoU 选定同一权重的结果，不能与上面的三种子均值混为一组。

| 方案 | 所选轮次 | global mIoU | 人行道 IoU | Boundary IoU |
|---|---:|---:|---:|---:|
| C：h-swish CA | 83 | 81.723 | 65.540 | 54.066 |
| D：固定混合系数 | 70 | 81.696 | 65.406 | 53.360 |
| E：学习混合系数 | 73 | 81.759 | 65.577 | 53.543 |

E 相对 C 的人行道 IoU 提升 0.037 个百分点，Boundary IoU 下降 0.523 个百分点；D 两项均低于 C。两者都未通过事先声明的“至少一项提高、另一项不下降”条件，因此结束这条探索分支，不补 D/E 的 43、44 种子。E 相对 D 的两项提升仅属单种子观察；不会改用 global mIoU 绕过原筛选条件。完整结果和决定见 [D/E 结果](results/de_seed42_summary.json)及[结果复核](results/de_review.json)。

![D/E 全部100轮与学习系数](results/de_exploration.png)

随后开展 A/B/C 的部署误差与延迟验证；保留 B 是因为它在现有三种子实验中的平均表现最好，工程比较统一使用种子 42。已有 A/B/C 各三张图的 CPU 预检通过，GPU 小样本也通过且实际节点确认为 CUDA。首次各 500 张 GPU 验收出现逐元素数值超限，完整失败记录已保留；标签与区域/边界指标差异均在原范围内。针对失败样本诊断后，保留独立 BatchNormalization 并关闭 ConvBN/ConvActivation 融合，重新完成各 500 张 GPU 验收，全部通过原来的数值与指标阈值。权重、数据和容差均未修改，实际图节点为 CUDA；原图与修正图分别留档。下一步开展 FP16 精度验证与独立性能测试，当前没有 FPS 结论。

[首次完整验收](results/deployment_cuda_fp32.json)、[数值诊断](results/fp32_numerical_diagnosis.json)和[修正后的完整验收](results/deployment_cuda_fp32_preserved_bn.json)分别记录失败、修正依据与复验结果。保留归一化的配置可能影响速度，后续实测后再作准确率与延迟取舍；不能将组合修改归因为单个优化器缺陷。

CPU 预检、GPU 并发保护和最新阶段入口见[部署准备结果](results/deployment_readiness.json)。真实边缘设备、长视频稳定性和本地自采标注验证尚未完成；当前平台为本机 RTX 4060 Laptop。

## 固定实验条件

[冻结协议 v1.2](experiments/attention_gate/protocol_v1_2.json)：Cityscapes train 2975 张 / val 500 张；输入 768×1024；原始标签 8 为人行道，其余映射背景，忽略区域另做敏感性分析；batch 4、保留尾批；100 轮；AdamW，lr 1e-4、weight decay 0.01；余弦调度；0.4 CE + 0.6 Dice；不使用预训练、增广、AMP 或早停；配对骨干初始化和样本顺序。Boundary IoU 使用 26 像素边界、逐图均值，双方均空时剔除并记录。

已验证环境为 Windows、Python 3.9、PyTorch 2.5.1+cu121、RTX 4060 Laptop 8GB。[环境快照](results/environment.json)记录实际版本，不是已经验证的全新环境安装清单。受控入口含 Windows 锁和防休眠操作，历史恢复需要本地数据、旧检查点和审计档案；不承诺 Linux 上逐位一致。

## 目录与运行

| 路径 | 用途 |
|---|---|
| `code/` | 历史模型、训练、评估、可视化和 Pipeline，保留原接口 |
| `code_sam/` | 自采图像和辅助标注工具；SAM2 环境及模型单独准备 |
| `experiments/attention_gate/` | 受控入口、指标、诊断、部署原型和冻结协议 |
| `results/` | 精简结果、环境及源码哈希 |
| `tools/` | 结果导出与仓库完整性检查 |
| `archive/figures_legacy/` | 原样保留的历史图表，不代表最新受控结果 |

Cityscapes 应按官方许可获取，放入 `dataset/cityscapes/leftImg8bit/{train,val}` 和 `dataset/cityscapes/gtFine/{train,val}`。原本机目录保持不变。

从项目根目录使用配置好的 Python（本机为 `E:\anaconda\envs\blind_road\python.exe`）：

```powershell
# 仅检查哈希、协议和结果一致性，不训练，不需要数据或 CUDA。
python tools/verify_repository.py

# 查看受控入口参数。
python experiments/attention_gate/common_runner_v2.py --help
python experiments/attention_gate/common_runner_c.py --help
python experiments/attention_gate/common_runner_mixed.py --help
python experiments/attention_gate/sidewalk_demo.py --help
python experiments/attention_gate/validate_deployment_fp32.py --help
python experiments/attention_gate/export_selected_fp32.py --help
python experiments/attention_gate/validate_deployment_fp32_preserved_bn.py --help

# 旧入口仅列出脚本或显示解析路径。
.\run.ps1 -List
.\run.ps1 -Script eval_main_table.py -DryRun
```

运行需指定新输出目录；恢复拒绝不匹配的数据、协议、源码或完整状态。历史监督脚本含本机路径与审计依赖，仅供追溯；新机器应使用基础训练入口并重新核验。旧训练可能自动恢复断点，旧评估可能覆盖同名输出，应先检查参数。SBNet 对应内部历史名称 `bisenetv2*`，见[命名与标签](code/NAMING_AND_LABELS.md)。海口自采数据尚未提供，工具存在不代表外部验证完成。

## 来源与记录

CA 参考实现来自 [CoordAttention](https://github.com/houqb/CoordAttention/tree/a3f3490ceba779a8443a231d0f792866bd24436d)，版本与来源保留在 `author_reference/source.json`。混合门控是任务适配候选，不宣称残差门控公式原创。后续保留负结果，同时与 A/B/C 比较。

2026-10-26 的文献更新核实了 [GCNet（CVPR 2025）](https://github.com/gyyang23/GCNet)的官方配置和测速脚本，以及 [TBMSNet（Scientific Reports 2026）](https://www.nature.com/articles/s41598-026-48759-x)的三分支方法与复现条件。当前借鉴部署及边界分析方法，优先完成已冻结的对照实验，不将重型新网络加入必做清单。来源、阅读限制和可行性取舍见 [研究记录](results/research_review_20261026.json)。

个人日志统一写入桌面的《工作记录.docx》。整理前的哈希清单、备份和完整实验输出留在原本机位置。Git 整理不改变冻结实验源码、权重或原结果。
