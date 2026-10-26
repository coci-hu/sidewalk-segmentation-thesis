# Pipeline 使用教程

本项目在 `code/pipeline/` 下搭建了统一可扩展的语义分割训练 Pipeline，所有超参、数据集、模型配置均通过 YAML 驱动，无需改动代码即可切换实验。

---

## 一、快速开始

### 1. 单卡训练（最常用）

```bash
cd code
conda activate blind_road

# 使用已有配置启动 BiSeNetV2+CA 实验
python pipeline/train_main.py --config pipeline/configs/bisenetv2_ca.yaml
```

### 2. 切换模型

编辑 `pipeline/configs/single_gpu.yaml`，修改 `model.name` 即可：

```yaml
model:
  name: enet          # 全部可选值见下方「可选模型名」表（共 16 个）
  num_classes: 2
```

```bash
python pipeline/train_main.py --config pipeline/configs/single_gpu.yaml
```

### 3. 梯度累加 + AMP（大模型用）

```bash
python pipeline/train_main.py --config pipeline/configs/accum_amp.yaml
```

### 4. DDP 多卡训练

```bash
# Linux（2卡示例）
torchrun --nproc_per_node=2 pipeline/train_main.py --config pipeline/configs/ddp.yaml

# Windows（需用 gloo 后端，YAML 中已配置 ddp_backend: gloo）
torchrun --nproc_per_node=2 pipeline/train_main.py --config pipeline/configs/ddp.yaml
```

---

## 二、配置文件说明

YAML 配置分四段：`data`、`model`、`train`、`log`。

### data 段 — 数据集

```yaml
data:
  dataset: cityscapes       # cityscapes | haikou
  root_dir: ../dataset/cityscapes
  crop_size: [768, 1024]    # 输入尺寸 [高, 宽]
  num_workers: 0            # DataLoader worker 数
  pin_memory: false         # 内存锁页
  persistent_workers: false
```

- `dataset: cityscapes` 使用 Cityscapes 二分类数据集
- `dataset: haikou` 使用海口盲道自建数据集
- 两个数据集接口完全一致，一键切换

### model 段 — 模型

```yaml
model:
  name: bisenetv2_ca        # 模型注册名
  num_classes: 2            # 分类数
  pretrained: null          # 预训练权重路径，null=从头训练
  freeze_backbone: false    # 是否冻结骨干网络
```

**可选模型名**（`model_zoo.py` 共注册 **16** 个；参数量为 2026-09-13 实测值）：

| name | 论文显示名 | 参数量(实测) |
|---|---|---|
| `bisenetv2` | SBNet | 0.9446M |
| `bisenetv2_ca` | SBNet+CA | 0.9513M |
| `bisenetv2_laspp` | SBNet+L-ASPP | 1.2812M |
| `bisenetv2_cbr` | SBNet-CBR | 1.3211M |
| `enet` | ENet | 0.2636M |
| `enet_ca` | ENet+CA | 0.2670M |
| `enet_ba` | ENet+BA | 0.2720M |
| `enet_dba` | ENet-DBA | 0.2754M |
| `enet_laspp` | ENet+L-ASPP | 0.3500M |
| `enet_cbr` | ENet-CBR | 0.3618M |
| `fastscnn` | Fast-SCNN | 1.4660M |
| `stdc_seg` | STDC-Seg | 2.2461M |
| `ppliteseg` | PP-LiteSeg-T | 2.7143M |
| `segformer_b0` | SegFormer-B0 | 4.3009M |
| `unet` | U-Net | 31.0377M |
| `deeplabv3plus` | DeepLabV3+ | 40.3473M |

> **命名口径**：`bisenetv2*` 是自写的**精简双分支网络**，并非官方 BiSeNetV2；论文与图表统一称
> **SBNet**。详见 `code/NAMING_AND_LABELS.md`。
>
> ⚠️ **Trainer 的限制**：CBR / BA / DBA 三类模型的 `forward` 返回 `(seg_out, bdy_loss)` 元组，
> 而 `pipeline/trainer.py` 的损失计算只接受单个 logits 张量，因此**这三个系列必须用对应的
> `code/train_*.py` 独立脚本训练**，不能用本 Pipeline 启动。本目录下的 `*_cbr.yaml` /
> `enet_ba.yaml` / `enet_dba.yaml` 仅作超参记录与对照。

### train 段 — 训练超参

```yaml
train:
  epochs: 100
  batch_size: 4             # 单步 batch
  accum_steps: 1            # 梯度累加步数（等效 batch = batch_size × accum_steps）
  lr: 0.0001                # 初始学习率
  weight_decay: 0.0001      # 权重衰减
  optimizer: adamw           # adamw | sgd
  scheduler: cosine          # cosine | step | none
  t_max: 100                 # CosineAnnealingLR 周期
  eta_min: 0.000001          # 最小学习率
  ce_weight: 0.4             # CrossEntropy 权重
  dice_weight: 0.6           # Dice Loss 权重
  grad_clip: 0.0             # 梯度裁剪（0=不裁剪）
  amp: false                 # AMP 混合精度
  compile_model: false       # torch.compile 加速（PyTorch 2.0+）
  ddp: false                 # DDP 多卡
  ddp_backend: nccl          # nccl(Linux) | gloo(Windows)
```

**显存不足时的配置示例**（U-Net / SegFormer）：

```yaml
train:
  batch_size: 2
  accum_steps: 2            # 等效 batch=4
  amp: true                 # 开启混合精度
  grad_clip: 1.0            # AMP 下建议开启梯度裁剪
```

### log 段 — 输出路径

```yaml
log:
  weight_dir: ../weights          # 权重保存目录
  log_dir: ../train_logs          # 训练日志目录
  csv_dir: ../train_logs          # CSV 指标目录（与日志同目录）
  experiment_name: bisenetv2_ca   # 实验名称（文件名前缀）
  save_interval: 10               # 每 N epoch 保存断点
```

输出文件命名规则：
- 最优权重：`{weight_dir}/{experiment_name}_best.pth`
- 断点续训：`{weight_dir}/{experiment_name}_ckpt.pth`
- 中断保存：`{weight_dir}/{experiment_name}_interrupt.pth`
- CSV 指标：`{csv_dir}/{experiment_name}_metrics.csv`
- 实验配置：`{log_dir}/{experiment_name}_config.yaml`

---

## 三、如何新增模型接入 Pipeline

### 步骤 1：创建模型文件

在 `code/` 下新建 `model_xxx.py`，定义模型类和构建函数：

```python
# code/model_xxx.py
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # ... 网络定义

    def forward(self, x):
        # ... 前向传播
        return out

def get_my_model(num_classes, device):
    return MyModel(num_classes=num_classes).to(device)
```

### 步骤 2：注册到 model_zoo

编辑 `code/pipeline/model_zoo.py`，添加构建函数和注册项：

```python
def _build_my_model(num_classes, device):
    from model_xxx import MyModel
    return MyModel(num_classes=num_classes).to(device)

MODEL_REGISTRY = {
    # ... 已有模型
    "my_model": _build_my_model,    # 新增
}
```

### 步骤 3：创建 YAML 配置

```yaml
# code/pipeline/configs/my_model.yaml
data:
  dataset: cityscapes
  root_dir: ../dataset/cityscapes
  crop_size: [768, 1024]
  num_workers: 0
  pin_memory: false
  persistent_workers: false

model:
  name: my_model        # 与 MODEL_REGISTRY 的 key 一致
  num_classes: 2
  pretrained: null
  freeze_backbone: false

train:
  epochs: 100
  batch_size: 4
  accum_steps: 1
  lr: 0.0001
  # ... 其余超参

log:
  weight_dir: ../weights
  log_dir: ../train_logs
  csv_dir: ../train_logs
  experiment_name: my_model
  save_interval: 10
```

### 步骤 4：启动训练

```bash
python pipeline/train_main.py --config pipeline/configs/my_model.yaml
```

---

## 四、如何将 CA 注意力插入任意网络

CA 模块已抽离为通用组件，位于 `code/components/ca_attention.py`。

### 方式 1：在模型定义中直接插入

```python
from components.ca_attention import CoordAtt

class MyModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = SomeBackbone()
        # 在骨干输出后插入 CA，不改变通道数和空间尺寸
        self.ca = CoordAtt(channels=256, reduction=32)
        self.head = SegHead(256, num_classes)

    def forward(self, x):
        feat = self.backbone(x)    # (B, 256, H, W)
        feat = self.ca(feat)       # (B, 256, H, W) — 形状不变
        out = self.head(feat)
        return out
```

### 方式 2：使用通用插入函数

```python
from components.ca_examples import insert_ca_after_layer

model = get_bisenetv2_2class(device)
# 在 semantic_branch 输出后自动插入 CA
insert_ca_after_layer(model, 'semantic_branch', channels=256, reduction=32)
```

### 方式 3：向已有模型类添加 CA 支持

```python
class MyModelWithCA(MyModel):
    def __init__(self, num_classes=2, use_ca=True):
        super().__init__(num_classes)
        self.use_ca = use_ca
        if use_ca:
            self.ca = CoordAtt(channels=256, reduction=32)

    def forward(self, x):
        feat = self.backbone(x)
        if self.use_ca:
            feat = self.ca(feat)
        out = self.head(feat)
        return out
```

### 插入原则

1. **位置**：CA 放在"特征提取完毕、融合/上采样之前"效果最好
2. **通道数**：CA 不改变输入输出形状，`channels` 设为该层输出通道数
3. **reduction**：默认 32，通道数少时可适当减小（如 16）

---

## 五、旧实验参数迁移 YAML 方法

旧版独立脚本中的超参与 YAML 配置的对应关系：

| 旧脚本变量 | YAML 配置路径 |
|---|---|
| `CROP_SIZE` | `data.crop_size` |
| `BATCH_SIZE` | `train.batch_size` |
| `ACCUM_STEPS` | `train.accum_steps` |
| `EPOCHS` | `train.epochs` |
| `LR_INIT` | `train.lr` |
| `NUM_CLASSES` | `model.num_classes` |
| `WEIGHT_SAVE_PATH` | `log.weight_dir` + `log.experiment_name` |
| `CSV_LOG_PATH` | `log.csv_dir` + `log.experiment_name` |

**示例**：旧版 `train_enet.py` 迁移为 YAML

```yaml
# 旧版: CROP_SIZE=(768,1024), BATCH_SIZE=4, ACCUM_STEPS=1, LR=1e-4
# 迁移为:
data:
  crop_size: [768, 1024]
train:
  batch_size: 4
  accum_steps: 1
  lr: 0.0001
log:
  experiment_name: enet
```

---

## 六、输出兼容性

Pipeline 生成的 CSV 格式与旧版脚本完全一致：

```csv
epoch,train_loss,val_loss,val_mIoU
1,0.6521,0.5432,0.3245
2,0.5832,0.4891,0.3876
...
```

可直接用旧版 `plot_train_log.py` 和 `plot_baseline_comparison.py` 绘图。

注意：Pipeline 导出的 CSV 只有 4 列（无 pixel_acc），而独立脚本的 CSV 有 5 列，
两者混用会导致绘图脚本读出多余列。论文定稿前需统一列结构。

---

## 七、断点续训

Pipeline 自动检测断点文件 `{experiment_name}_ckpt.pth`，存在则从上次中断处继续训练。

手动指定预训练权重：

```yaml
model:
  pretrained: ../weights/enet_best.pth    # 加载已有权重继续训练
```

---

## 八、项目目录结构

```
blind_road_project/
├── code/                          # 代码目录
│   ├── components/                # ★ 通用组件
│   │   ├── ca_attention.py        #   CA 坐标注意力模块
│   │   └── ca_examples.py         #   CA 插入示例
│   ├── pipeline/                  # ★ 统一训练 Pipeline
│   │   ├── config.py              #   YAML 配置加载
│   │   ├── data_module.py         #   统一 DataLoader
│   │   ├── model_zoo.py           #   模型注册（16个模型）
│   │   ├── trainer.py             #   通用训练器
│   │   ├── train_main.py          #   统一入口
│   │   └── configs/               #   YAML 配置文件（共 11 个）
│   │       ├── single_gpu.yaml        # 单卡通用水晶模板
│   │       ├── ddp.yaml               # DDP 多卡
│   │       ├── accum_amp.yaml         # 梯度累加 + AMP
│   │       ├── bisenetv2_ca.yaml      # SBNet+CA
│   │       ├── bisenetv2_laspp.yaml   # SBNet+L-ASPP
│   │       ├── bisenetv2_cbr.yaml     # SBNet-CBR
│   │       ├── enet_ca.yaml           # ENet+CA
│   │       ├── enet_ba.yaml           # ENet+BA
│   │       ├── enet_dba.yaml          # ENet-DBA
│   │       ├── enet_laspp.yaml        # ENet+L-ASPP
│   │       └── enet_cbr.yaml          # ENet-CBR
│   ├── model_*.py                 # 各模型定义
│   ├── train_*.py                 # 旧版独立训练脚本（保留不动）
│   ├── metrics.py                 # mIoU / Pixel Acc
│   └── cityscapes_dataset.py      # 数据集类
├── weights/                       # 所有模型权重（统一存放）
├── train_logs/                    # 所有训练日志和CSV（统一存放）
├── dataset/                       # 数据集
└── code_sam/                      # SAM2 标注工具
```
