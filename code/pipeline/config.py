"""
pipeline/config.py
YAML 配置驱动实验，所有超参、数据集、模型、训练开关全部写 YAML。

用法：
    from pipeline.config import Config
    cfg = Config.from_yaml("pipeline/configs/bisenetv2_ca.yaml")
    print(cfg.model.name, cfg.train.batch_size)
"""

import yaml
import os
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from project_paths import resolve_code_path


@dataclass
class DataConfig:
    """数据集配置"""
    dataset: str = "cityscapes"          # cityscapes | haikou
    root_dir: str = "../dataset/cityscapes"
    crop_size: Tuple[int, int] = (768, 1024)
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False


@dataclass
class ModelConfig:
    """模型配置"""
    name: str = "bisenetv2"              # 模型名称，对应 model_zoo 注册名
    num_classes: int = 2
    pretrained: Optional[str] = None     # 预训练权重路径，None 表示从头训练
    freeze_backbone: bool = False        # 是否冻结骨干
    # 辅助边界损失权重：仅独立训练脚本（train_*_cbr/ba/dba.py）使用。
    # 注意 pipeline/trainer.py 目前不支持返回 (seg_out, bdy_loss) 元组的模型，
    # 因此 CBR / BA / DBA 系列必须用对应独立脚本训练。
    boundary_weight: float = 0.5


@dataclass
class TrainConfig:
    """训练配置"""
    epochs: int = 100
    batch_size: int = 4
    accum_steps: int = 1                 # 梯度累加步数，等效 batch = batch_size * accum_steps
    lr: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"             # adamw | sgd
    scheduler: str = "cosine"            # cosine | step | none
    t_max: int = 100                     # CosineAnnealingLR 周期
    eta_min: float = 1e-6                # CosineAnnealingLR 最小学习率

    # 损失函数权重
    ce_weight: float = 0.4               # CrossEntropy 权重
    dice_weight: float = 0.6             # Dice Loss 权重

    # 梯度裁剪
    grad_clip: float = 0.0               # 0 表示不裁剪

    # AMP 混合精度
    amp: bool = False

    # torch.compile 加速（PyTorch 2.0+）
    compile_model: bool = False

    # DDP 多 GPU
    ddp: bool = False
    ddp_backend: str = "nccl"            # nccl | gloo


@dataclass
class LogConfig:
    """日志与输出配置"""
    weight_dir: str = "../weights"       # 权重保存目录
    log_dir: str = "../train_logs"       # 日志目录
    csv_dir: str = "../csv"              # CSV 指标目录
    experiment_name: str = "exp"         # 实验名称，用于文件名前缀
    save_interval: int = 10              # 每 N 个 epoch 保存断点


@dataclass
class Config:
    """顶层配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    log: LogConfig = field(default_factory=LogConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """从 YAML 文件加载配置"""
        # Existing caller-supplied files keep precedence; bundled paths also work
        # when the command is launched outside code/.
        if not os.path.isfile(yaml_path):
            yaml_path = resolve_code_path(yaml_path)
        with open(yaml_path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f)

        cfg = cls()
        if 'data' in raw:
            for k, v in raw['data'].items():
                if hasattr(cfg.data, k):
                    setattr(cfg.data, k, v)
        if 'model' in raw:
            for k, v in raw['model'].items():
                if hasattr(cfg.model, k):
                    setattr(cfg.model, k, v)
        if 'train' in raw:
            for k, v in raw['train'].items():
                if hasattr(cfg.train, k):
                    setattr(cfg.train, k, v)
        if 'log' in raw:
            for k, v in raw['log'].items():
                if hasattr(cfg.log, k):
                    setattr(cfg.log, k, v)

        # crop_size 从 list 转 tuple
        if isinstance(cfg.data.crop_size, list):
            cfg.data.crop_size = tuple(cfg.data.crop_size)

        # Historical YAML paths are relative to code/, not the terminal cwd.
        cfg.data.root_dir = resolve_code_path(cfg.data.root_dir)
        if cfg.model.pretrained:
            cfg.model.pretrained = resolve_code_path(cfg.model.pretrained)
        cfg.log.weight_dir = resolve_code_path(cfg.log.weight_dir)
        cfg.log.log_dir = resolve_code_path(cfg.log.log_dir)
        cfg.log.csv_dir = resolve_code_path(cfg.log.csv_dir)

        return cfg

    def to_dict(self) -> dict:
        """导出为字典（用于保存实验配置）"""
        import dataclasses

        def _plain(v):
            # asdict 对 Tuple[int, int] 会产出 tuple，yaml.dump 写成 !!python/tuple，
            # 之后 yaml.safe_load 再也读不回来（历史 bug，曾产出不可加载的
            # *_config.yaml）。这里递归转成纯 list/dict。
            if isinstance(v, tuple):
                return [_plain(x) for x in v]
            if isinstance(v, list):
                return [_plain(x) for x in v]
            if isinstance(v, dict):
                return {k: _plain(x) for k, x in v.items()}
            return v

        result = {}
        for field_name in dataclasses.fields(self):
            val = getattr(self, field_name.name)
            if dataclasses.is_dataclass(val):
                result[field_name.name] = _plain(dataclasses.asdict(val))
            else:
                result[field_name.name] = _plain(val)
        return result
