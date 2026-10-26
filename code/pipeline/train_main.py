"""
pipeline/train_main.py
统一入口脚本，读取 YAML 配置启动训练。

用法：
    # 单卡训练
    python pipeline/train_main.py --config pipeline/configs/bisenetv2_ca.yaml

    # DDP 多卡训练（2 GPU）
    torchrun --nproc_per_node=2 pipeline/train_main.py --config pipeline/configs/bisenetv2_ca_ddp.yaml

    # 梯度累加 + AMP
    python pipeline/train_main.py --config pipeline/configs/fastscnn_amp.yaml
"""

import argparse
import os
import sys
import torch

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pipeline.config import Config
from pipeline.data_module import create_dataloaders
from pipeline.model_zoo import build_model
from pipeline.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="统一语义分割训练Pipeline")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    args = parser.parse_args()

    # 加载配置
    cfg = Config.from_yaml(args.config)
    print(f"配置文件: {args.config}")
    print(f"实验名称: {cfg.log.experiment_name}")
    print(f"模型: {cfg.model.name} | 数据集: {cfg.data.dataset}")
    print(f"Epochs: {cfg.train.epochs} | Batch: {cfg.train.batch_size} | "
          f"Accum: {cfg.train.accum_steps} | LR: {cfg.train.lr}")

    # 设备
    if cfg.train.ddp:
        # DDP 初始化
        torch.distributed.init_process_group(backend=cfg.train.ddp_backend)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 构建数据
    train_loader, val_loader, _, _ = create_dataloaders(cfg)

    # 构建模型
    model = build_model(cfg, device)

    # torch.compile 加速（PyTorch 2.0+）
    if cfg.train.compile_model:
        try:
            model = torch.compile(model)
            print("torch.compile 已启用")
        except Exception as e:
            print(f"torch.compile 启用失败: {e}，跳过")

    # DDP 包装
    if cfg.train.ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)

    # 创建训练器并启动
    trainer = Trainer(model, cfg, train_loader, val_loader, device)

    try:
        trainer.fit()
    except KeyboardInterrupt:
        # Ctrl+C 中断时自动保存
        print("\n\n⚠️ 检测到键盘中断，正在保存权重...")
        interrupt_path = os.path.join(cfg.log.weight_dir, f"{cfg.log.experiment_name}_interrupt.pth")
        torch.save(model.state_dict(), interrupt_path)
        trainer._save_checkpoint(trainer.start_epoch)
        print(f"中断权重已保存: {interrupt_path}")
        print(f"断点已保存: {trainer.ckpt_path}")

    # DDP 清理
    if cfg.train.ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
