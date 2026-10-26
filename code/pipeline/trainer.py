"""
pipeline/trainer.py
通用训练类：封装 epoch 循环、loss 计算、metrics、验证、早停、
checkpoint 保存、断点续训、梯度累加、AMP、梯度裁剪、DDP。

兼容旧 plot_*.py 脚本读取 CSV 指标。
"""

import os
import csv
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# 确保能导入 metrics
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from metrics import compute_mIoU, compute_pixel_acc


class DiceLoss(nn.Module):
    """二分类 Dice Loss"""
    def __init__(self, num_classes=2, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, pred_logit, target_label):
        pred = F.softmax(pred_logit, dim=1)
        target = F.one_hot(target_label, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        inter = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class Trainer:
    """
    统一训练器，支持所有 pipeline 配置驱动的训练策略。

    参数:
        model     — nn.Module 模型
        cfg       — Config 实例
        train_loader — 训练集 DataLoader
        val_loader   — 验证集 DataLoader
        device    — torch.device
    """

    def __init__(self, model, cfg, train_loader, val_loader, device):
        self.model = model
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # ---- 损失函数 ----
        self.ce_criterion = nn.CrossEntropyLoss()
        self.dice_criterion = DiceLoss(num_classes=cfg.model.num_classes)

        # ---- 优化器 ----
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        if cfg.train.optimizer == "sgd":
            self.optimizer = SGD(trainable_params, lr=cfg.train.lr,
                                 weight_decay=cfg.train.weight_decay, momentum=0.9)
        else:
            self.optimizer = AdamW(trainable_params, lr=cfg.train.lr,
                                   weight_decay=cfg.train.weight_decay)

        # ---- 学习率调度器 ----
        if cfg.train.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=cfg.train.t_max, eta_min=cfg.train.eta_min)
        elif cfg.train.scheduler == "step":
            self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)
        else:
            self.scheduler = None

        # ---- AMP ----
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.amp)
        self.use_amp = cfg.train.amp

        # ---- 梯度裁剪 ----
        self.grad_clip = cfg.train.grad_clip

        # ---- 梯度累加 ----
        self.accum_steps = cfg.train.accum_steps

        # ---- 路径 ----
        exp = cfg.log.experiment_name
        self.weight_dir = cfg.log.weight_dir
        self.log_dir = cfg.log.log_dir
        self.csv_dir = cfg.log.csv_dir
        self.best_path = os.path.join(self.weight_dir, f"{exp}_best.pth")
        self.ckpt_path = os.path.join(self.weight_dir, f"{exp}_ckpt.pth")
        self.csv_path = os.path.join(self.csv_dir, f"{exp}_metrics.csv")
        self.save_interval = cfg.log.save_interval

        for d in [self.weight_dir, self.log_dir, self.csv_dir]:
            os.makedirs(d, exist_ok=True)

        # ---- 断点续训 ----
        self.start_epoch = 1
        self.best_miou = 0.0
        self.log_records = []
        self._load_checkpoint()

    def _load_checkpoint(self):
        """加载断点续训"""
        if os.path.exists(self.ckpt_path):
            ckpt = torch.load(self.ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            self.optimizer.load_state_dict(ckpt["optim_state"])
            if self.scheduler and "scheduler_state" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler_state"])
            self.start_epoch = ckpt["epoch"] + 1
            self.best_miou = ckpt.get("best_miou", 0.0)
            self.log_records = ckpt.get("log_records", [])
            print(f"\n===== 断点续训: 从 Epoch {self.start_epoch} 继续 =====")
            print(f"历史最优 mIoU: {self.best_miou:.4f}\n")

    def _save_checkpoint(self, epoch):
        """保存断点"""
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "best_miou": self.best_miou,
            "log_records": self.log_records,
            "config": self.cfg.to_dict(),
        }
        if self.scheduler:
            ckpt["scheduler_state"] = self.scheduler.state_dict()
        torch.save(ckpt, self.ckpt_path)

    def train_epoch(self, epoch):
        """训练一个 epoch，支持梯度累加、AMP、梯度裁剪"""
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:03d}", leave=False,
                    position=0, ncols=80, mininterval=5)

        for step, (img, mask) in enumerate(pbar):
            img = img.to(self.device)
            mask = mask.to(self.device)

            # 前向 + AMP
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                out = self.model(img)
                loss_ce = self.ce_criterion(out, mask)
                loss_dice = self.dice_criterion(out, mask)
                loss = (self.cfg.train.ce_weight * loss_ce +
                        self.cfg.train.dice_weight * loss_dice)
                loss = loss / self.accum_steps  # 梯度累加归一化

            # 反向
            self.scaler.scale(loss).backward()

            # 梯度累加：每 accum_steps 步更新一次
            if (step + 1) % self.accum_steps == 0:
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.accum_steps
            pbar.set_postfix(loss=f"{loss.item() * self.accum_steps:.4f}")

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    @torch.no_grad()
    def val_epoch(self):
        """验证一个 epoch"""
        self.model.eval()
        total_loss = 0.0
        total_miou = 0.0
        total_pix_acc = 0.0
        num_classes = self.cfg.model.num_classes

        for img, mask in self.val_loader:
            img = img.to(self.device)
            mask = mask.to(self.device)
            out = self.model(img)
            loss_ce = self.ce_criterion(out, mask)
            loss_dice = self.dice_criterion(out, mask)
            loss = (self.cfg.train.ce_weight * loss_ce +
                    self.cfg.train.dice_weight * loss_dice)
            total_loss += loss.item()

            pred = torch.argmax(out, dim=1)
            total_miou += compute_mIoU(pred, mask, num_classes=num_classes)
            total_pix_acc += compute_pixel_acc(pred, mask)

        n = len(self.val_loader)
        return total_loss / n, total_miou / n, total_pix_acc / n

    def fit(self):
        """完整训练循环"""
        print(f"\n{'='*60}")
        print(f"实验: {self.cfg.log.experiment_name}")
        print(f"模型: {self.cfg.model.name} | 类别数: {self.cfg.model.num_classes}")
        print(f"Epochs: {self.cfg.train.epochs} | Batch: {self.cfg.train.batch_size}"
              f" | Accum: {self.cfg.train.accum_steps}"
              f" (等效 {self.cfg.train.batch_size * self.cfg.train.accum_steps})")
        print(f"LR: {self.cfg.train.lr} | AMP: {self.use_amp} | GradClip: {self.grad_clip}")
        print(f"{'='*60}\n")

        for epoch in range(self.start_epoch, self.cfg.train.epochs + 1):
            t0 = time.time()

            # 训练
            train_loss = self.train_epoch(epoch)

            # 验证
            val_loss, val_miou, val_pix_acc = self.val_epoch()

            # 调度器
            if self.scheduler:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            elapsed = int(time.time() - t0)

            print(f"【{self.cfg.log.experiment_name} Epoch {epoch:03d}】 LR: {lr:.8f} | 耗时: {elapsed}s")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Val mIoU: {val_miou:.4f} | Pixel Acc: {val_pix_acc:.4f}")

            # 记录
            self.log_records.append([epoch, train_loss, val_loss, val_miou])

            # 保存最优权重
            if val_miou > self.best_miou:
                self.best_miou = val_miou
                torch.save(self.model.state_dict(), self.best_path)
                print(f"  ✅ 更新最优权重，mIoU={self.best_miou:.4f}")

            # 定期保存断点
            if epoch % self.save_interval == 0:
                self._save_checkpoint(epoch)

        # 训练结束：保存最终断点 + 导出 CSV
        self._save_checkpoint(self.cfg.train.epochs)
        self._export_csv()

        print(f"\n{'='*60}")
        print(f"训练结束！实验: {self.cfg.log.experiment_name}")
        print(f"最优 mIoU: {self.best_miou:.4f}")
        print(f"最优权重: {self.best_path}")
        print(f"CSV 日志: {self.csv_path}")
        print(f"{'='*60}")

    def _export_csv(self):
        """导出训练日志 CSV（兼容旧 plot_*.py 脚本）"""
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_mIoU"])
            writer.writerows(self.log_records)
        print(f"训练日志导出: {self.csv_path}")

        # 同时保存实验配置
        config_path = os.path.join(self.log_dir, f"{self.cfg.log.experiment_name}_config.yaml")
        import yaml
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.cfg.to_dict(), f, allow_unicode=True, default_flow_style=False)
        print(f"实验配置已保存: {config_path}")
