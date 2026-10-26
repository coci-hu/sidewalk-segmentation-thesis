"""
train_deeplab.py
DeepLabv3+ 二分类可通行区域分割独立训练脚本。

与原版 train.py 的唯一差异：
  1. 模型加载 → get_deeplabv3plus_2class
  2. 所有存储路径独立命名，绝不覆盖其他实验数据

超参与基线完全对齐（梯度累加修复 OOM）：
  CROP_SIZE=(768,1024)  BATCH_SIZE=2  ACCUM_STEPS=2  等效batch=4  EPOCHS=100
  LR_INIT=1e-4  T_max=100  NUM_CLASSES=2
  损失 = 0.4×CE + 0.6×Dice
  优化器 = AdamW   调度器 = CosineAnnealingLR
"""

from project_paths import resolve_code_path
import os
import csv
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from cityscapes_dataset import CityscapesSegDataset
from model_deeplabv3plus import get_deeplabv3plus_2class
from metrics import compute_mIoU, compute_pixel_acc

# ===================== 路径配置（基准：code 文件夹内启动终端） =====================
ROOT_DATA = resolve_code_path("../dataset/cityscapes")
WEIGHT_ROOT = resolve_code_path("../weights")
TRAIN_LOG_ROOT = resolve_code_path("../train_logs")

# DeepLabv3+ 专属路径
WEIGHT_SAVE_PATH = os.path.join(WEIGHT_ROOT, "best_deeplab_100epoch_bin.pth")
CHECKPOINT_PATH = os.path.join(WEIGHT_ROOT, "ckpt_deeplab_100bin.pth")
CSV_LOG_PATH = os.path.join(TRAIN_LOG_ROOT, "deeplab_metrics.csv")
TXT_LOG_PATH = os.path.join(TRAIN_LOG_ROOT, "deeplab_train_log.txt")
# ============================================================================

CROP_SIZE = (768, 1024)
BATCH_SIZE = 2
ACCUM_STEPS = 2  # 梯度累加步数，等效 batch_size = 2×2 = 4
EPOCHS = 100
LR_INIT = 1e-4
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_logit, target_label):
        pred = F.softmax(pred_logit, dim=1)
        target = F.one_hot(target_label, num_classes=NUM_CLASSES).permute(0, 3, 1, 2).float()
        inter = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


def val_epoch(model, val_loader, ce_crit, dice_crit):
    model.eval()
    total_loss = 0.0
    total_miou = 0.0
    total_pix_acc = 0.0
    batch_num = len(val_loader)
    with torch.no_grad():
        for img, mask in val_loader:
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)
            out = model(img)
            loss_ce = ce_crit(out, mask)
            loss_dice = dice_crit(out, mask)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            total_loss += loss.item()
            pred = torch.argmax(out, dim=1)
            total_miou += compute_mIoU(pred, mask, num_classes=NUM_CLASSES)
            total_pix_acc += compute_pixel_acc(pred, mask)
    return total_loss / batch_num, total_miou / batch_num, total_pix_acc / batch_num


def log_print(msg, log_file=None):
    """同时打印到控制台与 TXT 日志（与其它 train_*.py 的日志形式统一）。"""
    print(msg, flush=True)
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()


def dump_csv(log_records):
    """每个 epoch 结束立刻落盘，避免长时间训练中断后丢失全部指标。"""
    with open(CSV_LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_mIoU", "val_pixel_acc"])
        writer.writerows(log_records)


if __name__ == "__main__":
    os.makedirs(WEIGHT_ROOT, exist_ok=True)
    os.makedirs(TRAIN_LOG_ROOT, exist_ok=True)

    train_dataset = CityscapesSegDataset(ROOT_DATA, split="train", crop_size=CROP_SIZE)
    val_dataset = CityscapesSegDataset(ROOT_DATA, split="val", crop_size=CROP_SIZE)
    # drop_last=True 是必需的：训练集 2975 张为奇数，末尾会剩 1 张组成 batch=1，
    # 而 ASPP 的全局池化分支在 1x1 特征图上做 BatchNorm2d，训练模式下会抛
    # ValueError: Expected more than 1 value per channel（2026-09-13 实测崩溃点）。
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True, persistent_workers=False,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True, persistent_workers=False)
    n_used = (len(train_dataset) // BATCH_SIZE) * BATCH_SIZE
    print(f"训练集：{len(train_dataset)} 张（drop_last 后实际使用 {n_used} 张），验证集：{len(val_dataset)} 张")

    model = get_deeplabv3plus_2class(DEVICE)

    ce_criterion = nn.CrossEntropyLoss()
    dice_criterion = DiceLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_INIT)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    start_epoch = 1
    best_miou = 0.0
    log_records = []
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt["best_miou"]
        log_records = ckpt["log_records"]
        print(f"\n===== 检测到 DeepLabv3+ 断点，从 Epoch {start_epoch} 继续训练 =====")
        print(f"历史最优mIoU: {best_miou:.4f}\n")

    log_file = open(TXT_LOG_PATH, "a", encoding="utf-8")
    log_print("=" * 70, log_file)
    log_print("  DeepLabv3+ 二分类训练（可通行区域 = Cityscapes sidewalk 代理标签）", log_file)
    log_print(f"  训练集 {len(train_dataset)} 张 / 验证集 {len(val_dataset)} 张", log_file)
    log_print(f"  CROP_SIZE={CROP_SIZE}  BATCH_SIZE={BATCH_SIZE}  ACCUM_STEPS={ACCUM_STEPS}"
              f"  等效batch={BATCH_SIZE * ACCUM_STEPS}  EPOCHS={EPOCHS}", log_file)
    log_print(f"  损失 = 0.4*CE + 0.6*Dice   AdamW(lr={LR_INIT})   CosineAnnealingLR(T_max={EPOCHS})", log_file)
    log_print(f"  参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.4f}M", log_file)
    log_print("=" * 70, log_file)

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        optimizer.zero_grad()
        epoch_start = time.time()
        for step_idx, (img, mask) in enumerate(train_loader):
            img = img.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            out = model(img)
            loss_ce = ce_criterion(out, mask)
            loss_dice = dice_criterion(out, mask)
            # 梯度累加：损失除以累加步数
            total_loss = (0.4 * loss_ce + 0.6 * loss_dice) / ACCUM_STEPS
            total_loss.backward()
            train_loss_sum += total_loss.item() * ACCUM_STEPS

            # 每 ACCUM_STEPS 步更新一次参数
            if (step_idx + 1) % ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

        # 处理尾部不足 ACCUM_STEPS 的残余梯度
        if (step_idx + 1) % ACCUM_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad()

        train_loss_avg = train_loss_sum / len(train_loader)
        epoch_time = time.time() - epoch_start

        val_loss, val_miou, val_pix_acc = val_epoch(model, val_loader, ce_criterion, dice_criterion)
        current_lr = optimizer.param_groups[0]["lr"]

        log_print(f"【DeepLabv3+ Epoch {epoch:03d}】 LR: {current_lr:.8f} | 耗时: {epoch_time:.0f}s", log_file)
        log_print(f"  Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss:.4f}", log_file)
        log_print(f"  Val mIoU: {val_miou:.4f} | Pixel Acc: {val_pix_acc:.4f}", log_file)

        log_records.append([epoch, train_loss_avg, val_loss, val_miou, val_pix_acc])
        dump_csv(log_records)   # 每个 epoch 落盘，避免中断丢失

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), WEIGHT_SAVE_PATH)
            log_print(f"  [OK] 更新DeepLabv3+最优权重，当前mIoU={best_miou:.4f}", log_file)

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_miou": best_miou,
                "log_records": log_records,
            }, CHECKPOINT_PATH)
            log_print(f"  ===== 保存DeepLabv3+断点 Epoch {epoch} =====", log_file)

        scheduler.step()

    dump_csv(log_records)
    log_print(f"训练日志导出完成：{CSV_LOG_PATH}", log_file)
    log_print(f"DeepLabv3+训练结束，最优权重路径：{WEIGHT_SAVE_PATH}", log_file)
    log_file.close()
