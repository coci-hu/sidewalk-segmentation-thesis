"""
train_transfer.py
迁移学习微调脚本：加载 Cityscapes 预训练权重，在海口可通行区域数据集上微调。

与基线 train.py 的差异：
  1. 数据集 → HaikouBlindDataset（海口自建数据集）
  2. 损失权重 → 0.3×CE + 0.7×Dice（更侧重区域重叠）
  3. 支持加载 Cityscapes 预训练权重进行迁移学习
  4. 所有存储路径独立命名，绝不覆盖其他实验

超参与基线对齐：
  CROP_SIZE=(768,1024)  BATCH_SIZE=4  EPOCHS=100
  LR_INIT=1e-4  T_max=100  NUM_CLASSES=2
  优化器 = AdamW   调度器 = CosineAnnealingLR
"""

from project_paths import resolve_code_path
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset_haikou import HaikouBlindDataset
from model_bisenetv2 import get_bisenetv2_2class
from metrics import compute_mIoU, compute_pixel_acc

# ===================== 路径配置（基准：code 文件夹内启动终端） =====================
ROOT_DATA = resolve_code_path("../dataset/haikou_blind/split")
WEIGHT_ROOT = resolve_code_path("../weights")
TRAIN_LOG_ROOT = resolve_code_path("../train_logs")

# 迁移学习专属路径
WEIGHT_SAVE_PATH = os.path.join(WEIGHT_ROOT, "best_transfer_100epoch_bin.pth")
CHECKPOINT_PATH = os.path.join(WEIGHT_ROOT, "ckpt_transfer_100bin.pth")
CSV_LOG_PATH = os.path.join(TRAIN_LOG_ROOT, "train_log_transfer_100epoch_bin.csv")

# Cityscapes 预训练权重路径（可选，设 None 则从头训练）
PRETRAINED_PATH = os.path.join(WEIGHT_ROOT, "best_bisenet_100epoch_bin.pth")
# ============================================================================

CROP_SIZE = (768, 1024)
BATCH_SIZE = 4
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
            loss = 0.3 * loss_ce + 0.7 * loss_dice
            total_loss += loss.item()
            pred = torch.argmax(out, dim=1)
            total_miou += compute_mIoU(pred, mask, num_classes=NUM_CLASSES)
            total_pix_acc += compute_pixel_acc(pred, mask)
    return total_loss / batch_num, total_miou / batch_num, total_pix_acc / batch_num


if __name__ == "__main__":
    os.makedirs(WEIGHT_ROOT, exist_ok=True)
    os.makedirs(TRAIN_LOG_ROOT, exist_ok=True)

    # 加载海口可通行区域数据集
    train_dataset = HaikouBlindDataset(ROOT_DATA, split="train", crop_size=CROP_SIZE)
    val_dataset = HaikouBlindDataset(ROOT_DATA, split="val", crop_size=CROP_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"训练集：{len(train_dataset)} 张，验证集：{len(val_dataset)}")

    # 初始化模型
    model = get_bisenetv2_2class(DEVICE)

    # 迁移学习：加载 Cityscapes 预训练权重
    if os.path.exists(PRETRAINED_PATH):
        state_dict = torch.load(PRETRAINED_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"✅ 已加载 Cityscapes 预训练权重: {PRETRAINED_PATH}")
    else:
        print(f"[WARN] 预训练权重不存在: {PRETRAINED_PATH}，将从头训练")

    # 损失、优化器、余弦退火
    ce_criterion = nn.CrossEntropyLoss()
    dice_criterion = DiceLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_INIT)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # 断点续训初始化
    start_epoch = 1
    best_miou = 0.0
    log_records = []
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt["best_miou"]
        log_records = ckpt["log_records"]
        print(f"\n===== 检测到迁移学习断点，从 Epoch {start_epoch} 继续训练 =====")
        print(f"历史最优mIoU: {best_miou:.4f}\n")

    # 训练主循环
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        for img, mask in train_loader:
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)
            optimizer.zero_grad()
            out = model(img)
            loss_ce = ce_criterion(out, mask)
            loss_dice = dice_criterion(out, mask)
            total_loss = 0.3 * loss_ce + 0.7 * loss_dice
            total_loss.backward()
            optimizer.step()
            train_loss_sum += total_loss.item()
        train_loss_avg = train_loss_sum / len(train_loader)

        val_loss, val_miou, val_pix_acc = val_epoch(model, val_loader, ce_criterion, dice_criterion)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"【迁移学习 Epoch {epoch:02d}】 LR: {current_lr:.8f}")
        print(f"Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val mIoU: {val_miou:.4f} | Pixel Acc: {val_pix_acc:.4f}\n")

        log_records.append([epoch, train_loss_avg, val_loss, val_miou])

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), WEIGHT_SAVE_PATH)
            print(f"✅ 更新迁移学习最优权重，当前mIoU={best_miou:.4f}\n")

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_miou": best_miou,
                "log_records": log_records,
            }, CHECKPOINT_PATH)
            print(f"===== 保存迁移学习断点 Epoch {epoch} =====\n")

        scheduler.step()

    # 导出训练日志
    with open(CSV_LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_mIoU"])
        writer.writerows(log_records)
    print(f"训练日志导出完成：{CSV_LOG_PATH}")
    print(f"迁移学习训练结束，最优权重路径：{WEIGHT_SAVE_PATH}")
