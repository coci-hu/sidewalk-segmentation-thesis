from project_paths import resolve_code_path
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from cityscapes_dataset import CityscapesSegDataset
from model_stdc_seg import get_stdc_seg_2class
from metrics import compute_mIoU, compute_pixel_acc

# ===================== 全部使用相对路径，基准：code文件夹内启动终端 =====================
# 数据集：code/../dataset/cityscapes
ROOT_DATA = resolve_code_path("../dataset/cityscapes")
# 权重目录：code/../weights
WEIGHT_ROOT = resolve_code_path("../weights")
# 日志目录：code/../train_logs
TRAIN_LOG_ROOT = resolve_code_path("../train_logs")
# STDC-Seg 专属路径（独立命名，绝不覆盖其他实验数据）
WEIGHT_SAVE_PATH = os.path.join(WEIGHT_ROOT, "best_stdc_seg_100epoch_bin.pth")
CHECKPOINT_PATH = os.path.join(WEIGHT_ROOT, "ckpt_stdc_seg_100bin.pth")
CSV_LOG_PATH = os.path.join(TRAIN_LOG_ROOT, "train_log_stdc_seg_100epoch_bin.csv")
# =================================================================
CROP_SIZE = (768, 1024)
BATCH_SIZE = 4
EPOCHS = 100
LR_INIT = 1e-4
NUM_CLASSES = 2  # 二分类：0背景，1可通行区域
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 二分类Dice损失
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
            miou = compute_mIoU(pred, mask, num_classes=NUM_CLASSES)
            pix_acc = compute_pixel_acc(pred, mask)
            total_miou += miou
            total_pix_acc += pix_acc
    avg_loss = total_loss / batch_num
    avg_miou = total_miou / batch_num
    avg_pix_acc = total_pix_acc / batch_num
    return avg_loss, avg_miou, avg_pix_acc

if __name__ == "__main__":
    # 自动创建weights和train_logs文件夹
    os.makedirs(WEIGHT_ROOT, exist_ok=True)
    os.makedirs(TRAIN_LOG_ROOT, exist_ok=True)

    # 加载Cityscapes数据集
    train_dataset = CityscapesSegDataset(ROOT_DATA, split="train", crop_size=CROP_SIZE)
    val_dataset = CityscapesSegDataset(ROOT_DATA, split="val", crop_size=CROP_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"训练集：{len(train_dataset)} 张，验证集：{len(val_dataset)}")

    # 初始化STDC-Seg二分类模型
    model = get_stdc_seg_2class(DEVICE)

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
        print(f"\n===== 检测STDC-Seg断点，从 Epoch {start_epoch} 继续训练 =====")
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
            total_loss = 0.4 * loss_ce + 0.6 * loss_dice
            total_loss.backward()
            optimizer.step()
            train_loss_sum += total_loss.item()
        train_loss_avg = train_loss_sum / len(train_loader)

        # 验证
        val_loss, val_miou, val_pix_acc = val_epoch(model, val_loader, ce_criterion, dice_criterion)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"【STDC-Seg Epoch {epoch:02d}】 LR: {current_lr:.8f}")
        print(f"Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val mIoU: {val_miou:.4f} | Pixel Acc: {val_pix_acc:.4f}\n")

        log_records.append([epoch, train_loss_avg, val_loss, val_miou])

        # 保存最优权重
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), WEIGHT_SAVE_PATH)
            print(f"✅ 更新STDC-Seg最优权重，当前mIoU={best_miou:.4f}\n")

        # 每10轮保存断点
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_miou": best_miou,
                "log_records": log_records
            }, CHECKPOINT_PATH)
            print(f"===== 保存STDC-Seg断点 Epoch {epoch} =====\n")

        scheduler.step()

    # 导出日志
    with open(CSV_LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_mIoU"])
        writer.writerows(log_records)
    print(f"训练日志导出完成：{CSV_LOG_PATH}")
    print(f"STDC-Seg训练结束，最优权重路径：{WEIGHT_SAVE_PATH}")
