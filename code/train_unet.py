"""

⚠️ 本脚本不用于论文实验（U-Net 已从论文对比表中移除）。
   原因：batch=2、输入 768×1024 时实测峰值显存 9030 MiB，超过本机 8 GB 上限，
   会触发内存交换使单 epoch 耗时涨到 170 分钟；改 batch=1 后峰值 4687 MiB、
   单 epoch 约 25 分钟，100 epoch 仍需约 42 小时，来不及做公平的 100 轮对比。
   现有权重 weights/best_unet_100epoch_bin.pth 是 2026-08-03 的未充分训练中间产物
   （实测 val mIoU 66.0%），不得与已收敛模型同表比较。
   保留本脚本仅为记录，如将来要重训请把 BATCH_SIZE 改为 1、ACCUM_STEPS 改为 4。

受笔记本端 RTX 4060 Laptop（8GB）显存上限约束，参数量较大的 UNet 网络无法以 batch_size=4 完整训练。为保证等效训练批次与其余对比模型统一，本文采用梯度累加策略：设置单步 batch_size=2，梯度累加 2 步进行参数更新，等效批量大小仍为 4。该方式梯度下降效果与原生 batch_size=4 训练基本一致，不会对最终分割精度对比造成显著影响；其余轻量化网络显存充足，均采用原生 batch_size=4 完成训练。
train_unet.py
U-Net 二分类可通行区域分割独立训练脚本。

与原版 train.py 的唯一差异：
  1. 模型加载 → get_unet_2class
  2. 所有存储路径独立命名，绝不覆盖其他实验数据
  3. U-Net参数量大(~31M)，采用梯度累加保持等效batch_size=4

超参与基线完全对齐：
  CROP_SIZE=(768,1024)  等效BATCH_SIZE=4(实际2×累加2步)  EPOCHS=100
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
from tqdm import tqdm
from cityscapes_dataset import CityscapesSegDataset
from model_unet import get_unet_2class
from metrics import compute_mIoU, compute_pixel_acc

# ===================== 路径配置（基准：code 文件夹内启动终端） =====================
ROOT_DATA = resolve_code_path("../dataset/cityscapes")
WEIGHT_ROOT = resolve_code_path("../weights")
TRAIN_LOG_ROOT = resolve_code_path("../train_logs")

# U-Net 专属路径
WEIGHT_SAVE_PATH = os.path.join(WEIGHT_ROOT, "best_unet_100epoch_bin.pth")
CHECKPOINT_PATH = os.path.join(WEIGHT_ROOT, "ckpt_unet_100bin.pth")
CSV_LOG_PATH = os.path.join(TRAIN_LOG_ROOT, "unet_metrics.csv")
# ============================================================================

CROP_SIZE = (768, 1024)
BATCH_SIZE = 2          # U-Net参数量大(~31M)，显存不足时降至2
ACCUM_STEPS = 2         # 梯度累加2步，等效batch_size=4，与基线对齐
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


if __name__ == "__main__":
    os.makedirs(WEIGHT_ROOT, exist_ok=True)
    os.makedirs(TRAIN_LOG_ROOT, exist_ok=True)

    train_dataset = CityscapesSegDataset(ROOT_DATA, split="train", crop_size=CROP_SIZE)
    val_dataset = CityscapesSegDataset(ROOT_DATA, split="val", crop_size=CROP_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=1, pin_memory=True, persistent_workers=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=1, pin_memory=True, persistent_workers=False)
    print(f"训练集：{len(train_dataset)} 张，验证集：{len(val_dataset)}")
    print(f"实际BATCH_SIZE={BATCH_SIZE}, 梯度累加{ACCUM_STEPS}步, 等效batch_size={BATCH_SIZE * ACCUM_STEPS}")

    model = get_unet_2class(DEVICE)

    ce_criterion = nn.CrossEntropyLoss()
    dice_criterion = DiceLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_INIT)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

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
        print(f"\n===== 检测到 U-Net 断点，从 Epoch {start_epoch} 继续训练 =====")
        print(f"历史最优mIoU: {best_miou:.4f}\n")

    try:
        for epoch in range(start_epoch, EPOCHS + 1):
            model.train()
            train_loss_sum = 0.0
            optimizer.zero_grad()
            epoch_start = time.time()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False,
                        position=0, ncols=80, mininterval=5)
            for step_idx, (img, mask) in enumerate(pbar):
                img = img.to(DEVICE)
                mask = mask.to(DEVICE)
                out = model(img)
                loss_ce = ce_criterion(out, mask)
                loss_dice = dice_criterion(out, mask)
                total_loss = (0.4 * loss_ce + 0.6 * loss_dice) / ACCUM_STEPS
                total_loss.backward()
                train_loss_sum += total_loss.item() * ACCUM_STEPS
                # 每ACCUM_STEPS步更新一次参数
                if (step_idx + 1) % ACCUM_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                pbar.set_postfix(loss=f"{total_loss.item() * ACCUM_STEPS:.4f}")
            # 处理尾部不足accum_steps的梯度
            if (step_idx + 1) % ACCUM_STEPS != 0:
                optimizer.step()
                optimizer.zero_grad()
            train_loss_avg = train_loss_sum / len(train_loader)
            epoch_time = time.time() - epoch_start

            val_loss, val_miou, val_pix_acc = val_epoch(model, val_loader, ce_criterion, dice_criterion)
            current_lr = optimizer.param_groups[0]["lr"]

            print(f"【U-Net Epoch {epoch:02d}】 LR: {current_lr:.8f} | 耗时: {epoch_time:.0f}s")
            print(f"Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Val mIoU: {val_miou:.4f} | Pixel Acc: {val_pix_acc:.4f}\n")

            log_records.append([epoch, train_loss_avg, val_loss, val_miou, val_pix_acc])

            if val_miou > best_miou:
                best_miou = val_miou
                torch.save(model.state_dict(), WEIGHT_SAVE_PATH)
                print(f"✅ 更新U-Net最优权重，当前mIoU={best_miou:.4f}\n")

            if epoch % 10 == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_miou": best_miou,
                    "log_records": log_records,
                }, CHECKPOINT_PATH)
                print(f"===== 保存U-Net断点 Epoch {epoch} =====\n")

            scheduler.step()

    except KeyboardInterrupt:
        print("\n\n⚠️ 检测到 KeyboardInterrupt，正在保存 U-Net 权重和断点...")
        interrupt_path = os.path.join(WEIGHT_ROOT, "unet_interrupt.pth")
        torch.save(model.state_dict(), interrupt_path)
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_miou": best_miou,
            "log_records": log_records,
        }, CHECKPOINT_PATH)
        print(f"中断权重已保存：{interrupt_path}")
        print(f"断点已保存：{CHECKPOINT_PATH}")
        print(f"中断时 Epoch: {epoch}, 当前最优 mIoU: {best_miou:.4f}")

    with open(CSV_LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_mIoU", "val_pixel_acc"])
        writer.writerows(log_records)
    print(f"训练日志导出完成：{CSV_LOG_PATH}")
    print(f"U-Net训练结束，最优权重路径：{WEIGHT_SAVE_PATH}")
