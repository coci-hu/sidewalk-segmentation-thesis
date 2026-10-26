"""
train_enet_ba.py
ENet-BA（仅边界感知分支）消融实验训练脚本。

消融实验：验证边界感知分支（BA）单独的效果。
与 ENet-DBA（CA+BA 协同）对比，分离 CA 和 BA 各自的贡献。

硬性规范与 train_enet_dba.py 完全一致，仅模型改为 ENetBA。

存储路径：
  权重：  ../weights/enet_ba_best.pth
  断点：  ../weights/enet_ba_ckpt.pth
  日志：  ../train_logs/enet_ba_train_log.txt
  指标：  ../train_logs/enet_ba_metrics.csv
"""

from project_paths import resolve_code_path
import os
import csv
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from model_enet_ba import ENetBA, generate_boundary_gt
from metrics import compute_mIoU, compute_pixel_acc

# ===================== 超参数配置 =====================
CROP_SIZE = (768, 1024)
BATCH_SIZE = 4
EPOCHS = 100
LR_INIT = 1e-4
NUM_CLASSES = 2
LAMBDA_BDY = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== 存储路径配置 =====================
ROOT_DATA = resolve_code_path("../dataset/cityscapes")
WEIGHT_DIR = resolve_code_path("../weights")
WEIGHT_SAVE_PATH = os.path.join(WEIGHT_DIR, "enet_ba_best.pth")
CHECKPOINT_PATH = os.path.join(WEIGHT_DIR, "enet_ba_ckpt.pth")
LOG_DIR = resolve_code_path("../train_logs")
LOG_TXT_PATH = os.path.join(LOG_DIR, "enet_ba_train_log.txt")
CSV_LOG_PATH = os.path.join(LOG_DIR, "enet_ba_metrics.csv")

SIDEWALK_ID = 8


class BlindRoadDataset(Dataset):
    def __init__(self, root_dir, split="train", crop_size=(512, 512)):
        self.img_dir = os.path.join(root_dir, "leftImg8bit", split)
        self.label_dir = os.path.join(root_dir, "gtFine", split)
        self.crop_h, self.crop_w = crop_size
        self.img_paths = []
        self.label_paths = []
        for city in sorted(os.listdir(self.img_dir)):
            city_img_dir = os.path.join(self.img_dir, city)
            city_label_dir = os.path.join(self.label_dir, city)
            if not os.path.isdir(city_img_dir):
                continue
            for img_name in sorted(os.listdir(city_img_dir)):
                img_path = os.path.join(city_img_dir, img_name)
                label_name = img_name.replace("leftImg8bit", "gtFine_labelIds")
                label_path = os.path.join(city_label_dir, label_name)
                self.img_paths.append(img_path)
                self.label_paths.append(label_path)
        self.transform_img = transforms.Compose([
            transforms.Resize((self.crop_h, self.crop_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        label = np.array(Image.open(self.label_paths[idx]))
        binary_mask = np.zeros_like(label, dtype=np.uint8)
        binary_mask[label == SIDEWALK_ID] = 1
        mask_pil = Image.fromarray(binary_mask)
        mask_pil = mask_pil.resize((self.crop_w, self.crop_h), Image.NEAREST)
        img_tensor = self.transform_img(img)
        mask_tensor = torch.tensor(np.array(mask_pil), dtype=torch.long)
        return img_tensor, mask_tensor


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
            outputs = model(img)
            seg_out = outputs[0]
            loss_ce = ce_crit(seg_out, mask)
            loss_dice = dice_crit(seg_out, mask)
            total_loss += (0.4 * loss_ce + 0.6 * loss_dice).item()
            pred = torch.argmax(seg_out, dim=1)
            total_miou += compute_mIoU(pred, mask, num_classes=NUM_CLASSES)
            total_pix_acc += compute_pixel_acc(pred, mask)
    avg_loss = total_loss / batch_num
    avg_miou = total_miou / batch_num
    avg_pix_acc = total_pix_acc / batch_num
    return avg_loss, avg_miou, avg_pix_acc


def log_print(msg, log_file=None):
    print(msg)
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()


if __name__ == "__main__":
    os.makedirs(WEIGHT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = open(LOG_TXT_PATH, "a", encoding="utf-8")

    train_dataset = BlindRoadDataset(ROOT_DATA, split="train", crop_size=CROP_SIZE)
    val_dataset = BlindRoadDataset(ROOT_DATA, split="val", crop_size=CROP_SIZE)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=1, pin_memory=True, persistent_workers=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=1, pin_memory=True, persistent_workers=False
    )
    log_print(f"训练集：{len(train_dataset)} 张，验证集：{len(val_dataset)}", log_file)
    log_print(f"BATCH_SIZE={BATCH_SIZE}，输入尺寸：{CROP_SIZE}，设备：{DEVICE}", log_file)
    log_print(f"边界损失权重 λ_bdy={LAMBDA_BDY}", log_file)

    model = ENetBA(num_classes=NUM_CLASSES, boundary_weight=LAMBDA_BDY).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters())
    log_print(f"ENet-BA 参数量：{param_count / 1e6:.2f}M", log_file)

    ce_criterion = nn.CrossEntropyLoss()
    dice_criterion = DiceLoss()
    bdy_criterion = nn.BCELoss()

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_INIT)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    start_epoch = 1
    best_miou = 0.0
    best_epoch = 0
    best_pix_acc = 0.0
    best_val_loss = float("inf")
    log_records = []
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt["best_miou"]
        best_epoch = ckpt.get("best_epoch", 0)
        best_pix_acc = ckpt.get("best_pix_acc", 0.0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        log_records = ckpt["log_records"]
        log_print(f"\n===== 检测到 ENet-BA 断点，从 Epoch {start_epoch} 继续训练 =====", log_file)
        log_print(f"历史最优 mIoU: {best_miou:.4f} (Epoch {best_epoch})\n", log_file)

    train_start_time = time.time()

    try:
        for epoch in range(start_epoch, EPOCHS + 1):
            model.train()
            train_loss_sum = 0.0
            train_seg_loss_sum = 0.0
            train_bdy_loss_sum = 0.0
            epoch_start = time.time()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False,
                        position=0, ncols=80, mininterval=5)
            for img, mask in pbar:
                img = img.to(DEVICE, non_blocking=True)
                mask = mask.to(DEVICE, non_blocking=True)

                seg_out, boundary_out, boundary_gt, bdy_loss = model(img, mask=mask)

                loss_ce = ce_criterion(seg_out, mask)
                loss_dice = dice_criterion(seg_out, mask)
                seg_loss = 0.4 * loss_ce + 0.6 * loss_dice
                total_loss = seg_loss + LAMBDA_BDY * bdy_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                train_loss_sum += total_loss.item()
                train_seg_loss_sum += seg_loss.item()
                train_bdy_loss_sum += bdy_loss.item()
                pbar.set_postfix(loss=f"{total_loss.item():.4f}", bdy=f"{bdy_loss.item():.4f}")

            train_loss_avg = train_loss_sum / len(train_loader)
            train_seg_avg = train_seg_loss_sum / len(train_loader)
            train_bdy_avg = train_bdy_loss_sum / len(train_loader)
            epoch_time = time.time() - epoch_start

            val_loss, val_miou, val_pix_acc = val_epoch(model, val_loader, ce_criterion, dice_criterion)
            current_lr = optimizer.param_groups[0]["lr"]

            log_print(f"【ENet-BA Epoch {epoch:03d}】 LR: {current_lr:.8f} | 耗时: {epoch_time:.0f}s", log_file)
            log_print(f"  Train Loss: {train_loss_avg:.4f} (seg={train_seg_avg:.4f}, bdy={train_bdy_avg:.4f}) | Val Loss: {val_loss:.4f}", log_file)
            log_print(f"  Val mIoU: {val_miou:.4f} | Pixel Acc: {val_pix_acc:.4f}", log_file)

            log_records.append([epoch, train_loss_avg, val_loss, val_miou, val_pix_acc])

            if val_miou > best_miou:
                best_miou = val_miou
                best_epoch = epoch
                best_pix_acc = val_pix_acc
                best_val_loss = val_loss
                torch.save(model.state_dict(), WEIGHT_SAVE_PATH)
                log_print(f"  ✅ 更新最优权重！mIoU={best_miou:.4f}, Pixel Acc={best_pix_acc:.4f}", log_file)

            if epoch % 10 == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_miou": best_miou,
                    "best_epoch": best_epoch,
                    "best_pix_acc": best_pix_acc,
                    "best_val_loss": best_val_loss,
                    "log_records": log_records,
                }, CHECKPOINT_PATH)
                log_print(f"  ===== 保存断点 Epoch {epoch} =====", log_file)

            scheduler.step()

    except KeyboardInterrupt:
        log_print("\n\n⚠️ 检测到 KeyboardInterrupt，正在保存当前权重...", log_file)
        torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, "enet_ba_interrupt.pth"))
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_miou": best_miou,
            "best_epoch": best_epoch,
            "best_pix_acc": best_pix_acc,
            "best_val_loss": best_val_loss,
            "log_records": log_records,
        }, CHECKPOINT_PATH)
        log_print(f"中断权重已保存：{os.path.join(WEIGHT_DIR, 'enet_ba_interrupt.pth')}", log_file)
        log_print(f"断点已保存：{CHECKPOINT_PATH}", log_file)
        log_print(f"中断时 Epoch: {epoch}, 当前最优 mIoU: {best_miou:.4f}", log_file)

    train_total_time = time.time() - train_start_time
    log_print("\n" + "=" * 60, log_file)
    log_print("       ENet-BA 训练结束，最优指标汇总", log_file)
    log_print("=" * 60, log_file)
    log_print(f"  最优 Epoch:       {best_epoch}", log_file)
    log_print(f"  最优 mIoU:        {best_miou:.4f}", log_file)
    log_print(f"  最优 Pixel Acc:   {best_pix_acc:.4f}", log_file)
    log_print(f"  对应 Val Loss:    {best_val_loss:.4f}", log_file)
    log_print(f"  总训练时长:       {train_total_time / 60:.1f} 分钟", log_file)
    log_print(f"  最优权重路径:     {WEIGHT_SAVE_PATH}", log_file)
    log_print(f"  训练日志路径:     {LOG_TXT_PATH}", log_file)
    log_print(f"  指标CSV路径:      {CSV_LOG_PATH}", log_file)
    log_print("=" * 60, log_file)

    with open(CSV_LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_mIoU", "val_pixel_acc"])
        writer.writerows(log_records)
    log_print(f"指标 CSV 已导出：{CSV_LOG_PATH}", log_file)

    log_file.close()
