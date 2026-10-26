"""
train_segformer.py
SegFormer-B0 轻量 Transformer 语义分割完整训练脚本（二分类：背景 + 可通行区域）。

硬性规范：
  - 输入尺寸：CROP_SIZE = (768, 1024)
  - 训练批次：BATCH_SIZE=2, ACCUM_STEPS=2, 等效 batch_size=4
  - DataLoader：num_workers=1, pin_memory=True, persistent_workers=False
  - 优化器：AdamW, lr=1e-4
  - 调度器：CosineAnnealingLR, T_max=100, eta_min=1e-6
  - 损失函数：0.4×CrossEntropy + 0.6×Dice
  - 断点续训：自动检测 checkpoint 恢复训练
  - KeyboardInterrupt 捕获：Ctrl+C 中断时自动保存当前权重和断点
  - 训练结束汇总最优指标

存储路径（基准：code 文件夹内启动终端）：
  权重：  ../weights/segformer_best.pth      （最优模型）
  断点：  ../weights/segformer_ckpt.pth      （断点续训）
  日志：  ../train_logs/segformer_train_log.txt    （训练过程日志）
  指标：  ../train_logs/segformer_metrics.csv       （CSV 格式指标记录）

注意：SegFormer-B0 参数量约 4.30M，Transformer 注意力计算显存开销较大，
      故采用梯度累加策略（BATCH_SIZE=2, ACCUM_STEPS=2）确保 8GB 显存可训练。
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
from model_segformer import get_segformer_b0_2class
from metrics import compute_mIoU, compute_pixel_acc

# ===================== 超参数配置 =====================
CROP_SIZE = (768, 1024)      # 输入图像裁剪尺寸（高, 宽），与基线实验对齐
BATCH_SIZE = 2               # 单步批次大小（显存限制）
ACCUM_STEPS = 2              # 梯度累加步数，等效 batch_size = 2×2 = 4
EPOCHS = 100                 # 总训练轮数
LR_INIT = 1e-4               # 初始学习率
NUM_CLASSES = 2              # 分类数：0=背景, 1=可通行区域
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== 存储路径配置 =====================
# 数据集路径（基准：code 文件夹内启动终端）
ROOT_DATA = resolve_code_path("../dataset/cityscapes")
# 权重目录（统一存到项目根目录 ../weights）
WEIGHT_DIR = resolve_code_path("../weights")
WEIGHT_SAVE_PATH = os.path.join(WEIGHT_DIR, "segformer_best.pth")     # 最优权重
CHECKPOINT_PATH = os.path.join(WEIGHT_DIR, "segformer_ckpt.pth")      # 断点文件
# 日志目录（统一存到项目根目录 ../train_logs）
LOG_DIR = resolve_code_path("../train_logs")
LOG_TXT_PATH = os.path.join(LOG_DIR, "segformer_train_log.txt")       # 训练日志
# 指标CSV（与日志同目录）
CSV_LOG_PATH = os.path.join(LOG_DIR, "segformer_metrics.csv")         # 指标记录

# Cityscapes sidewalk(id=8) 像素 ID（作为"视障可通行区域"的代理标签）
SIDEWALK_ID = 8


# ===================== 自定义可通行区域数据集 =====================
class BlindRoadDataset(Dataset):
    """
    自定义可通行区域语义分割数据集。
    基于 Cityscapes 格式，将多类别标签转为「可通行区域(1) / 背景(0)」二分类掩码。

    参数：
        root_dir:  数据集根目录（包含 leftImg8bit/ 和 gtFine/ 子目录）
        split:     数据划分，"train" 或 "val"
        crop_size: 裁剪尺寸 (height, width)
    """
    def __init__(self, root_dir, split="train", crop_size=(512, 512)):
        self.img_dir = os.path.join(root_dir, "leftImg8bit", split)
        self.label_dir = os.path.join(root_dir, "gtFine", split)
        self.crop_h, self.crop_w = crop_size

        # 遍历城市子文件夹，配对原图与标注
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

        # 图像标准化预处理（ImageNet 均值/标准差）
        self.transform_img = transforms.Compose([
            transforms.Resize((self.crop_h, self.crop_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 读取原图
        img = Image.open(self.img_paths[idx]).convert("RGB")
        # 读取标签（单通道灰度图）
        label = np.array(Image.open(self.label_paths[idx]))
        # 二分类掩码：人行道/可通行区域像素 = 1，其余 = 0
        binary_mask = np.zeros_like(label, dtype=np.uint8)
        binary_mask[label == SIDEWALK_ID] = 1
        # 掩码缩放到裁剪尺寸（最近邻插值，避免引入新标签值）
        mask_pil = Image.fromarray(binary_mask)
        mask_pil = mask_pil.resize((self.crop_w, self.crop_h), Image.NEAREST)

        img_tensor = self.transform_img(img)
        mask_tensor = torch.tensor(np.array(mask_pil), dtype=torch.long)
        return img_tensor, mask_tensor


# ===================== Dice 损失 =====================
class DiceLoss(nn.Module):
    """
    二分类 Dice 损失。
    衡量预测与真值之间的重叠度，对类别不平衡更鲁棒。
    """
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


# ===================== 验证函数 =====================
def val_epoch(model, val_loader, ce_crit, dice_crit):
    """
    在验证集上评估一个 epoch，返回平均 loss、mIoU、Pixel Acc。
    """
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

    avg_loss = total_loss / batch_num
    avg_miou = total_miou / batch_num
    avg_pix_acc = total_pix_acc / batch_num
    return avg_loss, avg_miou, avg_pix_acc


# ===================== 日志写入辅助函数 =====================
def log_print(msg, log_file=None):
    """
    同时输出到终端和日志文件。
    """
    print(msg)
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()


# ===================== 主训练流程 =====================
if __name__ == "__main__":
    # ---- 创建所有输出目录 ----
    os.makedirs(WEIGHT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # ---- 打开日志文件 ----
    log_file = open(LOG_TXT_PATH, "a", encoding="utf-8")

    # ---- 加载数据集 ----
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
    log_print(f"实际 BATCH_SIZE={BATCH_SIZE}, 梯度累加 {ACCUM_STEPS} 步, "
              f"等效 batch_size={BATCH_SIZE * ACCUM_STEPS}", log_file)
    log_print(f"输入尺寸：{CROP_SIZE}，设备：{DEVICE}", log_file)

    # ---- 初始化模型 ----
    model = get_segformer_b0_2class(DEVICE)
    param_count = sum(p.numel() for p in model.parameters())
    log_print(f"SegFormer-B0 参数量：{param_count / 1e6:.2f}M", log_file)

    # ---- 损失函数 ----
    ce_criterion = nn.CrossEntropyLoss()
    dice_criterion = DiceLoss()

    # ---- 优化器与调度器 ----
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_INIT)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ---- 断点续训初始化 ----
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
        log_print(f"\n===== 检测到 SegFormer-B0 断点，从 Epoch {start_epoch} 继续训练 =====", log_file)
        log_print(f"历史最优 mIoU: {best_miou:.4f} (Epoch {best_epoch})\n", log_file)

    # ---- 记录训练开始时间 ----
    train_start_time = time.time()

    # ===================== 训练主循环 =====================
    try:
        for epoch in range(start_epoch, EPOCHS + 1):
            model.train()
            train_loss_sum = 0.0
            optimizer.zero_grad()
            epoch_start = time.time()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False,
                        position=0, ncols=80, mininterval=5)
            for step_idx, (img, mask) in enumerate(pbar):
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

                pbar.set_postfix(loss=f"{total_loss.item() * ACCUM_STEPS:.4f}")

            # 处理尾部不足 ACCUM_STEPS 的残余梯度
            if (step_idx + 1) % ACCUM_STEPS != 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss_avg = train_loss_sum / len(train_loader)
            epoch_time = time.time() - epoch_start

            # ---- 验证 ----
            val_loss, val_miou, val_pix_acc = val_epoch(model, val_loader, ce_criterion, dice_criterion)
            current_lr = optimizer.param_groups[0]["lr"]

            # ---- 打印本轮指标 ----
            log_print(f"【SegFormer-B0 Epoch {epoch:03d}】 LR: {current_lr:.8f} | 耗时: {epoch_time:.0f}s", log_file)
            log_print(f"  Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss:.4f}", log_file)
            log_print(f"  Val mIoU: {val_miou:.4f} | Pixel Acc: {val_pix_acc:.4f}", log_file)

            # ---- 记录指标 ----
            log_records.append([epoch, train_loss_avg, val_loss, val_miou, val_pix_acc])

            # ---- 保存最优权重 ----
            if val_miou > best_miou:
                best_miou = val_miou
                best_epoch = epoch
                best_pix_acc = val_pix_acc
                best_val_loss = val_loss
                torch.save(model.state_dict(), WEIGHT_SAVE_PATH)
                log_print(f"  ✅ 更新最优权重！mIoU={best_miou:.4f}, Pixel Acc={best_pix_acc:.4f}", log_file)

            # ---- 每10轮保存断点 ----
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
        # ---- Ctrl+C 中断时自动保存权重和断点 ----
        log_print("\n\n⚠️ 检测到 KeyboardInterrupt，正在保存当前权重...", log_file)
        torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, "segformer_interrupt.pth"))
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
        log_print(f"中断权重已保存：{os.path.join(WEIGHT_DIR, 'segformer_interrupt.pth')}", log_file)
        log_print(f"断点已保存：{CHECKPOINT_PATH}", log_file)
        log_print(f"中断时 Epoch: {epoch}, 当前最优 mIoU: {best_miou:.4f}", log_file)

    # ===================== 训练结束，汇总最优指标 =====================
    train_total_time = time.time() - train_start_time
    log_print("\n" + "=" * 60, log_file)
    log_print("         SegFormer-B0 训练结束，最优指标汇总", log_file)
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

    # ---- 导出 CSV 指标文件 ----
    with open(CSV_LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_mIoU", "val_pixel_acc"])
        writer.writerows(log_records)
    log_print(f"指标 CSV 已导出：{CSV_LOG_PATH}", log_file)

    # ---- 关闭日志文件 ----
    log_file.close()
