"""
split_haikou_dataset.py
将标注完毕的原图 + 掩码按 8:2 划分训练集/验证集，保证 6 类场景样本均匀分配。
输出标准数据集目录结构，可供 PyTorch Dataset 直接读取。

使用环境：sam2_env
运行目录：blind_road_project 根目录

输入目录结构：
  dataset/haikou_blind/raw_all/<场景名>/*.jpg     （原图）
  dataset/haikou_blind/masks_final/<场景名>/*.png  （人工修正后的掩码）

输出目录结构：
  dataset/haikou_blind/split/
    train/
      images/<场景名>/*.jpg
      masks/<场景名>/*.png
    val/
      images/<场景名>/*.jpg
      masks/<场景名>/*.png
"""

import os
import random
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "dataset" / "haikou_blind" / "raw_all"
MASK_ROOT = PROJECT_ROOT / "dataset" / "haikou_blind" / "masks_final"
SPLIT_ROOT = PROJECT_ROOT / "dataset" / "haikou_blind" / "split"

SCENES = ["coastwalkway", "hospital", "main_road", "park", "station", "university"]
DEFAULT_RATIO = 0.8  # 训练集比例
SEED = 42


def main():
    parser = argparse.ArgumentParser(description="海口可通行区域数据集划分 train/val")
    parser.add_argument("--ratio", type=float, default=DEFAULT_RATIO,
                        help=f"训练集比例（默认{DEFAULT_RATIO}）")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"随机种子（默认{SEED}）")
    args = parser.parse_args()

    random.seed(args.seed)
    print(f"训练集比例: {args.ratio:.1%}")
    print(f"随机种子: {args.seed}")
    print(f"原图目录: {RAW_ROOT}")
    print(f"掩码目录: {MASK_ROOT}")
    print(f"输出目录: {SPLIT_ROOT}")
    print("=" * 60)

    total_train, total_val = 0, 0

    for scene in SCENES:
        img_dir = RAW_ROOT / scene
        mask_dir = MASK_ROOT / scene

        if not img_dir.exists():
            print(f"[WARN] 原图目录不存在: {img_dir}，跳过")
            continue

        # 收集有配对掩码的图片
        # 注意：海口原图同时存在 .jpg 与 .jpeg 两种扩展名（23 张中 11 张是 .jpeg）。
        # 早期版本这里只写 glob("*.jpg")，会静默漏掉全部 .jpeg，漏掉率约 48%。
        img_files = []
        for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"):
            img_files.extend(img_dir.glob(ext))
        paired = []
        for img_path in sorted(img_files):
            mask_path = mask_dir / (img_path.stem + ".png")
            if mask_path.exists():
                paired.append((img_path, mask_path))
            else:
                # 也尝试同名 jpg 掩码
                mask_path_jpg = mask_dir / (img_path.stem + ".jpg")
                if mask_path_jpg.exists():
                    paired.append((img_path, mask_path_jpg))

        if not paired:
            print(f"[WARN] {scene} 无配对样本，跳过")
            continue

        # 按场景内随机打乱并划分
        random.shuffle(paired)
        n_train = int(len(paired) * args.ratio)
        # 单图/极少样本场景的保护：int(1*0.8)=0 会让训练集变空。
        # 样本数 >=2 时保证 train 与 val 都非空；只有 1 张时该场景无法划分，全部放 train。
        if len(paired) >= 2:
            n_train = max(1, min(len(paired) - 1, n_train))
        else:
            n_train = len(paired)
            print(f"[WARN] {scene} 只有 {len(paired)} 张，无法划分 train/val，全部放入 train")
        if len(paired) < 5:
            print(f"[WARN] {scene} 仅 {len(paired)} 张，划分结果不具统计意义")
        train_pairs = paired[:n_train]
        val_pairs = paired[n_train:]

        # 创建输出目录
        for split_name in ["train", "val"]:
            (SPLIT_ROOT / split_name / "images" / scene).mkdir(parents=True, exist_ok=True)
            (SPLIT_ROOT / split_name / "masks" / scene).mkdir(parents=True, exist_ok=True)

        # 复制文件
        for img_path, mask_path in train_pairs:
            shutil.copy2(str(img_path), str(SPLIT_ROOT / "train" / "images" / scene / img_path.name))
            shutil.copy2(str(mask_path), str(SPLIT_ROOT / "train" / "masks" / scene / mask_path.name))

        for img_path, mask_path in val_pairs:
            shutil.copy2(str(img_path), str(SPLIT_ROOT / "val" / "images" / scene / img_path.name))
            shutil.copy2(str(mask_path), str(SPLIT_ROOT / "val" / "masks" / scene / mask_path.name))

        total_train += len(train_pairs)
        total_val += len(val_pairs)
        print(f"[{scene}] 总计 {len(paired)} 对 → 训练 {len(train_pairs)}, 验证 {len(val_pairs)}")

    print("=" * 60)
    print(f"划分完成！训练集: {total_train}, 验证集: {total_val}")
    print(f"输出目录: {SPLIT_ROOT}")


if __name__ == "__main__":
    main()
