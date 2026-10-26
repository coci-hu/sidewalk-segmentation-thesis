"""
test_split_dataset.py
测试专用：将 test_haikou_blind 中的原图+掩码按 8:2 划分训练/验证集。
所有输出限定在 test_haikou_blind 内部，不影响正式数据集。

使用环境：sam2_env
运行目录：blind_road_project 根目录
"""

import random
import shutil
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "test_haikou_blind" / "raw_all"
MASK_ROOT = PROJECT_ROOT / "test_haikou_blind" / "masks_final"
SPLIT_ROOT = PROJECT_ROOT / "test_haikou_blind" / "split"

SCENES = ["coastwalkway", "hospital", "main_road", "park", "station", "university"]
DEFAULT_RATIO = 0.8
SEED = 42


def main():
    parser = argparse.ArgumentParser(description="[测试] 海口可通行区域数据集划分 train/val")
    parser.add_argument("--ratio", type=float, default=DEFAULT_RATIO, help=f"训练集比例（默认{DEFAULT_RATIO}）")
    parser.add_argument("--seed", type=int, default=SEED, help=f"随机种子（默认{SEED}）")
    args = parser.parse_args()

    random.seed(args.seed)
    print(f"[测试模式] 训练集比例: {args.ratio:.1%}")
    print(f"[测试模式] 原图目录: {RAW_ROOT}")
    print(f"[测试模式] 掩码目录: {MASK_ROOT}")
    print(f"[测试模式] 输出目录: {SPLIT_ROOT}")
    print("=" * 60)

    total_train, total_val = 0, 0

    for scene in SCENES:
        img_dir = RAW_ROOT / scene
        mask_dir = MASK_ROOT / scene

        if not img_dir.exists():
            print(f"[WARN] 原图目录不存在: {img_dir}，跳过")
            continue

        # 收集有配对掩码的图片
        paired = []
        for img_path in sorted(img_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            mask_path = mask_dir / (img_path.stem + ".png")
            if mask_path.exists():
                paired.append((img_path, mask_path))

        if not paired:
            print(f"[WARN] {scene} 无配对样本，跳过")
            continue

        random.shuffle(paired)
        n_train = int(len(paired) * args.ratio)
        train_pairs = paired[:n_train]
        val_pairs = paired[n_train:]

        for split_name in ["train", "val"]:
            (SPLIT_ROOT / split_name / "images" / scene).mkdir(parents=True, exist_ok=True)
            (SPLIT_ROOT / split_name / "masks" / scene).mkdir(parents=True, exist_ok=True)

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
    print(f"[测试] 划分完成！训练集: {total_train}, 验证集: {total_val}")
    print(f"[测试] 输出目录: {SPLIT_ROOT}")


if __name__ == "__main__":
    main()
