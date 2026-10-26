"""
sam_batch_generate_mask.py
遍历 raw_all 内所有场景图片，调用 SAM2 批量生成可通行区域粗分割掩码。
输出 png 单通道灰度图：可通行区域像素=1，背景=0。

使用环境：sam2_env
运行目录：blind_road_project 根目录

关键设计：
  - 默认 CPU 运行（device="cpu"），不抢占训练 GPU
  - 可通过 --device cuda 切换 GPU
  - 异常容错：单张图片失败不中断整体流程
  - 进度条打印
"""

import os
import argparse
import glob
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "dataset" / "haikou_blind" / "raw_all"
MASK_ROOT = PROJECT_ROOT / "dataset" / "haikou_blind" / "sam_masks"

# SAM2 checkpoint 路径（需提前下载）
SAM2_CKPT_DIR = PROJECT_ROOT / "sam2" / "checkpoints"

# 默认使用 sam2.1-hiera-tiny（轻量快速），也可切换为 small/base-plus/large
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
DEFAULT_CKPT = "sam2.1_hiera_tiny.pt"


def get_sam2_predictor(device):
    """构建 SAM2 自动掩码生成器，返回 SAM2AutomaticMaskGenerator 实例"""
    # 将 sam2 包路径加入 sys.path
    import sys
    sam2_repo = str(PROJECT_ROOT / "sam2")
    if sam2_repo not in sys.path:
        sys.path.insert(0, sam2_repo)

    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    ckpt_path = SAM2_CKPT_DIR / DEFAULT_CKPT
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"SAM2 权重文件不存在: {ckpt_path}\n"
            f"请先下载权重到 {SAM2_CKPT_DIR} 目录。\n"
            f"下载地址: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
        )

    print(f"加载 SAM2 模型: config={DEFAULT_CONFIG}, ckpt={ckpt_path}, device={device}")
    sam2_model = build_sam2(DEFAULT_CONFIG, str(ckpt_path), device=device)
    mask_generator = SAM2AutomaticMaskGenerator(sam2_model)
    return mask_generator


def select_blind_road_mask(masks, image_shape):
    """
    从 SAM2 输出的所有掩码中，启发式选择最可能是可通行区域的区域。
    策略：选择面积适中、位于图像下半部分、形状狭长的掩码。
    如果没有合适掩码，返回全零掩码。
    """
    h, w = image_shape[:2]
    best_mask = None
    best_score = -1

    for m in masks:
        seg = m["segmentation"]  # (H, W) bool
        area = seg.sum()
        # 面积过滤：太小(<0.5%图像)或太大(>40%图像)的跳过
        img_area = h * w
        if area < 0.005 * img_area or area > 0.4 * img_area:
            continue

        # 位置过滤：质心应在图像下半 70% 区域（可通行区域通常在地面）
        ys, xs = np.where(seg)
        cy = ys.mean() / h
        if cy < 0.3:  # 质心太靠上，跳过
            continue

        # 形状评分：长宽比越大越好（可通行区域狭长）
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        bw = max(x_max - x_min, 1)
        bh = max(y_max - y_min, 1)
        aspect = max(bw / bh, bh / bw)

        # 综合评分 = SAM置信度 * 0.3 + 长宽比 * 0.3 + 位置分 * 0.4
        stability = m.get("stability_score", m.get("predicted_iou", 0.5))
        pos_score = cy  # 越靠下分越高
        score = stability * 0.3 + min(aspect / 5.0, 1.0) * 0.3 + pos_score * 0.4

        if score > best_score:
            best_score = score
            best_mask = seg

    if best_mask is None:
        return np.zeros((h, w), dtype=np.uint8)
    return best_mask.astype(np.uint8)


def cv2_imread_chinese(path):
    """Windows 中文路径安全读取图片"""
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def cv2_imwrite_chinese(path, img):
    """Windows 中文路径安全写入图片"""
    ext = Path(path).suffix
    success, buf = cv2.imencode(ext, img)
    if success:
        buf.tofile(str(path))
    return success


def process_image(mask_generator, img_path, output_path, device):
    """处理单张图片，生成掩码并保存"""
    image = cv2_imread_chinese(img_path)
    if image is None:
        raise ValueError(f"无法读取图片: {img_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image_rgb)
    blind_mask = select_blind_road_mask(masks, image.shape)

    # 输出：可通行区域=1，背景=0（单通道灰度 png）
    cv2_imwrite_chinese(output_path, blind_mask)


def main():
    parser = argparse.ArgumentParser(description="SAM2 批量生成可通行区域粗分割掩码")
    parser.add_argument("--device", type=str, default="cpu",
                        help="推理设备: cpu 或 cuda（默认cpu，不占训练GPU）")
    parser.add_argument("--scenes", nargs="+", default=None,
                        help="指定场景子目录，默认处理 raw_all 下全部场景")
    args = parser.parse_args()

    print(f"推理设备: {args.device}")
    print(f"图片目录: {RAW_ROOT}")
    print(f"掩码输出: {MASK_ROOT}")
    print("=" * 60)

    # 收集所有待处理图片
    if args.scenes:
        scene_dirs = [RAW_ROOT / s for s in args.scenes]
    else:
        scene_dirs = sorted([d for d in RAW_ROOT.iterdir() if d.is_dir()])

    all_tasks = []  # (img_path, mask_output_path)
    for scene_dir in scene_dirs:
        if not scene_dir.exists():
            print(f"[WARN] 场景目录不存在: {scene_dir}")
            continue
        mask_dir = MASK_ROOT / scene_dir.name
        mask_dir.mkdir(parents=True, exist_ok=True)
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            for img_path in sorted(scene_dir.glob(ext)):
                mask_path = mask_dir / (img_path.stem + ".png")
                # 跳过已存在的掩码（支持断点续跑）
                if mask_path.exists():
                    continue
                all_tasks.append((img_path, mask_path))

    if not all_tasks:
        print("无待处理图片（可能已全部生成或目录为空）")
        return

    print(f"待处理图片: {len(all_tasks)} 张")
    print("=" * 60)

    # 加载 SAM2 模型
    mask_generator = get_sam2_predictor(args.device)

    # 批量处理
    success, fail = 0, 0
    for img_path, mask_path in tqdm(all_tasks, desc="SAM2 推理", ncols=80):
        try:
            process_image(mask_generator, img_path, mask_path, args.device)
            success += 1
        except Exception as e:
            fail += 1
            print(f"\n[ERROR] {img_path.name}: {e}")

    print("=" * 60)
    print(f"完成！成功: {success}, 失败: {fail}, 输出: {MASK_ROOT}")


if __name__ == "__main__":
    main()
