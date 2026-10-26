"""
test_sam_mask.py
测试专用：读取 test_haikou_blind 图片生成 SAM 粗掩码。
默认 CPU 运行，不占用训练 GPU。
所有输出限定在 test_haikou_blind 内部，不影响正式数据集。

使用环境：sam2_env
运行目录：blind_road_project 根目录
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "test_haikou_blind" / "raw_all"
MASK_ROOT = PROJECT_ROOT / "test_haikou_blind" / "sam_masks"

SAM2_CKPT_DIR = PROJECT_ROOT / "sam2" / "checkpoints"
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
DEFAULT_CKPT = "sam2.1_hiera_tiny.pt"


def get_sam2_predictor(device):
    sam2_repo = str(PROJECT_ROOT / "sam2")
    if sam2_repo not in sys.path:
        sys.path.insert(0, sam2_repo)
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    ckpt_path = SAM2_CKPT_DIR / DEFAULT_CKPT
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"SAM2 权重文件不存在: {ckpt_path}\n"
            f"请先下载权重到 {SAM2_CKPT_DIR} 目录。"
        )
    print(f"加载 SAM2 模型: config={DEFAULT_CONFIG}, ckpt={ckpt_path}, device={device}")
    sam2_model = build_sam2(DEFAULT_CONFIG, str(ckpt_path), device=device)
    mask_generator = SAM2AutomaticMaskGenerator(sam2_model)
    return mask_generator


def select_blind_road_mask(masks, image_shape):
    h, w = image_shape[:2]
    best_mask = None
    best_score = -1
    for m in masks:
        seg = m["segmentation"]
        area = seg.sum()
        img_area = h * w
        if area < 0.005 * img_area or area > 0.4 * img_area:
            continue
        ys, xs = np.where(seg)
        cy = ys.mean() / h
        if cy < 0.3:
            continue
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        bw = max(x_max - x_min, 1)
        bh = max(y_max - y_min, 1)
        aspect = max(bw / bh, bh / bw)
        stability = m.get("stability_score", m.get("predicted_iou", 0.5))
        pos_score = cy
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
    image = cv2_imread_chinese(img_path)
    if image is None:
        raise ValueError(f"无法读取图片: {img_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image_rgb)
    blind_mask = select_blind_road_mask(masks, image.shape)
    cv2_imwrite_chinese(output_path, blind_mask)


def main():
    parser = argparse.ArgumentParser(description="[测试] SAM2 批量生成可通行区域粗掩码")
    parser.add_argument("--device", type=str, default="cpu", help="cpu 或 cuda（默认cpu）")
    parser.add_argument("--scenes", nargs="+", default=None, help="指定场景，默认全部")
    args = parser.parse_args()

    print(f"[测试模式] 推理设备: {args.device}")
    print(f"[测试模式] 图片目录: {RAW_ROOT}")
    print(f"[测试模式] 掩码输出: {MASK_ROOT}")
    print("=" * 60)

    if args.scenes:
        scene_dirs = [RAW_ROOT / s for s in args.scenes]
    else:
        scene_dirs = sorted([d for d in RAW_ROOT.iterdir() if d.is_dir()]) if RAW_ROOT.exists() else []

    if not scene_dirs:
        print(f"[ERROR] 未找到图片目录，请先运行 copy_raw_to_test.py")
        return

    all_tasks = []
    for scene_dir in scene_dirs:
        if not scene_dir.exists():
            continue
        mask_dir = MASK_ROOT / scene_dir.name
        mask_dir.mkdir(parents=True, exist_ok=True)
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            for img_path in sorted(scene_dir.glob(ext)):
                mask_path = mask_dir / (img_path.stem + ".png")
                if mask_path.exists():
                    continue
                all_tasks.append((img_path, mask_path))

    if not all_tasks:
        print("无待处理图片（可能已全部生成或目录为空）")
        return

    print(f"待处理图片: {len(all_tasks)} 张")
    print("=" * 60)

    mask_generator = get_sam2_predictor(args.device)

    success, fail = 0, 0
    for img_path, mask_path in tqdm(all_tasks, desc="SAM2推理[测试]", ncols=80):
        try:
            process_image(mask_generator, img_path, mask_path, args.device)
            success += 1
        except Exception as e:
            fail += 1
            print(f"\n[ERROR] {img_path.name}: {e}")

    print("=" * 60)
    print(f"[测试] 完成！成功: {success}, 失败: {fail}, 输出: {MASK_ROOT}")


if __name__ == "__main__":
    main()
