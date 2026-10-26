"""
compare_masks.py
简易对比查看：SAM 原始掩码 vs 人工修正后掩码。
将原图、SAM掩码、修正掩码横向拼接保存为对比图。

使用环境：sam2_env
运行目录：blind_road_project 根目录
"""

import argparse
import numpy as np
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENES = ["coastwalkway", "hospital", "main_road", "park", "station", "university"]


def cv2_imread_chinese(path, flags=cv2.IMREAD_COLOR):
    """Windows 中文路径安全读取图片"""
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, flags)
    return img


def cv2_imwrite_chinese(path, img):
    """Windows 中文路径安全写入图片"""
    ext = Path(path).suffix
    success, buf = cv2.imencode(ext, img)
    if success:
        buf.tofile(str(path))
    return success


def main():
    parser = argparse.ArgumentParser(description="对比 SAM 掩码 vs 人工修正掩码")
    parser.add_argument("--scene", type=str, default=None, help="指定场景名，默认遍历全部")
    parser.add_argument("--max_images", type=int, default=10, help="每个场景最多对比几张（默认10）")
    args = parser.parse_args()

    raw_root = PROJECT_ROOT / "test_haikou_blind" / "raw_all"
    sam_mask_root = PROJECT_ROOT / "test_haikou_blind" / "sam_masks"
    final_mask_root = PROJECT_ROOT / "test_haikou_blind" / "masks_final"
    compare_root = PROJECT_ROOT / "test_haikou_blind" / "compare"
    compare_root.mkdir(parents=True, exist_ok=True)

    scenes = [args.scene] if args.scene else SCENES
    total = 0

    for scene in scenes:
        img_dir = raw_root / scene
        sam_dir = sam_mask_root / scene
        final_dir = final_mask_root / scene
        out_dir = compare_root / scene
        out_dir.mkdir(parents=True, exist_ok=True)

        if not img_dir.exists():
            print(f"[WARN] 场景目录不存在: {scene}")
            continue

        count = 0
        for img_path in sorted(img_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            if count >= args.max_images:
                break

            sam_path = sam_dir / (img_path.stem + ".png")
            final_path = final_dir / (img_path.stem + ".png")

            # 读取原图
            img = cv2_imread_chinese(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            # 读取 SAM 掩码（绿色叠加）
            if sam_path.exists():
                sam_mask = cv2_imread_chinese(sam_path, cv2.IMREAD_GRAYSCALE)
                sam_overlay = img.copy()
                sam_overlay[sam_mask > 0] = [0, 255, 0]  # 绿色
                sam_vis = cv2.addWeighted(img, 0.6, sam_overlay, 0.4, 0)
            else:
                sam_vis = np.zeros_like(img)
                cv2.putText(sam_vis, "No SAM Mask", (10, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 读取修正掩码（红色叠加）
            if final_path.exists():
                final_mask = cv2_imread_chinese(final_path, cv2.IMREAD_GRAYSCALE)
                final_overlay = img.copy()
                final_overlay[final_mask > 0] = [0, 0, 255]  # 红色
                final_vis = cv2.addWeighted(img, 0.6, final_overlay, 0.4, 0)
            else:
                final_vis = np.zeros_like(img)
                cv2.putText(final_vis, "No Final Mask", (10, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 统一高度
            target_h = min(img.shape[0], 512)
            scale = target_h / img.shape[0]
            target_w = int(img.shape[1] * scale)
            dim = (target_w, target_h)

            img_r = cv2.resize(img, dim)
            sam_r = cv2.resize(sam_vis, dim)
            final_r = cv2.resize(final_vis, dim)

            # 添加标题
            cv2.putText(img_r, "Original", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(sam_r, "SAM Mask (Green)", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(final_r, "Corrected Mask (Red)", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # 横向拼接
            compare = np.hstack([img_r, sam_r, final_r])
            out_path = out_dir / f"{img_path.stem}_compare.jpg"
            cv2_imwrite_chinese(out_path, compare)
            count += 1
            total += 1

        print(f"[{scene}] 生成 {count} 张对比图 → {out_dir}")

    print(f"\n完成！共生成 {total} 张对比图，保存至 {compare_root}")


if __name__ == "__main__":
    main()
