"""
labelme_json_to_mask.py
批量将 LabelMe 人工修正后的标注 JSON 文件，转换为标准二分类 mask 图片。
标签规则：可通行区域=1，背景=0，与 Cityscapes 数据集完全统一。

使用环境：sam2_env（需要 labelme 包）
运行目录：blind_road_project 根目录

目录结构约定：
  输入：dataset/haikou_blind/labeled/<场景名>/*.json  （LabelMe 标注文件，与原图同目录）
  输出：dataset/haikou_blind/masks_final/<场景名>/*.png（二值掩码，可通行区域=1 背景=0）

  LabelMe JSON 中 label 字段包含 "blind" 或 "sidewalk" 或 "tactile" 的视为可通行区域。
"""

import os
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELED_ROOT = PROJECT_ROOT / "dataset" / "haikou_blind" / "labeled"
MASK_OUTPUT_ROOT = PROJECT_ROOT / "dataset" / "haikou_blind" / "masks_final"

# 可通行区域标签关键词（LabelMe 中 label 字段包含这些词的视为可通行区域）
TRAVERSABLE_KEYWORDS = ["blind", "sidewalk", "tactile", "盲道", "人行道", "可通行区域"]


def is_traversable_label(label_str):
    """判断 LabelMe 标签是否为可通行区域"""
    label_lower = label_str.lower()
    return any(kw in label_lower for kw in TRAVERSABLE_KEYWORDS)


def json_to_mask(json_path, output_path):
    """将单个 LabelMe JSON 转为二值掩码"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_h = data["imageHeight"]
    img_w = data["imageWidth"]
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    for shape in data["shapes"]:
        label = shape["label"]
        if not is_traversable_label(label):
            continue  # 非可通行区域标签，跳过

        shape_type = shape["shape_type"]
        points = np.array(shape["points"], dtype=np.int32)

        if shape_type == "polygon":
            cv2.fillPoly(mask, [points], color=1)
        elif shape_type == "rectangle":
            cv2.rectangle(mask, tuple(points[0]), tuple(points[1]), color=1, thickness=-1)
        elif shape_type == "line":
            cv2.polylines(mask, [points], isClosed=False, color=1, thickness=3)
        elif shape_type == "circle":
            center = tuple(points[0])
            radius = int(np.linalg.norm(points[1] - points[0]))
            cv2.circle(mask, center, radius, color=1, thickness=-1)

    cv2.imwrite(str(output_path), mask)


def main():
    parser = argparse.ArgumentParser(description="LabelMe JSON → 二分类掩码")
    parser.add_argument("--scenes", nargs="+", default=None,
                        help="指定场景子目录，默认处理 labeled 下全部场景")
    args = parser.parse_args()

    print(f"LabelMe 标注目录: {LABELED_ROOT}")
    print(f"掩码输出目录: {MASK_OUTPUT_ROOT}")
    print("=" * 60)

    if args.scenes:
        scene_dirs = [LABELED_ROOT / s for s in args.scenes]
    else:
        scene_dirs = sorted([d for d in LABELED_ROOT.iterdir() if d.is_dir()]) if LABELED_ROOT.exists() else []

    if not scene_dirs:
        print(f"[WARN] 未找到标注目录，请确认 {LABELED_ROOT} 下有场景子文件夹")
        return

    total, success, fail = 0, 0, 0
    for scene_dir in scene_dirs:
        if not scene_dir.exists():
            print(f"[WARN] 场景目录不存在: {scene_dir}")
            continue

        out_dir = MASK_OUTPUT_ROOT / scene_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        json_files = sorted(scene_dir.glob("*.json"))
        if not json_files:
            print(f"[WARN] {scene_dir.name} 无 JSON 文件，跳过")
            continue

        print(f"\n[{scene_dir.name}] 发现 {len(json_files)} 个标注文件")
        for jf in tqdm(json_files, desc=scene_dir.name, ncols=80):
            total += 1
            try:
                out_path = out_dir / (jf.stem + ".png")
                json_to_mask(jf, out_path)
                success += 1
            except Exception as e:
                fail += 1
                print(f"\n[ERROR] {jf.name}: {e}")

    print("\n" + "=" * 60)
    print(f"转换完成！总计: {total}, 成功: {success}, 失败: {fail}")
    print(f"掩码输出: {MASK_OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
