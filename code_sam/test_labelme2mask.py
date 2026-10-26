"""
test_labelme2mask.py
测试专用：将 test_haikou_blind/labeled 中的 LabelMe JSON 转为标准二分类掩码。
所有输出限定在 test_haikou_blind 内部，不影响正式数据集。

使用环境：sam2_env
运行目录：blind_road_project 根目录
"""

import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELED_ROOT = PROJECT_ROOT / "test_haikou_blind" / "labeled"
MASK_OUTPUT_ROOT = PROJECT_ROOT / "test_haikou_blind" / "masks_final"

TRAVERSABLE_KEYWORDS = ["blind", "sidewalk", "tactile", "盲道", "人行道", "可通行区域"]


def cv2_imwrite_chinese(path, img):
    """Windows 中文路径安全写入图片"""
    ext = Path(path).suffix
    success, buf = cv2.imencode(ext, img)
    if success:
        buf.tofile(str(path))
    return success


def is_traversable_label(label_str):
    label_lower = label_str.lower()
    return any(kw in label_lower for kw in TRAVERSABLE_KEYWORDS)


def json_to_mask(json_path, output_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    img_h = data["imageHeight"]
    img_w = data["imageWidth"]
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for shape in data["shapes"]:
        label = shape["label"]
        if not is_traversable_label(label):
            continue
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
    cv2_imwrite_chinese(output_path, mask)


def main():
    parser = argparse.ArgumentParser(description="[测试] LabelMe JSON → 二分类掩码")
    parser.add_argument("--scenes", nargs="+", default=None, help="指定场景，默认全部")
    args = parser.parse_args()

    print(f"[测试模式] LabelMe 标注目录: {LABELED_ROOT}")
    print(f"[测试模式] 掩码输出目录: {MASK_OUTPUT_ROOT}")
    print("=" * 60)

    if args.scenes:
        scene_dirs = [LABELED_ROOT / s for s in args.scenes]
    else:
        scene_dirs = sorted([d for d in LABELED_ROOT.iterdir() if d.is_dir()]) if LABELED_ROOT.exists() else []

    if not scene_dirs:
        print(f"[WARN] 未找到标注目录: {LABELED_ROOT}")
        print("提示：请先用 LabelMe 打开 test_haikou_blind/raw_all 中的图片进行标注，")
        print("      将 JSON 保存到 test_haikou_blind/labeled/<场景名>/ 目录下。")
        return

    total, success, fail = 0, 0, 0
    for scene_dir in scene_dirs:
        if not scene_dir.exists():
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
    print(f"[测试] 转换完成！总计: {total}, 成功: {success}, 失败: {fail}")
    print(f"[测试] 掩码输出: {MASK_OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
