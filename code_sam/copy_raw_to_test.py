"""
copy_raw_to_test.py
将 dataset/haikou_blind/raw_all 中的图片完整复制到 test_haikou_blind/raw_all。
仅复制，不移动、不删除原图。

使用环境：sam2_env
运行目录：blind_road_project 根目录
"""

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "dataset" / "haikou_blind" / "raw_all"
DST_ROOT = PROJECT_ROOT / "test_haikou_blind" / "raw_all"

SCENES = ["coastwalkway", "hospital", "main_road", "park", "station", "university"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    if not SRC_ROOT.exists():
        print(f"[ERROR] 源目录不存在: {SRC_ROOT}")
        return

    total = 0
    for scene in SCENES:
        src_dir = SRC_ROOT / scene
        dst_dir = DST_ROOT / scene
        dst_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            print(f"[WARN] 场景目录不存在，跳过: {src_dir}")
            continue

        count = 0
        for f in sorted(src_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                shutil.copy2(str(f), str(dst_dir / f.name))
                count += 1
        print(f"[{scene}] 复制 {count} 张 → {dst_dir}")
        total += count

    print(f"\n完成！共复制 {total} 张图片到 {DST_ROOT}")


if __name__ == "__main__":
    main()
