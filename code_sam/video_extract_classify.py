"""
video_extract_classify.py
遍历6类场景视频文件，按固定间隔均匀抽帧，自动分类存入 raw_all 对应子文件夹。
支持去重（SSIM 相似度阈值），避免连续帧高度重复。

使用环境：sam2_env
运行目录：blind_road_project 根目录

目录结构约定：
  视频输入：dataset/haikou_blind/videos/<场景名>/*.mp4
  图片输出：dataset/haikou_blind/raw_all/<场景名>/*.jpg

  场景名：coastwalkway, hospital, main_road, park, station, university
"""

import os
import argparse
import cv2
import numpy as np
from pathlib import Path

# ======================== 配置 ========================
SCENES = ["coastwalkway", "hospital", "main_road", "park", "station", "university"]

# 项目根目录（脚本位于 code_sam/，上一级即为项目根目录）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_ROOT = PROJECT_ROOT / "dataset" / "haikou_blind" / "videos"
OUTPUT_ROOT = PROJECT_ROOT / "dataset" / "haikou_blind" / "raw_all"

# 默认抽帧间隔（每隔 N 帧抽取一帧）
DEFAULT_INTERVAL = 30
# 去重阈值：SSIM > 此值视为重复帧，跳过（范围 0~1，越接近1越严格）
SIMILARITY_THRESHOLD = 0.92


def compute_ssim_gray(img1, img2):
    """简易灰度 SSIM 计算（无需 skimage 依赖）"""
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)
    # 缩小到 128x128 加速
    img1 = cv2.resize(img1, (128, 128))
    img2 = cv2.resize(img2, (128, 128))

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def extract_frames_from_video(video_path, interval, output_dir, scene_name, sim_thresh):
    """从单个视频抽帧，返回抽取数量"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] 无法打开视频: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_stem = Path(video_path).stem
    saved_count = 0
    prev_frame = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            # 去重检查
            if prev_frame is not None and sim_thresh > 0:
                ssim_val = compute_ssim_gray(prev_frame, frame)
                if ssim_val > sim_thresh:
                    frame_idx += 1
                    continue
            # 保存
            out_name = f"{scene_name}_{video_stem}_{frame_idx:06d}.jpg"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), frame)
            saved_count += 1
            prev_frame = frame.copy()
        frame_idx += 1

    cap.release()
    return saved_count


def main():
    parser = argparse.ArgumentParser(description="视频抽帧并按场景分类")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL,
                        help=f"抽帧间隔，每N帧抽1帧（默认{DEFAULT_INTERVAL}）")
    parser.add_argument("--sim_thresh", type=float, default=SIMILARITY_THRESHOLD,
                        help=f"去重SSIM阈值，高于此值跳过（默认{SIMILARITY_THRESHOLD}，设0关闭去重）")
    args = parser.parse_args()

    print(f"抽帧间隔: 每 {args.interval} 帧")
    print(f"去重阈值: SSIM > {args.sim_thresh} 视为重复")
    print(f"视频目录: {VIDEO_ROOT}")
    print(f"输出目录: {OUTPUT_ROOT}")
    print("=" * 60)

    total_saved = 0
    for scene in SCENES:
        scene_video_dir = VIDEO_ROOT / scene
        scene_output_dir = OUTPUT_ROOT / scene
        scene_output_dir.mkdir(parents=True, exist_ok=True)

        if not scene_video_dir.exists():
            print(f"[WARN] 场景目录不存在，跳过: {scene_video_dir}")
            continue

        video_files = sorted(list(scene_video_dir.glob("*.mp4")) +
                             list(scene_video_dir.glob("*.avi")) +
                             list(scene_video_dir.glob("*.MOV")) +
                             list(scene_video_dir.glob("*.mov")))
        if not video_files:
            print(f"[WARN] 场景 {scene} 无视频文件，跳过")
            continue

        scene_count = 0
        for vf in video_files:
            n = extract_frames_from_video(vf, args.interval, scene_output_dir, scene, args.sim_thresh)
            scene_count += n
            print(f"  {vf.name} → 抽取 {n} 帧")

        print(f"[{scene}] 共抽取 {scene_count} 帧 → {scene_output_dir}")
        total_saved += scene_count

    print("=" * 60)
    print(f"全部完成！共抽取 {total_saved} 帧，保存至 {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
