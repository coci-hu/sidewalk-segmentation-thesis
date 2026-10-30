"""Reference image/video prototype; original preprocessing and postprocessing.

Image: --models models.json --input scene.jpg --output overlay.png --headless
Video: --models models.json --input clip.mp4 --output overlay.mp4
Keys1/2 switch exported models, Q exits. Explicit --backend cpu (default) or cuda.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
from reference_deployment_pipeline import read_bgr, preprocess_bgr, binary_mask, restore_mask, overlay_bgr
from verified_deployment_runtime import VerifiedRuntime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model-index', type=int, default=0)
    p.add_argument('--backend', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--max-frames', type=int)
    p.add_argument('--source-note', default='User-provided media', help='Describe constructed validation clips honestly')
    a = p.parse_args()
    run(a)


def run(a):
    evidence=a.output.with_suffix(a.output.suffix+'.evidence')
    with VerifiedRuntime(a.models,a.backend,evidence) as runtime:
        process(a,runtime)


def process(a,runtime):
    if a.output.exists() or a.output.with_suffix(a.output.suffix + '.json').exists():
        raise FileExistsError('Use a new output file; existing results are preserved')
    if not a.input.is_file():
        raise FileNotFoundError(a.input)
    config=runtime.config
    models=runtime.models
    assert 0 <= a.model_index < len(models)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    provider=runtime.provider
    cv2.setNumThreads(1)
    runtime.session_for(a.model_index)
    image_mode = a.input.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    capture, writer = None, None
    if not image_mode:
        capture = cv2.VideoCapture(str(a.input))
        if not capture.isOpened():
            raise RuntimeError('Video could not be opened')
    records = []
    index = a.model_index
    try:
        while True:
            runtime.session_for(index)
            began = time.perf_counter()
            if image_mode:
                frame = read_bgr(a.input)
                ok = frame is not None
            else:
                ok, frame = capture.read()
            if not ok:
                if not records:
                    raise RuntimeError('No readable frames')
                break
            tensor = preprocess_bgr(frame)
            infer_start = time.perf_counter()
            logits = runtime.infer(index,tensor)
            infer_ms = (time.perf_counter() - infer_start) * 1000
            assert np.isfinite(logits).all()
            low_resolution_mask=binary_mask(logits)
            mask=restore_mask(low_resolution_mask,frame)
            selected=mask==1
            overlay=overlay_bgr(frame,mask)
            processing_ms = (time.perf_counter() - began) * 1000
            font = min(0.65, max(0.35, frame.shape[1] / 1800))
            panel = overlay.copy()
            cv2.rectangle(panel, (0, 0), (frame.shape[1], 76), (0, 0, 0), -1)
            overlay[:76] = cv2.addWeighted(panel[:76], 0.7, overlay[:76], 0.3, 0)
            lines = [f'{models[index]["name"]} | {a.backend.upper()}',
                f'Inference {infer_ms:.1f} ms | frame {processing_ms:.1f} ms',
                'Sidewalk segmentation research demo']
            for line_number, label in enumerate(lines):
                cv2.putText(overlay, label, (10, 21 + 23 * line_number), cv2.FONT_HERSHEY_SIMPLEX,
                    font, (255, 255, 255), 1, cv2.LINE_AA)
            records.append({'frame': len(records), 'model': models[index]['name'], 'inference_ms': infer_ms,
                'read_preprocess_infer_overlay_ms': processing_ms, 'foreground_fraction': float(selected.mean()),
                'mask_sha256': hashlib.sha256(low_resolution_mask.tobytes()).hexdigest()})
            if image_mode:
                success, encoded = cv2.imencode(a.output.suffix, overlay)
                if not success: raise RuntimeError('Image encoder failed')
                encoded.tofile(a.output)
            else:
                if writer is None:
                    fps = capture.get(cv2.CAP_PROP_FPS)
                    if not np.isfinite(fps) or fps <= 0: fps = 25.0
                    writer = cv2.VideoWriter(str(a.output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0]))
                    if not writer.isOpened(): raise RuntimeError('Video writer failed')
                writer.write(overlay)
            if not a.headless:
                cv2.imshow('Sidewalk prototype 1/2 switch Q exit', overlay)
                key = cv2.waitKey(0 if image_mode else 1) & 0xff
                if key in (ord('q'), 27): break
                if ord('1') <= key < ord('1') + len(models):
                    index = key - ord('1')
                    if image_mode:
                        continue
            if image_mode or (a.max_frames and len(records) >= a.max_frames): break
    finally:
        if capture: capture.release()
        if writer: writer.release()
        if not a.headless: cv2.destroyAllWindows()
    report = {'input': str(a.input.resolve()), 'output': str(a.output.resolve()), 'backend_requested': provider,
        'frames': records, 'speed_is_not_a_controlled_benchmark': True,
        'processing_excludes_video_write_display_and_session_initialization': True,
        'semantic_sidewalk_only_not_safe_path_planning': True,
        'execution_verified': runtime.proofs, 'models_manifest_sha256': runtime.manifest_hash,
        'pipeline': 'reference_torchvision_argmax_numpy_overlay',
        'CPU_full500_validated': False, 'source_note': a.source_note}
    a.output.with_suffix(a.output.suffix + '.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'frames': len(records), 'output': str(a.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
