"""Exact-value CPU preprocessing/postprocessing for the fixed binary deployment."""
import cv2
import numpy as np
from PIL import Image
import torch

HEIGHT, WIDTH = 768, 1024
MEAN = [.485, .456, .406]
STD = [.229, .224, .225]
# Match the frozen torchvision ToTensor/Normalize floating-point operation order.
_BYTE_VALUES = torch.arange(256, dtype=torch.float32).div(255.)
NORMALIZE_LUT = torch.stack([_BYTE_VALUES.sub(m).div(s) for m, s in zip(MEAN, STD)], dim=1).numpy().reshape(256, 1, 3)
OVERLAY_LUT = (.6 * np.arange(256, dtype=np.float64)[:, None] + .4 * np.array([40, 210, 40])).astype(np.uint8).reshape(256, 1, 3)


def read_bgr(path):
    value = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if value is None: raise ValueError('Image could not be decoded: ' + str(path))
    return value


def preprocess_bgr(frame):
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError('Expected uint8 BGR image with three channels')
    rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((WIDTH, HEIGHT), Image.BILINEAR)
    normalized = cv2.LUT(np.asarray(rgb), NORMALIZE_LUT)
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])


def binary_mask(logits):
    if logits.shape != (1, 2, HEIGHT, WIDTH) or not np.isfinite(logits).all():
        raise ValueError('Expected finite batch1 binary logits at the fixed resolution')
    # argmax selects class0 on a tie; strict greater-than preserves this rule.
    return (logits[0, 1] > logits[0, 0]).astype(np.uint8)


def restore_mask(mask, frame):
    return cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)


def overlay_bgr(frame, mask):
    if mask.shape != frame.shape[:2] or mask.dtype != np.uint8 or not np.isin(mask, [0, 1]).all():
        raise ValueError('Expected binary uint8 mask aligned to the frame')
    # LUT preserves the old float64 blend followed by uint8 truncation exactly.
    # copyTo avoids allocating/selecting a large foreground pixel matrix.
    output = frame.copy()
    cv2.copyTo(cv2.LUT(frame, OVERLAY_LUT), mask, output)
    return output
