"""Protocol v1 metrics, independent of historical evaluation scripts.

Boundary IoU follows Bowen Cheng's boundary_iou_example.py:
https://gist.github.com/bowenc0221/71f7a02afee92646ca05efeeb14d687d
Only metric definitions are supplied here; no experiments or files are written.
"""
import cv2
import numpy as np


def confusion_matrix_numpy(pred, target, num_classes=2):
    pred = np.asarray(pred)
    target = np.asarray(target)
    if pred.shape != target.shape:
        raise ValueError("Prediction and target shapes must match")
    valid = (target >= 0) & (target < num_classes)
    if np.any((pred[valid] < 0) | (pred[valid] >= num_classes)):
        raise ValueError("Prediction contains an invalid class")
    indices = num_classes * target[valid].astype(np.int64) + pred[valid].astype(np.int64)
    return np.bincount(indices.ravel(), minlength=num_classes ** 2).reshape(num_classes, num_classes)


def scores_from_confusion(confusion):
    conf = np.asarray(confusion, dtype=np.float64)
    intersection = np.diag(conf)
    union = conf.sum(axis=0) + conf.sum(axis=1) - intersection
    # Match the established project convention for an absent class.
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    return {"iou": iou.tolist(), "miou": float(iou.mean()),
            "pixel_acc": float(intersection.sum() / conf.sum()) if conf.sum() else 0.0}


def boundary_band(mask, dilation_ratio=0.02, width_pixels=None):
    mask = np.asarray(mask, dtype=bool).astype(np.uint8)
    if mask.ndim != 2:
        raise ValueError("Boundary masks must be two-dimensional")
    height, width = mask.shape
    distance = max(1, int(round(dilation_ratio * np.hypot(height, width)))) if width_pixels is None else int(width_pixels)
    if distance < 1:
        raise ValueError("Boundary width must be positive")
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    eroded = cv2.erode(padded, np.ones((3, 3), dtype=np.uint8), iterations=distance)
    return (mask - eroded[1:height + 1, 1:width + 1]) > 0


def boundary_iou_author(target, pred, dilation_ratio=0.02, width_pixels=None):
    if np.asarray(target).shape != np.asarray(pred).shape:
        raise ValueError("Prediction and target shapes must match")
    target_band = boundary_band(target, dilation_ratio, width_pixels)
    pred_band = boundary_band(pred, dilation_ratio, width_pixels)
    union = int(np.count_nonzero(target_band | pred_band))
    # Both absent: excluded from the per-image BIoU mean, with count reported.
    return float(np.count_nonzero(target_band & pred_band) / union) if union else None
