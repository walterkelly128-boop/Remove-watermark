from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

MAX_DETECT_DIM = 1600


def _kernel(size: int, shape=cv2.MORPH_ELLIPSE):
    size = max(3, int(size))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(shape, (size, size))


def _dilate(mask: np.ndarray, k: int) -> np.ndarray:
    return cv2.dilate(mask, _kernel(k), iterations=1)


def _tight_component_mask(binary: np.ndarray, h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area < max(8, int(h * w * 0.000002)):
            continue
        if area > h * w * 0.015:
            continue
        ratio = cw / max(ch, 1)
        if ratio < 0.25 or ratio > 35:
            continue
        near_edge = x < w * 0.22 or y < h * 0.22 or x + cw > w * 0.78 or y + ch > h * 0.78
        if not near_edge:
            continue
        if ch > h * 0.075 or cw > w * 0.55:
            continue
        out = np.maximum(out, (labels == i).astype(np.uint8) * 255)
    return out


def _detect_small(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = np.zeros((h, w), np.uint8)

    for block in (21, 35, 51):
        bw = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block, 7,
        )
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, _kernel(2, cv2.MORPH_RECT))
        mask = np.maximum(mask, _tight_component_mask(bw, h, w))

    corner_w = max(64, int(w * 0.22))
    corner_h = max(64, int(h * 0.16))
    rois = [
        (w - corner_w, h - corner_h, w, h),
        (0, h - corner_h, corner_w, h),
        (w - corner_w, 0, w, corner_h),
        (0, 0, corner_w, corner_h),
    ]
    for x1, y1, x2, y2 in rois:
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        edges = cv2.Canny(roi, 45, 130)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, _kernel(3))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(edges, 8)
        for i in range(1, n):
            x, y, cw, ch, area = stats[i]
            if not (10 <= area <= roi.size * 0.035):
                continue
            if ch < 3 or ch > roi.shape[0] * 0.55:
                continue
            if cw > roi.shape[1] * 0.75:
                continue
            component = (labels == i).astype(np.uint8) * 255
            mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], component)

    join = max(3, int(round(min(w, h) / 900)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel(join, cv2.MORPH_ELLIPSE))
    safety = max(1, int(round(min(w, h) / 1800)))
    if safety > 1:
        mask = _dilate(mask, safety)
    return mask


def detect_candidates(image: Image.Image) -> np.ndarray:
    """Fast automatic candidate detection.

    Large uploads are detected on a reduced copy and the mask is restored to
    the original dimensions. This prevents OpenCV connected-component work
    from becoming extremely slow on phone/camera images.
    """
    pil = image.convert("RGB")
    ow, oh = pil.size
    scale = min(1.0, MAX_DETECT_DIM / max(ow, oh))
    if scale < 1.0:
        work = pil.resize(
            (max(1, int(ow * scale)), max(1, int(oh * scale))),
            Image.Resampling.LANCZOS,
        )
    else:
        work = pil
    mask = _detect_small(work)
    if mask.shape != (oh, ow):
        mask = np.asarray(
            Image.fromarray(mask, "L").resize((ow, oh), Image.Resampling.NEAREST)
        )
    return np.where(mask > 30, 255, 0).astype(np.uint8)


def overlay_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    base = np.asarray(image.convert("RGB")).copy()
    m = mask > 0
    base[m] = (base[m] * 0.35 + np.array([255, 40, 40]) * 0.65).astype(np.uint8)
    return Image.fromarray(base)
