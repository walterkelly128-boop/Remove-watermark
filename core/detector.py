from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _kernel(size: int, shape=cv2.MORPH_ELLIPSE):
    size = max(3, int(size))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(shape, (size, size))


def _dilate(mask: np.ndarray, k: int) -> np.ndarray:
    return cv2.dilate(mask, _kernel(k), iterations=1)


def _tight_component_mask(binary: np.ndarray, h: int, w: int) -> np.ndarray:
    """Keep compact text-like components, but preserve their actual shapes."""
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
        component = (labels == i).astype(np.uint8) * 255
        out = np.maximum(out, component)
    return out


def detect_candidates(image: Image.Image) -> np.ndarray:
    """Detect likely Gemini-style corner watermark pixels with a tight mask.

    The important change is that the detector no longer fills large bounding
    rectangles around text. The inpainting model gets an expanded private mask
    for context, while the returned mask remains close to the actual watermark.
    """
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    mask = np.zeros((h, w), np.uint8)

    # Multi-scale local contrast catches pale/white watermark strokes that a
    # single adaptive threshold can miss.
    for block in (21, 35, 51):
        bw = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block, 7,
        )
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, _kernel(2, cv2.MORPH_RECT))
        mask = np.maximum(mask, _tight_component_mask(bw, h, w))

    # Corner-specific edge detection. Keep the edge pixels themselves rather
    # than replacing them with coarse rectangles.
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
        # Close small gaps in glyphs, then retain only compact components.
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

    # Join nearby glyph strokes without creating a large filled rectangle.
    join = max(3, int(round(min(w, h) / 900)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel(join, cv2.MORPH_ELLIPSE))

    # Only a very small safety dilation is returned. This is deliberately much
    # smaller than the old bounding-box approach to prevent halo artifacts.
    safety = max(1, int(round(min(w, h) / 1800)))
    if safety > 1:
        mask = _dilate(mask, safety)

    return mask


def overlay_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    base = np.asarray(image.convert("RGB")).copy()
    m = mask > 0
    base[m] = (base[m] * 0.35 + np.array([255, 40, 40]) * 0.65).astype(np.uint8)
    return Image.fromarray(base)
