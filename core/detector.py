from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _dilate(mask: np.ndarray, k: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel, iterations=1)


def detect_candidates(image: Image.Image) -> np.ndarray:
    """Return a conservative binary mask for likely overlay/watermark regions.

    The detector focuses on text-like connected components and corner overlays.
    It deliberately avoids treating every piece of text in the image as a target.
    """
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # OCR-like text candidates using adaptive thresholding.
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9
    )
    bw = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    )

    mask = np.zeros((h, w), np.uint8)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Prefer small text-like components, especially near image edges/corners.
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < 12 or area > (w * h * 0.08):
            continue
        ratio = cw / max(ch, 1)
        if ratio < 0.35 or ratio > 25:
            continue
        near_edge = x < w * 0.18 or y < h * 0.18 or x + cw > w * 0.82 or y + ch > h * 0.82
        if near_edge and ch < h * 0.10 and cw < w * 0.65:
            cv2.rectangle(mask, (x, y), (x + cw, y + ch), 255, -1)

    # A small corner ROI catches compact Gemini-style corner marks better than OCR alone.
    corner_w = max(48, int(w * 0.18))
    corner_h = max(48, int(h * 0.12))
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
        edges = cv2.Canny(roi, 80, 180)
        # Only mark compact edge clusters; do not mask the whole corner.
        n, labels, stats, _ = cv2.connectedComponentsWithStats(edges, 8)
        for i in range(1, n):
            x, y, cw, ch, area = stats[i]
            if 8 <= area <= roi.size * 0.08 and 2 <= ch <= roi.shape[0] * 0.8:
                cv2.rectangle(mask, (x1 + x, y1 + y), (x1 + x + cw, y1 + y + ch), 255, -1)

    mask = _dilate(mask, max(3, int(round(min(w, h) / 700))))
    return mask


def overlay_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    base = np.asarray(image.convert("RGB")).copy()
    m = mask > 0
    # Red preview without changing the underlying image data.
    base[m] = (base[m] * 0.35 + np.array([255, 40, 40]) * 0.65).astype(np.uint8)
    return Image.fromarray(base)
