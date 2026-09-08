from __future__ import annotations

import hashlib
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "https://huggingface.co/sapienkit/LaMa-ONNX/resolve/main/lama_fp32.onnx",
)
MODEL_SHA256 = os.environ.get(
    "LAMA_MODEL_SHA256",
    "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6",
)
MODEL_PATH = Path(os.environ.get("LAMA_MODEL_PATH", "/app/models/lama_fp32.onnx"))
MODEL_RETRIES = max(1, int(os.environ.get("LAMA_MODEL_RETRIES", "5")))
CHUNK_SIZE = 1024 * 1024


class InpaintEngine:
    """CPU-only watermark removal engine.

    Disconnected watermark regions are repaired independently. This is important
    for images containing several text/logo watermarks: feeding one giant crop
    containing all regions to LaMa wastes its 512x512 context and often leaves
    one or more watermarks behind.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = Path(model_path or MODEL_PATH)
        self._session: Optional[ort.InferenceSession] = None

    def _verify_model(self, path: Path) -> bool:
        if not path.exists() or path.stat().st_size == 0:
            return False
        if not MODEL_SHA256:
            return True
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
                digest.update(chunk)
        return digest.hexdigest().lower() == MODEL_SHA256.lower()

    def _download_model(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        part_path = self.model_path.with_suffix(".download")
        if part_path.exists() and self._verify_model(part_path):
            part_path.replace(self.model_path)
            return
        for attempt in range(1, MODEL_RETRIES + 1):
            try:
                current_size = part_path.stat().st_size if part_path.exists() else 0
                headers = {"Range": f"bytes={current_size}-"} if current_size else {}
                request = urllib.request.Request(MODEL_URL, headers=headers)
                print(f"[inpaint] downloading LaMa (attempt {attempt}/{MODEL_RETRIES}, resume={current_size})", flush=True)
                with urllib.request.urlopen(request, timeout=60) as response:
                    status = getattr(response, "status", 200)
                    if current_size and status != 206:
                        current_size = 0
                        part_path.unlink(missing_ok=True)
                        with urllib.request.urlopen(urllib.request.Request(MODEL_URL), timeout=60) as full:
                            self._stream_to_file(full, part_path, False)
                    else:
                        self._stream_to_file(response, part_path, current_size > 0)
                if not self._verify_model(part_path):
                    raise RuntimeError("LaMa ONNX SHA-256 verification failed")
                part_path.replace(self.model_path)
                return
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, RuntimeError) as exc:
                print(f"[inpaint] download interrupted: {exc}", flush=True)
                if attempt < MODEL_RETRIES:
                    time.sleep(min(2 ** (attempt - 1), 16))
                else:
                    raise RuntimeError("LaMa model download failed after retries") from exc

    @staticmethod
    def _stream_to_file(response, path: Path, append: bool) -> None:
        mode = "ab" if append else "wb"
        downloaded = path.stat().st_size if append and path.exists() else 0
        last_report = downloaded
        with open(path, mode) as f:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded - last_report >= 10 * 1024 * 1024:
                    print(f"[inpaint] downloaded {downloaded / 1024 / 1024:.1f} MB", flush=True)
                    last_report = downloaded

    def _get_session(self) -> ort.InferenceSession:
        if self._session is None:
            if not self.model_path.exists() or not self._verify_model(self.model_path):
                self.model_path.unlink(missing_ok=True)
                self._download_model()
            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, int(os.environ.get("OMP_NUM_THREADS", "4")))
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                str(self.model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        return self._session

    @staticmethod
    def _expand_mask(mask: np.ndarray, strength: int = 1) -> np.ndarray:
        h, w = mask.shape
        base = max(2, int(round(min(h, w) / 420.0)))
        radius = min(24, base * max(1, strength))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        return cv2.dilate(mask, kernel, iterations=1)

    @staticmethod
    def _prepare_crop(image: np.ndarray, mask: np.ndarray):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            raise ValueError("Mask is empty")
        h, w = mask.shape
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        mw, mh = x2 - x1, y2 - y1
        crop_w = min(w, max(256, int(mw * 3.2) + 160))
        crop_h = min(h, max(256, int(mh * 3.2) + 160))
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        left = max(0, min(cx - crop_w // 2, w - crop_w))
        top = max(0, min(cy - crop_h // 2, h - crop_h))
        right, bottom = left + crop_w, top + crop_h
        return image[top:bottom, left:right], mask[top:bottom, left:right], (left, top, right, bottom)

    @staticmethod
    def _normalise_output(output: np.ndarray) -> np.ndarray:
        repaired = np.asarray(output[0])
        if repaired.ndim != 3:
            raise RuntimeError(f"Unexpected LaMa output shape: {repaired.shape}")
        if repaired.shape[0] == 3:
            repaired = np.transpose(repaired, (1, 2, 0))
        elif repaired.shape[-1] != 3:
            raise RuntimeError(f"Unexpected LaMa output shape: {repaired.shape}")
        repaired = repaired.astype(np.float32)
        if float(np.nanmax(repaired)) <= 1.5:
            repaired *= 255.0
        return np.clip(np.nan_to_num(repaired, nan=0.0, posinf=255.0, neginf=0.0), 0, 255).astype(np.uint8)

    @staticmethod
    def _run_lama(session, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        image_input = np.transpose((image.astype(np.float32) / 255.0)[None, ...], (0, 3, 1, 2))
        mask_input = (mask.astype(np.float32) / 255.0)[None, None, ...]
        image_name = mask_name = None
        for item in session.get_inputs():
            shape = item.shape
            if len(shape) == 4 and shape[1] == 3:
                image_name = item.name
            elif len(shape) == 4 and shape[1] == 1:
                mask_name = item.name
        if not image_name or not mask_name:
            raise RuntimeError("LaMa ONNX inputs were not recognised")
        output = session.run([session.get_outputs()[0].name], {image_name: image_input, mask_name: mask_input})[0]
        return InpaintEngine._normalise_output(output)

    @staticmethod
    def _opencv_pass(source: np.ndarray, mask: np.ndarray, radius: float) -> np.ndarray:
        telea = cv2.inpaint(source, mask, radius, cv2.INPAINT_TELEA)
        ns = cv2.inpaint(source, mask, radius, cv2.INPAINT_NS)
        return cv2.addWeighted(telea, 0.65, ns, 0.35, 0.0)

    @staticmethod
    def _blend(original: np.ndarray, repaired: np.ndarray, mask: np.ndarray, sigma: float = 0.0) -> np.ndarray:
        alpha = mask.astype(np.float32) / 255.0
        if sigma > 0:
            alpha = cv2.GaussianBlur(alpha, (0, 0), sigma)
        alpha = np.clip(alpha, 0.0, 1.0)[..., None]
        return np.clip(
            original.astype(np.float32) * (1.0 - alpha) + repaired.astype(np.float32) * alpha,
            0,
            255,
        ).astype(np.uint8)

    @staticmethod
    def _masked_delta(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
        m = mask > 0
        if not np.any(m):
            return 0.0
        return float(np.mean(np.abs(a[m].astype(np.int16) - b[m].astype(np.int16))))

    @staticmethod
    def _component_masks(mask: np.ndarray) -> list[np.ndarray]:
        """Split a mask into useful watermark regions.

        Tiny glyph fragments are kept. Nearby fragments are grouped first so a
        multi-character text watermark is repaired as one region, while distant
        logos/text watermarks are sent to LaMa separately.
        """
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        h, w = binary.shape
        # Connect characters that belong to the same text line, but don't bridge
        # distant watermarks across the image.
        join_x = max(5, min(31, int(round(w / 180))))
        join_y = max(3, min(17, int(round(h / 420))))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (join_x | 1, join_y | 1))
        grouped = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, 8)
        components: list[np.ndarray] = []
        min_area = max(12, int(h * w * 0.000001))
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            comp = np.where(labels == i, 255, 0).astype(np.uint8)
            # Recover the original mask inside the grouped component instead of
            # repairing the artificial closing pixels themselves.
            comp = cv2.bitwise_and(binary, comp)
            if np.count_nonzero(comp) >= min_area:
                components.append(comp)

        if not components and np.count_nonzero(binary):
            components = [binary]

        # Very close components can still be joined by bounding-box proximity.
        # This helps text with unusually large gaps between words.
        merged = True
        while merged and len(components) > 1:
            merged = False
            boxes = []
            for comp in components:
                ys, xs = np.where(comp > 0)
                boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
            for i in range(len(components)):
                if merged:
                    break
                for j in range(i + 1, len(components)):
                    ax1, ay1, ax2, ay2 = boxes[i]
                    bx1, by1, bx2, by2 = boxes[j]
                    gap_x = max(0, max(ax1, bx1) - min(ax2, bx2) - 1)
                    gap_y = max(0, max(ay1, by1) - min(ay2, by2) - 1)
                    near = gap_x <= max(12, w // 80) and gap_y <= max(12, h // 80)
                    if near:
                        components[i] = cv2.bitwise_or(components[i], components[j])
                        components.pop(j)
                        merged = True
                        break
        return components

    def _repair_region(self, source: np.ndarray, user_mask: np.ndarray, session) -> np.ndarray:
        """Repair one disconnected watermark region at a useful local scale."""
        repair_mask = self._expand_mask(user_mask, 2)
        edge_mask = self._expand_mask(user_mask, 3)

        cv_a = self._opencv_pass(source, repair_mask, 5.0)
        cv_b = self._opencv_pass(source, edge_mask, 7.0)
        opencv_final = self._blend(cv_a, cv_b, repair_mask, sigma=0.5)
        final = opencv_final

        try:
            crop, crop_mask, (left, top, right, bottom) = self._prepare_crop(source, edge_mask)
            ch, cw = crop.shape[:2]
            model_image = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)
            model_mask = cv2.resize(crop_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

            lama_1 = self._run_lama(session, model_image, model_mask)
            lama_1 = cv2.resize(lama_1, (cw, ch), interpolation=cv2.INTER_CUBIC)

            lama_2 = self._run_lama(
                session,
                cv2.resize(lama_1, (512, 512), interpolation=cv2.INTER_AREA),
                model_mask,
            )
            lama_2 = cv2.resize(lama_2, (cw, ch), interpolation=cv2.INTER_CUBIC)

            lama_full = source.copy()
            lama_full[top:bottom, left:right] = lama_2
            lama_final = self._blend(source, lama_full, edge_mask, sigma=0.4)

            interior = cv2.erode(
                repair_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                1,
            )
            interior = cv2.bitwise_and(interior, user_mask)
            alpha = (interior.astype(np.float32) / 255.0)[..., None]
            final = np.clip(
                opencv_final.astype(np.float32) * (1.0 - alpha)
                + lama_final.astype(np.float32) * alpha,
                0,
                255,
            ).astype(np.uint8)

            cv_delta = self._masked_delta(opencv_final, source, user_mask)
            lama_delta = self._masked_delta(lama_final, source, user_mask)
            print(f"[inpaint] region delta: opencv={cv_delta:.2f}, lama={lama_delta:.2f}", flush=True)
            if lama_delta < max(2.0, cv_delta * 0.20):
                final = opencv_final
        except Exception as exc:
            print(f"[inpaint] region LaMa failed; OpenCV fallback: {exc}", flush=True)

        boundary = cv2.subtract(
            edge_mask,
            cv2.erode(edge_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), 1),
        )
        boundary_repair = self._opencv_pass(final, boundary, 4.0)
        return self._blend(final, boundary_repair, boundary, sigma=0.7)

    def run(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        source = np.asarray(image.convert("RGB"), dtype=np.uint8)
        user_mask = np.asarray(mask.convert("L"), dtype=np.uint8)
        user_mask = np.where(user_mask > 20, 255, 0).astype(np.uint8)
        pixels = int(np.count_nonzero(user_mask))
        if pixels == 0:
            raise ValueError("Mask is empty")

        print(f"[inpaint] USER MASK = {pixels:,} px / {user_mask.size:,} ({pixels / user_mask.size:.2%})", flush=True)

        components = self._component_masks(user_mask)
        print(f"[inpaint] detected repair regions = {len(components)}", flush=True)

        final = source.copy()
        session = None
        try:
            session = self._get_session()
        except Exception as exc:
            print(f"[inpaint] LaMa unavailable; using native OpenCV: {exc}", flush=True)

        total_changed = 0.0
        for index, component in enumerate(components, 1):
            region_pixels = int(np.count_nonzero(component))
            print(f"[inpaint] repairing region {index}/{len(components)}: {region_pixels:,} px", flush=True)
            try:
                if session is not None:
                    repaired = self._repair_region(source, component, session)
                else:
                    repair_mask = self._expand_mask(component, 2)
                    repaired = self._opencv_pass(source, repair_mask, 5.0)
                # Apply only this component's expanded reconstruction. Keeping
                # separate regions prevents one watermark from contaminating the
                # context used to reconstruct another.
                apply_mask = self._expand_mask(component, 3)
                final = self._blend(final, repaired, apply_mask, sigma=0.35)
                delta = self._masked_delta(repaired, source, component)
                total_changed += delta
                print(f"[inpaint] region {index} masked delta={delta:.2f}", flush=True)
            except Exception as exc:
                print(f"[inpaint] region {index} failed: {exc}", flush=True)

        masked_changed = self._masked_delta(final, source, user_mask)
        changed = float(np.mean(np.abs(final.astype(np.int16) - source.astype(np.int16))))
        print(f"[inpaint] final delta={changed:.3f}, masked delta={masked_changed:.3f}, regions={len(components)}", flush=True)
        if masked_changed < 0.5:
            raise RuntimeError("修复区域几乎没有发生变化，请确认 Mask 覆盖了完整水印区域。")

        return Image.fromarray(final, mode="RGB")
