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
    """CPU-only hybrid watermark removal engine.

    Uses two fundamentally different reconstruction methods:
    1. OpenCV Telea/Navier-Stokes at the original resolution for hard,
       high-contrast text and small logos.
    2. LaMa on a local 512px crop for semantic texture reconstruction.

    The final result is selected/blended from both rather than relying on LaMa
    alone. This is intentionally generic and is not tied to any watermark type.
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
                print(
                    f"[inpaint] downloading LaMa model (attempt {attempt}/{MODEL_RETRIES}, "
                    f"resume at {current_size / 1024 / 1024:.1f} MB) ...",
                    flush=True,
                )
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
                    raise RuntimeError("LaMa ONNX SHA-256 verification failed after download.")
                part_path.replace(self.model_path)
                return
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, RuntimeError) as exc:
                print(f"[inpaint] download interrupted: {exc}", flush=True)
                if attempt < MODEL_RETRIES:
                    time.sleep(min(2 ** (attempt - 1), 16))
                else:
                    raise RuntimeError(
                        "LaMa model download failed after retries. "
                        f"Partial file kept at {part_path} so the next run can resume."
                    ) from exc

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
    def _expand_mask(mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        radius = max(2, int(round(min(h, w) / 700.0)))
        radius = min(radius, 12)
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

        # Keep enough context for LaMa while making the masked area large in the
        # 512x512 model input. Rectangular crops preserve wide watermark detail.
        crop_w = min(w, max(192, int(mw * 2.6) + 96))
        crop_h = min(h, max(192, int(mh * 2.6) + 96))
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        left = max(0, min(cx - crop_w // 2, w - crop_w))
        top = max(0, min(cy - crop_h // 2, h - crop_h))
        right, bottom = left + crop_w, top + crop_h
        return image[top:bottom, left:right], mask[top:bottom, left:right], (left, top, right, bottom)

    @staticmethod
    def _resize_to_512(image: np.ndarray, mask: np.ndarray):
        return (
            cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA),
            cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST),
        )

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
        repaired = np.nan_to_num(repaired, nan=0.0, posinf=255.0, neginf=0.0)
        return np.clip(repaired, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _run_lama(session, model_image: np.ndarray, model_mask: np.ndarray) -> np.ndarray:
        image_input = np.transpose(
            (model_image.astype(np.float32) / 255.0)[None, ...], (0, 3, 1, 2)
        )
        mask_input = (model_mask.astype(np.float32) / 255.0)[None, None, ...]
        image_name = mask_name = None
        for item in session.get_inputs():
            shape = item.shape
            if len(shape) == 4 and shape[1] == 3:
                image_name = item.name
            elif len(shape) == 4 and shape[1] == 1:
                mask_name = item.name
        if not image_name or not mask_name:
            raise RuntimeError("LaMa ONNX inputs were not recognised; expected RGB and mask tensors.")
        output = session.run([session.get_outputs()[0].name], {image_name: image_input, mask_name: mask_input})[0]
        return InpaintEngine._normalise_output(output)

    @staticmethod
    def _opencv_inpaint(source: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Original-resolution inpainting for crisp/high-contrast watermarks."""
        # Telea is generally strong for thin text/lines; Navier-Stokes gives a
        # second independent reconstruction for flatter backgrounds.
        telea = cv2.inpaint(source, mask, 3.0, cv2.INPAINT_TELEA)
        ns = cv2.inpaint(source, mask, 3.0, cv2.INPAINT_NS)

        # Prefer Telea for narrow strokes and NS for broader solid regions.
        area = float(np.count_nonzero(mask)) / float(mask.size)
        if area < 0.035:
            return telea
        return cv2.addWeighted(telea, 0.55, ns, 0.45, 0.0)

    @staticmethod
    def _blend(original: np.ndarray, repaired: np.ndarray, mask: np.ndarray, feather: float = 1.0) -> np.ndarray:
        alpha = mask.astype(np.float32) / 255.0
        if feather > 0:
            alpha = cv2.GaussianBlur(alpha, (0, 0), feather)
        alpha = np.clip(alpha, 0.0, 1.0)[..., None]
        return np.clip(
            original.astype(np.float32) * (1.0 - alpha) + repaired.astype(np.float32) * alpha,
            0,
            255,
        ).astype(np.uint8)

    def run(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        source = np.asarray(image.convert("RGB"), dtype=np.uint8)
        user_mask = np.asarray(mask.convert("L"), dtype=np.uint8)
        user_mask = np.where(user_mask > 20, 255, 0).astype(np.uint8)
        if not np.any(user_mask):
            return image.convert("RGB")

        # Slight expansion removes anti-aliased edges that are commonly left
        # behind when the brush is drawn directly over text/logo pixels.
        work_mask = self._expand_mask(user_mask)

        # Strategy 1: original-resolution OpenCV reconstruction. This does not
        # suffer from the 512px bottleneck and is intentionally always executed.
        opencv_result = self._opencv_inpaint(source, work_mask)
        opencv_final = self._blend(source, opencv_result, work_mask, feather=0.6)

        # Strategy 2: LaMa local semantic reconstruction. If the model cannot be
        # loaded, keep the OpenCV result usable rather than failing the whole job.
        try:
            session = self._get_session()
            crop, crop_mask, box = self._prepare_crop(source, work_mask)
            left, top, right, bottom = box
            model_image, model_mask = self._resize_to_512(crop, crop_mask)
            lama_crop = self._run_lama(session, model_image, model_mask)
            lama_crop = cv2.resize(lama_crop, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_CUBIC)
            lama_full = source.copy()
            lama_full[top:bottom, left:right] = self._blend(
                crop, lama_crop, crop_mask, feather=0.8
            )

            # Build the final hybrid. The center of the user mask uses LaMa,
            # while a narrow boundary band uses the original-resolution OpenCV
            # result. This preserves LaMa texture but prevents dark/bright seams.
            center = cv2.erode(work_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1)
            boundary = cv2.subtract(work_mask, center)
            hybrid = opencv_final.copy()
            center_alpha = center.astype(np.float32) / 255.0
            center_alpha = center_alpha[..., None]
            hybrid = np.clip(
                hybrid.astype(np.float32) * (1.0 - center_alpha)
                + lama_full.astype(np.float32) * center_alpha,
                0,
                255,
            ).astype(np.uint8)
            final = hybrid

            # If LaMa barely changed the masked area, use the stronger OpenCV
            # reconstruction instead of allowing the original watermark through.
            m = center > 0
            if np.any(m):
                lama_delta = float(np.mean(np.abs(lama_full[m].astype(np.int16) - source[m].astype(np.int16))))
                cv_delta = float(np.mean(np.abs(opencv_final[m].astype(np.int16) - source[m].astype(np.int16))))
                if lama_delta < max(1.0, cv_delta * 0.25):
                    final = opencv_final
        except Exception as exc:
            print(f"[inpaint] LaMa unavailable/failed, using OpenCV result: {exc}", flush=True)
            final = opencv_final

        changed = float(np.mean(np.abs(final.astype(np.int16) - source.astype(np.int16))))
        if changed < 0.02:
            raise RuntimeError(
                "The supplied mask produced almost no change. Please paint fully over the watermark."
            )
        return Image.fromarray(final, mode="RGB")
