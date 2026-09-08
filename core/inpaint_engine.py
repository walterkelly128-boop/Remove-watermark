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
    """CPU-only LaMa inpainting engine optimized for painted watermark masks."""

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
    def _prepare_crop(image: np.ndarray, mask: np.ndarray):
        """Build a model crop that keeps the watermark large enough in 512x512."""
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            raise ValueError("Mask is empty")

        h, w = mask.shape
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        target = max(x2 - x1, y2 - y1, 24)

        # The previous 4x context made large watermarks occupy too few pixels
        # after resizing to 512, producing only a weak correction. Around 2.4x
        # gives LaMa much more detail while still leaving useful background context.
        side = max(int(target * 2.4), 192)
        side = min(side, max(h, w))

        # Avoid an unnecessarily huge square when the mask is a wide/short logo.
        crop_w = min(w, max(side, x2 - x1 + 96))
        crop_h = min(h, max(side, y2 - y1 + 96))
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        left = max(0, min(cx - crop_w // 2, w - crop_w))
        top = max(0, min(cy - crop_h // 2, h - crop_h))
        right = min(w, left + crop_w)
        bottom = min(h, top + crop_h)
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
    def _expand_mask(mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        # Stronger expansion is important for high-contrast/anti-aliased text.
        radius = max(3, int(round(min(h, w) / 380.0)))
        radius = min(radius, 16)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
        )
        return cv2.dilate(mask, kernel, iterations=1)

    @staticmethod
    def _run_lama(session, model_image: np.ndarray, model_mask: np.ndarray) -> np.ndarray:
        image_input = np.transpose(
            (model_image.astype(np.float32) / 255.0)[None, ...],
            (0, 3, 1, 2),
        )
        mask_input = (model_mask.astype(np.float32) / 255.0)[None, None, ...]

        inputs = session.get_inputs()
        image_name = None
        mask_name = None
        for item in inputs:
            shape = item.shape
            if len(shape) == 4 and shape[1] == 3:
                image_name = item.name
            elif len(shape) == 4 and shape[1] == 1:
                mask_name = item.name
        if not image_name or not mask_name:
            raise RuntimeError("LaMa ONNX inputs were not recognised; expected RGB and mask tensors.")

        output = session.run(
            [session.get_outputs()[0].name],
            {image_name: image_input, mask_name: mask_input},
        )[0]
        return InpaintEngine._normalise_output(output)

    def run(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        source = np.asarray(image.convert("RGB"), dtype=np.uint8)
        user_mask = np.asarray(mask.convert("L"), dtype=np.uint8)
        user_mask = np.where(user_mask > 30, 255, 0).astype(np.uint8)

        if not np.any(user_mask):
            return image.convert("RGB")

        session = self._get_session()
        model_mask_full = self._expand_mask(user_mask)
        crop, _, box = self._prepare_crop(source, model_mask_full)
        left, top, right, bottom = box
        crop_model_mask = model_mask_full[top:bottom, left:right]
        model_image, model_mask = self._resize_to_512(crop, crop_model_mask)

        # First pass: primary reconstruction.
        repaired = self._run_lama(session, model_image, model_mask)

        # A second light pass is useful for high-contrast logos/text. It receives
        # the first reconstruction instead of the original watermark pixels,
        # allowing LaMa to clean residual outlines without changing the rest of
        # the image. Keep it conditional so CPU processing remains reasonable.
        mask_fraction = float(np.count_nonzero(model_mask)) / float(model_mask.size)
        if mask_fraction < 0.30:
            second_mask = cv2.dilate(
                model_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )
            repaired = self._run_lama(session, repaired, second_mask)

        repaired = cv2.resize(
            repaired,
            (crop.shape[1], crop.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

        # Use the expanded mask for full-strength replacement. Only feather a
        # narrow outer boundary; the center must not retain the original
        # high-contrast watermark pixels.
        composite_mask = model_mask_full[top:bottom, left:right]
        alpha = composite_mask.astype(np.float32) / 255.0
        feather = max(0.8, min(2.0, min(crop.shape[:2]) / 900.0))
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather)
        alpha = np.clip(alpha, 0.0, 1.0)[..., None]

        result = source.astype(np.float32)
        original_crop = source[top:bottom, left:right].astype(np.float32)
        result[top:bottom, left:right] = (
            original_crop * (1.0 - alpha) + repaired.astype(np.float32) * alpha
        )

        final = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="RGB")
        final_arr = np.asarray(final)
        changed = np.mean(np.abs(final_arr.astype(np.int16) - source.astype(np.int16)))
        if changed < 0.01:
            raise RuntimeError(
                "LaMa returned no visible change for the supplied Mask. "
                "Please enlarge the Mask so it fully covers the watermark."
            )
        return final
