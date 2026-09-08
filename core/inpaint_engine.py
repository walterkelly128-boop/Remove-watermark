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
    """CPU-only LaMa engine with tight final compositing."""

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
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            raise ValueError("Mask is empty")
        h, w = mask.shape
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        target = max(x2 - x1, y2 - y1, 24)
        # More context for texture/structure, but the final composite will only
        # touch the original tight mask.
        side = min(max(int(target * 3.2), 96), max(h, w))
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        left = max(0, min(cx - side // 2, w - side))
        top = max(0, min(cy - side // 2, h - side))
        right, bottom = min(w, left + side), min(h, top + side)
        return image[top:bottom, left:right], mask[top:bottom, left:right], (left, top, right, bottom)

    @staticmethod
    def _resize_to_512(image: np.ndarray, mask: np.ndarray):
        return (
            cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA),
            cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST),
        )

    def run(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        session = self._get_session()
        source = np.asarray(image.convert("RGB"), dtype=np.uint8)
        tight_mask = np.asarray(mask.convert("L"), dtype=np.uint8)
        tight_mask = np.where(tight_mask > 30, 255, 0).astype(np.uint8)
        if not np.any(tight_mask):
            return image.convert("RGB")

        # Give LaMa a modestly expanded mask so it can remove edge remnants,
        # but NEVER use that expanded mask for final compositing.
        h, w = tight_mask.shape
        expansion = max(2, int(round(min(h, w) / 900)))
        model_mask_full = cv2.dilate(tight_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expansion * 2 + 1, expansion * 2 + 1)))

        crop, _, box = self._prepare_crop(source, model_mask_full)
        left, top, right, bottom = box
        crop_model_mask = model_mask_full[top:bottom, left:right]
        model_image, model_mask = self._resize_to_512(crop, crop_model_mask)

        image_input = np.transpose((model_image.astype(np.float32) / 255.0)[None, ...], (0, 3, 1, 2))
        mask_input = (model_mask.astype(np.float32) / 255.0)[None, None, ...]
        input_names = [x.name for x in session.get_inputs()]
        output_name = session.get_outputs()[0].name
        output = session.run([output_name], {input_names[0]: image_input, input_names[1]: mask_input})[0]
        repaired = np.asarray(output[0])
        if repaired.ndim == 3 and repaired.shape[0] == 3:
            repaired = np.transpose(repaired, (1, 2, 0))
        repaired = np.clip(repaired, 0, 255).astype(np.uint8)
        repaired = cv2.resize(repaired, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_CUBIC)

        # Crucial: composite only where the ORIGINAL mask says the watermark is.
        # A 1px-ish feather prevents a hard edge without modifying surrounding pixels.
        tight_crop = tight_mask[top:bottom, left:right]
        feather_px = max(0.7, min(2.2, min(crop.shape[:2]) / 900.0))
        alpha = cv2.GaussianBlur(tight_crop.astype(np.float32) / 255.0, (0, 0), feather_px)
        alpha = np.clip(alpha, 0.0, 1.0)[..., None]

        result = source.copy().astype(np.float32)
        original_crop = source[top:bottom, left:right].astype(np.float32)
        result[top:bottom, left:right] = original_crop * (1.0 - alpha) + repaired.astype(np.float32) * alpha
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="RGB")
