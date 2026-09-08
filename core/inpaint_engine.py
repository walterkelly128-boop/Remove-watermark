from __future__ import annotations

import hashlib
import os
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
MODEL_PATH = Path(
    os.environ.get("LAMA_MODEL_PATH", "/app/models/lama_fp32.onnx")
)


class InpaintEngine:
    """CPU-only LaMa ONNX inpainting engine.

    The model accepts a fixed 512x512 image and mask. For normal-sized images,
    we crop a context window around the requested mask, run LaMa at 512x512,
    then paste only the repaired mask area back into the original image. This
    avoids shrinking an entire high-resolution image to 512 pixels.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = Path(model_path or MODEL_PATH)
        self._session: Optional[ort.InferenceSession] = None

    def _download_model(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.model_path.with_suffix(".download")
        print(f"[inpaint] downloading CPU model to {self.model_path} ...", flush=True)
        urllib.request.urlretrieve(MODEL_URL, tmp_path)

        digest = hashlib.sha256()
        with open(tmp_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if MODEL_SHA256 and actual.lower() != MODEL_SHA256.lower():
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                "LaMa ONNX model SHA-256 verification failed. "
                f"Expected {MODEL_SHA256}, got {actual}."
            )
        tmp_path.replace(self.model_path)

    def _get_session(self) -> ort.InferenceSession:
        if self._session is None:
            if not self.model_path.exists():
                self._download_model()

            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, int(os.environ.get("OMP_NUM_THREADS", "4")))
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # Explicitly select CPUExecutionProvider. No CUDA, TensorRT or
            # NVIDIA runtime is required by this project.
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

        # Add context around the target. Keep the crop reasonably compact so
        # small watermarks do not force the whole image through a 512x512 model.
        target_w = max(x2 - x1, 32)
        target_h = max(y2 - y1, 32)
        context = max(32, int(max(target_w, target_h) * 1.8))
        side = max(target_w, target_h, context)
        side = min(side, max(h, w))

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        left = max(0, min(cx - side // 2, w - side))
        top = max(0, min(cy - side // 2, h - side))
        right = min(w, left + side)
        bottom = min(h, top + side)

        crop = image[top:bottom, left:right]
        crop_mask = mask[top:bottom, left:right]
        return crop, crop_mask, (left, top, right, bottom)

    @staticmethod
    def _resize_to_512(image: np.ndarray, mask: np.ndarray):
        resized_image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        return resized_image, resized_mask

    def run(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        session = self._get_session()

        source = np.asarray(image.convert("RGB"), dtype=np.uint8)
        source_mask = np.asarray(mask.convert("L"), dtype=np.uint8)
        source_mask = np.where(source_mask > 30, 255, 0).astype(np.uint8)

        if not np.any(source_mask):
            return image.convert("RGB")

        crop, crop_mask, box = self._prepare_crop(source, source_mask)
        model_image, model_mask = self._resize_to_512(crop, crop_mask)

        # Model contract: RGB float32 in [0,1], mask float32 where 1 means erase.
        image_input = (model_image.astype(np.float32) / 255.0)[None, ...]
        image_input = np.transpose(image_input, (0, 3, 1, 2))
        mask_input = (model_mask.astype(np.float32) / 255.0)[None, None, ...]

        input_names = [x.name for x in session.get_inputs()]
        output_name = session.get_outputs()[0].name
        inputs = {input_names[0]: image_input, input_names[1]: mask_input}
        output = session.run([output_name], inputs)[0]
        repaired = np.asarray(output[0])

        if repaired.ndim == 3 and repaired.shape[0] == 3:
            repaired = np.transpose(repaired, (1, 2, 0))
        repaired = np.clip(repaired, 0, 255).astype(np.uint8)
        repaired = cv2.resize(
            repaired,
            (crop.shape[1], crop.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

        # Feather the mask edge to avoid a visible rectangular seam around the
        # repaired area while preserving untouched pixels outside the mask.
        feather = max(3, int(round(min(crop.shape[:2]) / 180)))
        alpha = cv2.GaussianBlur(crop_mask.astype(np.float32) / 255.0, (0, 0), feather)
        alpha = np.clip(alpha, 0.0, 1.0)[..., None]

        left, top, right, bottom = box
        result = source.copy().astype(np.float32)
        original_crop = source[top:bottom, left:right].astype(np.float32)
        blended = original_crop * (1.0 - alpha) + repaired.astype(np.float32) * alpha
        result[top:bottom, left:right] = blended

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="RGB")
