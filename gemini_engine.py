from __future__ import annotations

import base64
import io
import os
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter
from dotenv import load_dotenv
from google import genai

# Load .env automatically when running locally. Docker Compose also injects
# GEMINI_API_KEY from .env into the container environment.
load_dotenv()

DEFAULT_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image")


class GeminiInpaintEngine:
    """Gemini image-edit based local-region restoration."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
        self.model = model or os.environ.get("GEMINI_IMAGE_MODEL", DEFAULT_MODEL)
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if not self.api_key:
            raise RuntimeError("未配置 GEMINI_API_KEY。请在项目根目录 .env 中设置 Gemini API Key。")
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    @staticmethod
    def _png_bytes(image: Image.Image) -> bytes:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    @staticmethod
    def _make_mask_reference(image: Image.Image, mask: np.ndarray) -> Image.Image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        m = np.asarray(mask) > 30
        if m.shape != rgb.shape[:2]:
            m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize(image.size, Image.Resampling.NEAREST)) > 30
        rgb[m] = (255, 40, 40)
        return Image.fromarray(rgb, "RGB")

    @staticmethod
    def _crop_region(image: Image.Image, mask: np.ndarray, padding: float = 2.5):
        m = np.asarray(mask) > 30
        ys, xs = np.where(m)
        if len(xs) == 0:
            raise ValueError("Mask 为空")
        w, h = image.size
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        mw, mh = x2 - x1, y2 - y1
        cw = min(w, max(256, int(mw * padding) + 120))
        ch = min(h, max(256, int(mh * padding) + 120))
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        left = max(0, min(cx - cw // 2, w - cw))
        top = max(0, min(cy - ch // 2, h - ch))
        right, bottom = left + cw, top + ch
        return image.crop((left, top, right, bottom)), m[top:bottom, left:right], (left, top, right, bottom)

    def repair(self, image: Image.Image, mask: np.ndarray, use_crop: bool = True) -> Image.Image:
        image = image.convert("RGB")
        mask = np.where(np.asarray(mask) > 30, 255, 0).astype(np.uint8)
        if not np.any(mask):
            raise ValueError("Mask 为空")

        expanded = Image.fromarray(mask, "L").filter(ImageFilter.MaxFilter(9))
        mask = np.asarray(expanded, dtype=np.uint8)

        if use_crop:
            work_image, work_mask, box = self._crop_region(image, mask)
        else:
            work_image, work_mask, box = image, mask, (0, 0, image.width, image.height)

        mask_reference = self._make_mask_reference(work_image, work_mask)
        prompt = """You are performing precise image restoration on an image the user is authorized to edit.

The FIRST image is the source image. The SECOND image is a mask reference: pixels shown in red identify the ONLY region that must be reconstructed.

Remove the watermark/text/logo/overlay inside the red region by naturally reconstructing the content that should be behind it.

STRICT REQUIREMENTS:
- Change ONLY the red/masked region.
- Do NOT alter anything outside the red region.
- Preserve the original person's identity, face, hair, body, clothing, objects, geometry, perspective, lighting, shadows, colors, texture, sharpness and composition.
- Do not blur the repaired area.
- Do not crop or resize the composition.
- Do not add new objects, text, logos or decorations.
- The repaired area must seamlessly match the surrounding pixels.
- If the masked area overlaps a person's face or an important object, reconstruct that existing detail rather than inventing a different person/object.

Return the restored image only."""

        client = self._get_client()
        interaction = client.interactions.create(
            model=self.model,
            input=[
                {"type": "image", "data": base64.b64encode(self._png_bytes(work_image)).decode("utf-8"), "mime_type": "image/png"},
                {"type": "image", "data": base64.b64encode(self._png_bytes(mask_reference)).decode("utf-8"), "mime_type": "image/png"},
                {"type": "text", "text": prompt},
            ],
        )

        output_data = None
        for step in interaction.steps:
            if getattr(step, "type", None) != "model_output":
                continue
            for block in getattr(step, "content", []) or []:
                if getattr(block, "type", None) == "image" and getattr(block, "data", None):
                    output_data = block.data
        if not output_data:
            raise RuntimeError("Gemini 没有返回图片结果")

        repaired = Image.open(io.BytesIO(base64.b64decode(output_data))).convert("RGB")
        if repaired.size != work_image.size:
            repaired = repaired.resize(work_image.size, Image.Resampling.LANCZOS)

        if use_crop:
            result = image.copy()
            alpha = Image.fromarray(mask, "L").filter(ImageFilter.GaussianBlur(2.0))
            result.paste(repaired, box, alpha)
            return result

        alpha = Image.fromarray(mask, "L").filter(ImageFilter.GaussianBlur(2.0))
        return Image.composite(repaired, image, alpha)
