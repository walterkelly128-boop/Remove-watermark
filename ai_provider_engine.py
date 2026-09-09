from __future__ import annotations

import base64
import io
import os
from typing import Optional

import numpy as np
import requests
from PIL import Image, ImageFilter
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE_URL = os.environ.get("RC_API_BASE_URL", "https://api.rcouyi.com/v1").rstrip("/")
DEFAULT_GEMINI_MODEL = os.environ.get("RC_GEMINI_MODEL", "gemini-3.1-flash-image")
DEFAULT_OPENAI_MODEL = os.environ.get("RC_OPENAI_MODEL", "gpt-image-1")


class RCImageEngine:
    """Image editing through an OpenAI-compatible third-party API endpoint."""

    def __init__(self, provider: str, api_key: Optional[str] = None, model: Optional[str] = None):
        self.provider = provider.lower().strip()
        self.api_key = api_key or os.environ.get("RC_API_KEY", "").strip()
        self.base_url = os.environ.get("RC_API_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        default = DEFAULT_GEMINI_MODEL if self.provider == "gemini" else DEFAULT_OPENAI_MODEL
        self.model = model or os.environ.get(
            "RC_GEMINI_MODEL" if self.provider == "gemini" else "RC_OPENAI_MODEL", default
        )
        self.timeout = int(os.environ.get("RC_API_TIMEOUT", "180"))

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    @staticmethod
    def _png_bytes(image: Image.Image) -> bytes:
        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    @classmethod
    def _data_url(cls, image: Image.Image) -> str:
        return "data:image/png;base64," + base64.b64encode(cls._png_bytes(image)).decode("ascii")

    @staticmethod
    def _mask_reference(image: Image.Image, mask: np.ndarray) -> Image.Image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        rgb[np.asarray(mask) > 30] = (255, 40, 40)
        return Image.fromarray(rgb, "RGB")

    @staticmethod
    def _crop_region(image: Image.Image, mask: np.ndarray, padding: float = 2.8):
        m = np.asarray(mask) > 30
        ys, xs = np.where(m)
        if len(xs) == 0:
            raise ValueError("Mask 为空")
        w, h = image.size
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        mw, mh = x2 - x1, y2 - y1
        cw = min(w, max(256, int(mw * padding) + 160))
        ch = min(h, max(256, int(mh * padding) + 160))
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        left = max(0, min(cx - cw // 2, w - cw))
        top = max(0, min(cy - ch // 2, h - ch))
        right, bottom = left + cw, top + ch
        return image.crop((left, top, right, bottom)), m[top:bottom, left:right], (left, top, right, bottom)

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    @staticmethod
    def _prompt() -> str:
        return (
            "Generate exactly one edited image from the provided source image. "
            "Perform precise image restoration on an image the user is authorized to edit. "
            "The transparent area of the mask identifies the ONLY area that must be reconstructed. "
            "Remove the watermark, text, logo or overlay inside that masked region by naturally "
            "reconstructing the content behind it. Change only the masked area. Preserve identity, "
            "faces, hair, objects, geometry, perspective, lighting, shadows, colors, texture, "
            "sharpness and composition outside the mask. Do not blur the repaired area. "
            "Do not add objects, text or logos. Return the restored image only."
        )

    @staticmethod
    def _extract_image_from_response(data: dict) -> Optional[Image.Image]:
        for item in data.get("data", []) or []:
            b64 = item.get("b64_json") or item.get("base64")
            if b64:
                return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            url = item.get("url")
            if url:
                r = requests.get(url, timeout=120)
                r.raise_for_status()
                return Image.open(io.BytesIO(r.content)).convert("RGB")

        for choice in data.get("choices", []) or []:
            msg = choice.get("message", {}) or {}
            for item in msg.get("images", []) or []:
                value = item if isinstance(item, str) else item.get("image_url", item.get("data", ""))
                if isinstance(value, dict):
                    value = value.get("url", value.get("data", ""))
                if isinstance(value, str) and value.startswith("data:image"):
                    return Image.open(io.BytesIO(base64.b64decode(value.split(",", 1)[1]))).convert("RGB")

            content = msg.get("content", "")
            parts = content if isinstance(content, list) else [content]
            for part in parts:
                if isinstance(part, dict):
                    value = part.get("image_url") or part.get("data")
                    if isinstance(value, dict):
                        value = value.get("url", value.get("data", ""))
                else:
                    value = part
                if isinstance(value, str) and "data:image" in value:
                    value = value[value.find("data:image"):].split()[0]
                    return Image.open(io.BytesIO(base64.b64decode(value.split(",", 1)[1]))).convert("RGB")
        return None

    def _openai_image_edit(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        # RCouyi documents /images/edits as multipart/form-data for gpt-image-1.
        # The mask must be RGBA with transparent pixels marking the region to edit.
        alpha = np.full(mask.shape, 255, dtype=np.uint8)
        alpha[np.asarray(mask) > 30] = 0
        rgba_mask = Image.new("RGBA", image.size, (255, 255, 255, 255))
        rgba_mask.putalpha(Image.fromarray(alpha, "L"))

        files = {
            "image": ("image.png", self._png_bytes(image.convert("RGBA")), "image/png"),
            "mask": ("mask.png", self._png_bytes(rgba_mask), "image/png"),
        }
        data = {
            "model": self.model,
            "prompt": self._prompt(),
            "n": "1",
            "response_format": "b64_json",
        }
        # Avoid the unsupported/ambiguous size="auto" value on the proxy.
        # The service accepts an optional size; omitting it lets the upstream
        # image editor choose a valid output size for the cropped input.
        response = requests.post(
            f"{self.base_url}/images/edits",
            headers={"Authorization": f"Bearer {self.api_key}"},
            files=files,
            data=data,
            timeout=self.timeout,
        )
        self._raise_for_status(response)
        repaired = self._extract_image_from_response(response.json())
        if repaired is None:
            raise RuntimeError("OpenAI 兼容接口没有返回图片，请确认该模型支持 /images/edits。")
        return repaired

    def _chat_image_edit(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        mask_ref = self._mask_reference(image, mask)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": self._prompt()},
                {"type": "image_url", "image_url": {"url": self._data_url(image)}},
                {"type": "image_url", "image_url": {"url": self._data_url(mask_ref)}},
            ]}],
            "max_tokens": 4096,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(), json=payload, timeout=self.timeout,
        )
        self._raise_for_status(response)
        repaired = self._extract_image_from_response(response.json())
        if repaired is None:
            raise RuntimeError(
                "接口返回成功但没有图片。请确认这个 Gemini 模型支持图片输出/编辑，而不是只有图片理解。"
            )
        return repaired

    @staticmethod
    def _raise_for_status(response):
        if response.ok:
            return
        try:
            detail = response.json()
        except Exception:
            detail = response.text[:1000]
        raise RuntimeError(f"API HTTP {response.status_code}: {detail}")

    def repair(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        if not self.available:
            raise RuntimeError("未配置第三方 API。请设置 RC_API_KEY、RC_API_BASE_URL 和模型名。")
        image = image.convert("RGB")
        mask = np.where(np.asarray(mask) > 30, 255, 0).astype(np.uint8)
        mask = np.asarray(Image.fromarray(mask, "L").filter(ImageFilter.MaxFilter(9)), dtype=np.uint8)
        work_image, work_mask, box = self._crop_region(image, mask)

        if self.provider == "openai" and self.model.lower().startswith("gpt-image"):
            repaired = self._openai_image_edit(work_image, work_mask)
        else:
            repaired = self._chat_image_edit(work_image, work_mask)

        if repaired.size != work_image.size:
            repaired = repaired.resize(work_image.size, Image.Resampling.LANCZOS)
        result = image.copy()
        alpha = Image.fromarray(mask, "L").filter(ImageFilter.GaussianBlur(1.5))
        result.paste(repaired, box, alpha)
        return result
