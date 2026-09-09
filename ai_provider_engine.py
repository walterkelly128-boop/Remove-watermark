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
    """Image editing through an OpenAI-compatible third-party API endpoint.

    The provider is configurable so the real API key never reaches the browser.
    Gemini-style image models use /chat/completions; OpenAI image models use
    /images/edits when the model name starts with gpt-image.
    """

    def __init__(self, provider: str, api_key: Optional[str] = None, model: Optional[str] = None):
        self.provider = provider.lower().strip()
        self.api_key = api_key or os.environ.get("RC_API_KEY", "").strip()
        self.base_url = os.environ.get("RC_API_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        default = DEFAULT_GEMINI_MODEL if self.provider == "gemini" else DEFAULT_OPENAI_MODEL
        self.model = model or os.environ.get(
            "RC_GEMINI_MODEL" if self.provider == "gemini" else "RC_OPENAI_MODEL",
            default,
        )
        self.timeout = int(os.environ.get("RC_API_TIMEOUT", "180"))

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    @staticmethod
    def _png_bytes(image: Image.Image) -> bytes:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    @staticmethod
    def _data_url(image: Image.Image) -> str:
        return "data:image/png;base64," + base64.b64encode(RCImageEngine._png_bytes(image)).decode("ascii")

    @staticmethod
    def _mask_reference(image: Image.Image, mask: np.ndarray) -> Image.Image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        m = np.asarray(mask) > 30
        rgb[m] = (255, 40, 40)
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

    def _prompt(self) -> str:
        return (
            "Perform precise image restoration on an image the user is authorized to edit. "
            "The first image is the source. The second image is a red mask reference; "
            "red pixels identify the ONLY area that must be reconstructed. Remove the watermark, "
            "text, logo or overlay inside that region by naturally reconstructing the content "
            "that should be behind it. Change only the masked area. Preserve identity, faces, "
            "hair, objects, geometry, perspective, lighting, shadows, colors, texture, sharpness "
            "and composition outside the mask. Do not blur the repaired area. Do not add objects, "
            "text or logos. Return the restored image only."
        )

    @staticmethod
    def _extract_image_from_response(data: dict) -> Optional[Image.Image]:
        # OpenAI-compatible image response: data[].b64_json or data[].url
        for item in data.get("data", []) or []:
            b64 = item.get("b64_json") or item.get("base64")
            if b64:
                return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            url = item.get("url")
            if url:
                r = requests.get(url, timeout=120)
                r.raise_for_status()
                return Image.open(io.BytesIO(r.content)).convert("RGB")

        # Some OpenAI-compatible multimodal providers return images in message.images.
        for choice in data.get("choices", []) or []:
            msg = choice.get("message", {}) or {}
            for item in msg.get("images", []) or []:
                if isinstance(item, str):
                    value = item
                else:
                    value = item.get("image_url", item.get("data", ""))
                    if isinstance(value, dict):
                        value = value.get("url", value.get("data", ""))
                if isinstance(value, str) and value.startswith("data:image"):
                    return Image.open(io.BytesIO(base64.b64decode(value.split(",", 1)[1]))).convert("RGB")

            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    value = part.get("image_url") or part.get("data")
                    if isinstance(value, dict):
                        value = value.get("url", value.get("data", ""))
                    if isinstance(value, str) and value.startswith("data:image"):
                        return Image.open(io.BytesIO(base64.b64decode(value.split(",", 1)[1]))).convert("RGB")
            elif isinstance(content, str) and "data:image" in content:
                value = content[content.find("data:image"):]
                value = value.split()[0]
                return Image.open(io.BytesIO(base64.b64decode(value.split(",", 1)[1]))).convert("RGB")
        return None

    def _openai_image_edit(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        endpoint = f"{self.base_url}/images/edits"
        rgba = Image.new("RGBA", image.size, (255, 255, 255, 255))
        alpha = np.full(mask.shape, 255, dtype=np.uint8)
        alpha[np.asarray(mask) > 30] = 0
        rgba.putalpha(Image.fromarray(alpha, "L"))

        files = {
            "image": ("image.png", self._png_bytes(image), "image/png"),
            "mask": ("mask.png", self._png_bytes(rgba.convert("RGB")), "image/png"),
        }
        data = {"model": self.model, "prompt": self._prompt(), "size": "auto"}
        response = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},
            files=files,
            data=data,
            timeout=self.timeout,
        )
        self._raise_for_status(response)
        repaired = self._extract_image_from_response(response.json())
        if repaired is None:
            raise RuntimeError("OpenAI 兼容接口没有返回图片数据，请确认该模型支持 /images/edits。")
        return repaired

    def _chat_image_edit(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        mask_ref = self._mask_reference(image, mask)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._prompt()},
                        {"type": "image_url", "image_url": {"url": self._data_url(image)}},
                        {"type": "image_url", "image_url": {"url": self._data_url(mask_ref)}},
                    ],
                }
            ],
            "max_tokens": 4096,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        self._raise_for_status(response)
        repaired = self._extract_image_from_response(response.json())
        if repaired is None:
            raise RuntimeError(
                "接口返回成功但没有图片。请确认该 Gemini 模型支持图片输出/编辑；" 
                "普通文本 Gemini 模型不能直接完成图片修复。"
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
            raise RuntimeError("未配置第三方 API。请在 .env 设置 RC_API_KEY、RC_API_BASE_URL 和模型名。")
        image = image.convert("RGB")
        mask = np.where(np.asarray(mask) > 30, 255, 0).astype(np.uint8)
        expanded = Image.fromarray(mask, "L").filter(ImageFilter.MaxFilter(9))
        mask = np.asarray(expanded, dtype=np.uint8)

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
