from __future__ import annotations

from io import BytesIO
from PIL import Image

MAX_IMAGE_MB = 25


def compress_image(image, quality=80, output_format="保持原格式"):
    if image is None:
        raise ValueError("请先上传图片。")
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.copy()
    quality = max(10, min(95, int(quality)))
    original_format = (image.format or "JPEG").upper()
    if output_format == "JPG":
        fmt = "JPEG"
    elif output_format == "PNG":
        fmt = "PNG"
    elif output_format == "WebP":
        fmt = "WEBP"
    else:
        fmt = original_format if original_format in {"JPEG", "PNG", "WEBP"} else "JPEG"

    if fmt == "JPEG" and image.mode not in ("RGB", "L"):
        if image.mode == "RGBA":
            bg = Image.new("RGB", image.size, "white")
            bg.paste(image, mask=image.getchannel("A"))
            image = bg
        else:
            image = image.convert("RGB")

    out = BytesIO()
    save_kwargs = {"format": fmt, "optimize": True}
    if fmt in ("JPEG", "WEBP"):
        save_kwargs["quality"] = quality
    image.save(out, **save_kwargs)
    data = out.getvalue()
    if not data:
        raise ValueError("图片压缩失败。")
    return data, fmt


def format_size(size):
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / 1024 / 1024:.2f} MB"
