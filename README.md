# Remove Watermark - CPU Docker Edition

AI image restoration tool for images you are authorized to edit. It combines automatic watermark candidate detection with manual mask painting/erasing, then uses LaMa inpainting to reconstruct the selected area instead of blurring it.

## Features

- CPU-only; no CUDA or GPU required
- Docker Desktop friendly
- JPG / PNG / WEBP input
- Automatic candidate detection for corner/text-like overlays
- Manual brush to add missed regions
- Eraser to remove false positives
- LaMa AI inpainting for content reconstruction
- Before/after preview and PNG download
- No account, payment, points, or API keys in V1

## Quick start

```bash
git clone https://github.com/walterkelly128-boop/Remove-watermark.git
cd Remove-watermark
docker compose up --build
```

Open http://localhost:7860

The first startup downloads the LaMa model and OCR model into the mounted `models` directory. CPU inference is slower than GPU inference; use moderate image sizes for testing.

## UI workflow

1. Upload an image.
2. Click **自动识别候选区域**.
3. Review the red mask overlay.
4. Use **添加区域** to paint missed areas.
5. Use **擦除区域** to remove false positives.
6. Click **AI 智能修复**.
7. Download the result.

Automatic detection is intentionally conservative. It is a candidate generator, not a guarantee that every text/logo in an image is a watermark. Review the mask before processing.

## License note

Only use this software on images you own or are authorized to modify. Respect creator attribution, platform terms, and applicable law.
