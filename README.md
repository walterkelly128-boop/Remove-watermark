# Remove Watermark - Gemini AI Docker Edition

用于处理你拥有或获得授权可以修改的图片。当前版本默认使用 **Gemini 3.1 Flash Image（Nano Banana 2）** 做局部 AI 图像重建，并保留自动候选检测 + 手动 Mask 编辑流程。

Google 官方文档显示，Gemini 3.1 Flash Image 支持图片输入、图片编辑以及对指定区域进行语义修改；它面向低延迟和高吞吐场景。citeturn0search0turn1search0

## 核心流程

```text
上传图片
  ↓
自动识别候选水印区域
  ↓
红色 Mask 预览
  ↓
画笔增加 / 橡皮擦删除
  ↓
局部裁剪 + Mask 扩张
  ↓
Gemini 3.1 Flash Image
  ↓
只替换 Mask 区域并贴回原图
```

### 为什么这一版不同

以前的版本主要依赖 LaMa/传统 inpainting。新版增加 Gemini 图像编辑作为主要修复引擎：程序不会直接把整张图交给 Gemini，而是根据 Mask 截取水印附近的局部区域，并同时发送原图和红色 Mask 参考图，要求模型只重建被标记区域。

这对明显的文字/Logo 覆盖、复杂背景和纹理区域通常比单纯模糊或简单 OpenCV 修复更合适。但生成式修复不是像素级无损恢复，人物脸部、文字和复杂物体仍可能发生细微变化。

## Quick start

```bash
git clone https://github.com/walterkelly128-boop/Remove-watermark.git
cd Remove-watermark
```

设置 Gemini API Key：

### Windows PowerShell

```powershell
$env:GEMINI_API_KEY="你的 Gemini API Key"
docker compose up --build
```

### Linux / macOS

```bash
export GEMINI_API_KEY="你的 Gemini API Key"
docker compose up --build
```

打开：

```text
http://localhost:7860
```

## Docker Compose

如果 compose 文件支持环境变量，建议写成：

```yaml
services:
  remove-watermark:
    build: .
    ports:
      - "7860:7860"
    environment:
      GEMINI_API_KEY: ${GEMINI_API_KEY}
      GEMINI_IMAGE_MODEL: gemini-3.1-flash-image
```

不要把真实 API Key 提交到 GitHub。

## 可选模型

默认：

```text
gemini-3.1-flash-image
```

也可以通过：

```text
GEMINI_IMAGE_MODEL
```

切换到你的 Gemini 图像模型版本。Google 当前文档将 Gemini 3.1 Flash Image 定位为速度、质量和成本之间的平衡型图像生成/编辑模型。citeturn0search7

## 使用步骤

1. 上传图片。
2. 点击 **自动识别候选区域**。
3. 查看红色 Mask。
4. 用画笔补充遗漏区域。
5. 用橡皮擦删除误识别区域。
6. 点击 **Gemini AI 智能修复**。
7. 查看修复结果。

自动检测只是候选区域生成器，并不保证能识别所有水印。最终 Mask 应由用户确认。

## API Key 安全

API Key 只在 Docker 后端读取 `GEMINI_API_KEY`，浏览器前端不会显示 Key。不要把 `.env`、API Key 或任何 Secret 提交到 GitHub。

## 合法使用

只对你拥有或明确获得授权可以修改的图片进行处理，并遵守图片来源平台的条款、版权和署名要求。
