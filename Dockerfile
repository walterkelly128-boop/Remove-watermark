FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_VISIBLE_DEVICES="" \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libgomp1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/*

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --only-binary=:all: -r requirements.txt \
    && rm -rf /root/.cache/pip

COPY app.py .
COPY gemini_app.py .
COPY tools_app.py .
COPY image_compress.py .
COPY ai_provider_engine.py .
COPY account_system.py .
COPY recharge_system.py .
COPY core ./core
RUN mkdir -p /app/models /app/data /app/data/output

EXPOSE 7860
CMD ["python", "tools_app.py"]
