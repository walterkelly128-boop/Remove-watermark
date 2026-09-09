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
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --only-binary=:all: -r requirements.txt \
    && rm -rf /root/.cache/pip

COPY app.py gemini_app.py tools_app.py image_compress.py ai_provider_engine.py account_system.py recharge_system.py admin_app.py server.py ./
COPY core ./core
RUN mkdir -p /app/models /app/data /app/data/output

EXPOSE 7860
CMD ["python", "server.py"]
