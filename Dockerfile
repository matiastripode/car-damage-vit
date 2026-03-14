# ── build stage ──────────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /app

# Instalar torch CPU-only (sin CUDA) para mantener la imagen liviana (~1.5 GB vs ~6 GB)
RUN pip install --no-cache-dir \
    torch==2.2.0+cpu \
    torchvision==0.17.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements-prod.txt .
# torch/torchvision ya instalados arriba; el resto viene del requirements-api
RUN pip install --no-cache-dir \
    transformers>=4.35.0 \
    Pillow>=10.0.0 \
    fastapi>=0.104.0 \
    uvicorn>=0.24.0 \
    python-multipart>=0.0.6

# ── runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# Copiar paquetes instalados desde el builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Código fuente y checkpoint
COPY src/  src/
COPY app/  app/
COPY checkpoints/mobilevit_small/best_model.pt checkpoints/mobilevit_small/best_model.pt

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
