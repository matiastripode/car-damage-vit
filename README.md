# car-damage-vit

Clasificación de daños en vehículos usando Vision Transformers.

---

## ¿De qué se trata?

Fine-tuning de modelos ViT livianos sobre el dataset CarDD para identificar y clasificar tipos de daños en autos. El dataset cubre seis categorías: abolladuras, rayones, fisuras, roturas de vidrio, neumáticos pinchados y faros dañados.

La estrategia central es trabajar con parches de 224×224 px extraídos de las imágenes originales de alta resolución, lo que permite usar modelos livianos sin perder detalle visual relevante.

---

## Modelos

| Modelo | Parámetros | Descripción |
|---|---|---|
| `facebook/deit-tiny-patch16-224` | ~5.7M | ViT compacto, eficiente en recursos |
| `apple/mobilevit-small` | ~5.6M | Híbrido CNN-Transformer para móviles |
| `openai/clip-vit-base-patch32` | ~151M | Evaluación zero-shot multimodal |

---

## Dataset

[CarDD](https://huggingface.co/datasets/harpreetsahota/CarDD) — 4.000 imágenes de alta resolución con anotaciones en formato COCO (bounding boxes + máscaras de segmentación). Uso no comercial.

**Clases:** dent · scratch · crack · glass shatter · tire flat · lamp broken

---

## Estructura del proyecto

```
car-damage-vit/
├── app/              # API de inferencia (FastAPI)
├── configs/          # Configuraciones por modelo y entorno
├── data/             # raw → interim → processed
├── docs/             # Documentación técnica
├── experiments/      # Registro de experimentos
├── logs/             # Logs de entrenamiento
├── models/           # Metadatos del modelo en producción
├── notebooks/        # EDA, entrenamiento y visualizaciones
├── reports/          # Métricas y figuras exportadas
├── scripts/          # Entry points CLI
├── src/vit/          # Código fuente principal
└── tests/            # Tests unitarios, de integración y e2e
```

---

## Cómo arrancar

### 1. Crear el entorno

```bash
conda env create -f environment.yml
conda activate car-damage-vit
```

### 2. GPU — configuración por plataforma

| Plataforma | Qué hacer |
|---|---|
| macOS Apple Silicon (M1/M2/M3) | Nada extra. PyTorch usa MPS automáticamente. |
| Linux / Windows con GPU NVIDIA | Ver abajo. |

En Linux o Windows con CUDA, después de crear el entorno:

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Para saber qué versión de CUDA tenés: `nvidia-smi`

### 3. Descargar el dataset

```bash
python scripts/descargar_dataset.py
```

### 4. Entrenar

```bash
python scripts/entrenar.py --config configs/model/deit_tiny.yaml --env dev
```

---

## API de inferencia

### Levantar la API localmente

```bash
conda activate car-damage-vit
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> Usar `python -m uvicorn` (no `uvicorn` directamente) para garantizar que se usa el Python del entorno conda y no el del sistema.

Verificar que está corriendo:

```bash
curl http://localhost:8000/
# {"estado":"ok","version":"0.1.0","modelo":"mobilevit-small"}
```

Probar inferencia:

```bash
curl -X POST http://localhost:8000/predecir \
  -F "archivo=@data/sample_test.jpg" | python3 -m json.tool
```

Respuesta esperada:

```json
{
    "clase": "scratch",
    "confianza": 0.87,
    "top3": [
        {"clase": "scratch", "confianza": 0.87},
        {"clase": "dent",    "confianza": 0.09},
        {"clase": "crack",   "confianza": 0.02}
    ]
}
```

### Exponer la API al iPhone (demo)

```bash
ngrok http 8000
```

Ngrok genera una URL pública HTTPS (ej. `https://abc123.ngrok-free.app`) que el iPhone puede usar desde cualquier red. La URL cambia con cada sesión en el plan gratuito.

Al hacer curl a través de ngrok, agregar el header para evitar la página de advertencia:

```bash
curl -X POST https://abc123.ngrok-free.app/predecir \
  -H "ngrok-skip-browser-warning: true" \
  -F "archivo=@data/sample_test.jpg" | python3 -m json.tool
```

### Dependencias adicionales de la API

Si uvicorn o fastapi no están en el entorno:

```bash
pip install uvicorn fastapi python-multipart
```

---

## Requisitos

- Python 3.10+
- GPU recomendada (MPS en Apple Silicon, CUDA en Linux/Windows, o Google Colab T4)
