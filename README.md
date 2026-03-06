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

## Requisitos

- Python 3.10+
- GPU recomendada (MPS en Apple Silicon, CUDA en Linux/Windows, o Google Colab T4)
