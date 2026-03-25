# car-damage-vit

Clasificación de daños en vehículos usando Vision Transformers.

---

## Arquitectura del sistema

```mermaid
graph TB
    subgraph iPhone["📱 iPhone — SwiftUI App"]
        UI["ContentView\nPhotosPicker + ResultadoView"]
        SVC["APIService\nmultipart/form-data POST"]
        UI -->|"UIImage seleccionada"| SVC
        SVC -->|"Prediccion (clase, confianza, top3)"| UI
    end

    subgraph Tunnel["🌐 Túnel ngrok"]
        NG["ngrok\nhttps://xxx.ngrok-free.app"]
    end

    subgraph Backend["🖥️ Mac — FastAPI (uvicorn :8000)"]
        EP["POST /predecir\napp/main.py"]
        INF["predecir_imagen()\nsrc/vit/inference/predecir.py"]
        MDL["MobileViT-small\n~5.6M params · MPS"]
        CKPT[("best_model.pt\ncheckpoints/mobilevit_small/")]
        EP -->|"PIL Image"| INF
        INF -->|"tensor [1,3,224,224]"| MDL
        CKPT -.->|"load_state_dict"| MDL
        MDL -->|"logits → softmax → top3"| INF
        INF -->|"JSON response"| EP
    end

    subgraph Training["🔬 Entrenamiento (offline)"]
        DS["CarDD dataset\n4000 imgs · 6 clases"]
        PATCH["CarDamageDataset\nparches 224×224"]
        TR["Fine-tuning\nlr=2e-4 · batch=32 · MPS"]
        DS --> PATCH --> TR --> CKPT
    end

    SVC -->|"HTTPS POST\n/predecir"| NG
    NG -->|"HTTP → localhost:8000"| EP
    EP -->|"JSON"| NG
    NG -->|"HTTPS"| SVC
```

---

## Flujo de interacción — demo iPhone

```mermaid
sequenceDiagram
    actor Usuario
    participant App as ContentView<br/>(SwiftUI)
    participant Svc as APIService<br/>(Swift)
    participant Ngrok as ngrok<br/>(túnel HTTPS)
    participant API as FastAPI<br/>POST /predecir
    participant Inf as predecir_imagen()
    participant Modelo as MobileViT-small<br/>(PyTorch · MPS)

    Usuario->>App: Selecciona foto\n(PhotosPicker)
    App->>App: onChange → carga UIImage\ncargando = true\nmuestra ProgressView
    App->>Svc: predecir(imagen: UIImage)
    Svc->>Svc: Convierte a JPEG (quality 0.8)\nArma multipart/form-data
    Svc->>Ngrok: POST /predecir\nContent-Type: multipart/form-data
    Ngrok->>API: HTTP POST → localhost:8000/predecir
    API->>API: Lee UploadFile\nconvierte bytes → PIL Image RGB
    API->>Inf: predecir_imagen(imagen, modelo, procesador, device)
    Inf->>Inf: Resize 224×224\nAplica eval transforms\ntensor [1, 3, 224, 224]
    Inf->>Modelo: forward(pixel_values=tensor)
    Modelo-->>Inf: logits [1, 7]
    Inf->>Inf: softmax → probs\nargmax → clase\ntop3 por prob desc.
    Inf-->>API: {clase, confianza, top3}
    API-->>Ngrok: HTTP 200 JSON
    Ngrok-->>Svc: HTTPS 200 JSON
    Svc->>Svc: JSONDecoder → Prediccion struct
    Svc-->>App: Prediccion(clase, confianza, top3)
    App->>App: prediccion = result\ncargando = false
    App->>Usuario: Muestra ResultadoView\n─ clase en mayúsculas\n─ barra de confianza\n─ top 3 con porcentajes
```

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
python scripts/entrenar.py --config model/deit_tiny.yaml --env dev
```

### 5. Entrenar con tracking en MLflow

Si levantaste MLflow en Docker Compose:

```bash
docker compose up -d mlflow
```

Entrenamiento con logging de métricas por epoch, params, checkpoint e historial:

```bash
python scripts/entrenar.py \
  --config model/mobilevit_small.yaml \
  --env dev \
  --mlflow-uri http://localhost:6000 \
  --mlflow-experiment car-damage-vit-train
```

Además, al iniciar el run de entrenamiento se registra el dataset en el campo **Dataset** de MLflow y se suben como artifacts los contenidos de `data/raw/{train,validation,test}` y `data/raw/annotations`.

Si `nvidia-ml-py` está instalado y hay GPU NVIDIA disponible, MLflow también registra métricas de GPU (`system/gpu_*`) además de CPU/memoria/disco.

Opcional: registrar el modelo en Model Registry de MLflow:

```bash
python scripts/entrenar.py \
  --config model/mobilevit_small.yaml \
  --env dev \
  --mlflow-uri http://localhost:6000 \
  --mlflow-experiment car-damage-vit-train \
  --mlflow-register-name car-damage-mobilevit
```

Para que la API cargue desde Registry por alias (configurada con `MLFLOW_MODEL_ALIAS=production`), asigná el alias a la última versión registrada:

```bash
python -c "from mlflow.tracking import MlflowClient; c=MlflowClient('http://localhost:6000'); name='car-damage-mobilevit'; v=max(c.search_model_versions(f\"name='{name}'\"), key=lambda m:int(m.version)); c.set_registered_model_alias(name, 'production', v.version); print(f'Alias production -> v{v.version}')"
```

### 5.1 Pipeline completo recomendado (preparación + train + eval)

Comando de referencia para correr el flujo end-to-end con registro en MLflow y Model Registry:

```bash
python scripts/pipeline_entrenar_evaluar.py --config model/mobilevit_small.yaml --env dev --mlflow-uri http://localhost:6000 --mlflow-train-experiment car-damage-vit-train --mlflow-eval-experiment car-damage-vit-eval --mlflow-register-name car-damage-mobilevit
```

### 5.2 Comando recomendado para generar run + Dataset + registro de modelo

Este comando crea un run de entrenamiento en MLflow, guarda el Dataset en el campo **Dataset**, sube artifacts de `data/raw` y registra el modelo en Model Registry:

```bash
python scripts/entrenar.py --config model/mobilevit_small.yaml --env dev --mlflow-uri http://localhost:6000 --mlflow-experiment car-damage-vit-train --mlflow-register-name car-damage-mobilevit
```

### 6. Evaluar checkpoint y loguear test en MLflow

```bash
python scripts/evaluar.py \
  --checkpoint checkpoints/mobilevit_small/best_model.pt \
  --config model/mobilevit_small.yaml \
  --env dev \
  --mlflow-uri http://localhost:6000 \
  --mlflow-experiment car-damage-vit-eval
```

Para vincular el run de evaluación con uno de entrenamiento:

```bash
python scripts/evaluar.py \
  --checkpoint checkpoints/mobilevit_small/best_model.pt \
  --config model/mobilevit_small.yaml \
  --env dev \
  --mlflow-uri http://localhost:6000 \
  --mlflow-experiment car-damage-vit-eval \
  --train-run-id <RUN_ID_DE_TRAIN>
```

Esto genera además:

- `reports/eval/mobilevit_small/metricas_test.json`
- `reports/eval/mobilevit_small/confusion_matrix_test.png`

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

---

## Docker

### Stack completo con Docker Compose (recomendado)

Construir imágenes del stack:

```bash
docker compose build
```

Levantar servicios:

```bash
docker compose up -d
```

Servicios principales:

- Web UI (Streamlit): `https://localhost/`
- API (via Traefik): `https://localhost/api/`
- MLflow UI: `http://localhost:6000/mlflow`

La UI web muestra el estado del modelo activo de la API y tiene el botón **Cargar ultima desde MLflow** (endpoint `POST /modelo/recargar`).

Si la carga desde Model Registry falla por cualquier motivo, la API mantiene fallback automático al checkpoint local `checkpoints/mobilevit_small/best_model.pt`, y la UI lo informa en pantalla.

Los datos de tracking y artifacts de MLflow persisten en el volumen `mlruns_data`.

La API puede levantarse como contenedor sin necesidad de instalar el entorno conda.
Usa `requirements-prod.txt` (solo deps de inferencia) en lugar del `requirements.txt` completo.

```bash
docker build -t car-damage-vit .
docker run -p 8000:8000 car-damage-vit
```

> Dentro del contenedor el device es CPU (Docker en Mac no expone MPS). La latencia para una imagen es comparable a la de MPS para inferencia individual.

---

## Dependencias

El proyecto tiene tres archivos de dependencias según el contexto:

| Archivo | Uso | Deps incluidas |
|---|---|---|
| `requirements.txt` | Desarrollo completo (entrenamiento, notebooks, tests) | torch, transformers, datasets, fiftyone, scikit-learn, matplotlib, pytest, ... |
| `requirements-prod.txt` | API de inferencia en producción / Docker | torch CPU, transformers, fastapi, uvicorn, Pillow, python-multipart |
| `requirements-ci.txt` | Pipeline de CI (GitHub Actions) | subset para correr tests sin GPU |

Para instalar según el contexto:

```bash
# Desarrollo local (entorno completo)
conda env create -f environment.yml

# Solo la API (sin conda, ej. servidor o Docker)
pip install -r requirements-prod.txt

# CI
pip install -r requirements-ci.txt
```
