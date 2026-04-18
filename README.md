# car-damage-vit

Clasificación de daños en vehículos usando Vision Transformers.

---

## Arquitectura del sistema

```mermaid
graph TB
    subgraph iPhone["📱 iPhone — SwiftUI App"]
        UI["ContentView\nPhotosPicker + ResultadoView\nToggle modo local/API"]
        SVC["APIService\nmultipart/form-data POST"]
        LC["LocalClassifier\nCoreML on-device"]
        MLPKG[("CarDamageClassifier\n.mlpackage\n~20 MB")]
        UI -->|"modo API"| SVC
        UI -->|"modo local"| LC
        LC -->|"carga"| MLPKG
        SVC -->|"Prediccion (clase, confianza, top3)"| UI
        LC -->|"Prediccion (clase, confianza, top3)"| UI
    end

    subgraph Backend["🐳 Docker Compose Stack"]
        NET{{"red interna\nbridge (edge)"}}

        subgraph C_TRF["contenedor traefik"]
            TRF["reverse proxy\nSSL/TLS\nmkcert localhost"]
        end

        subgraph C_WEB["contenedor web"]
            WEB["Streamlit UI\nclients/web/app.py"]
        end

        subgraph C_API["contenedor api"]
            API["FastAPI\napp/main.py"]
            INF["predecir_imagen()\nsrc/vit/inference/predecir.py"]
            MDL["MobileViT-small\ninferencia CPU"]
            CKPT[("best_model.pt\nsolo fallback local")]
            TRN["job train/eval\npipeline_entrenar_evaluar.py"]
        end

        subgraph C_MLF["contenedor mlflow"]
            MLF["tracking server + model registry"]
            VOL[("mlruns_data\nartifacts + sqlite")]
        end

        subgraph C_MNO["contenedor minio"]
            MNO["MinIO S3\nobject storage"]
            MNOV[("minio_data\nrlhf bucket")]
        end

        WEB --- NET
        API --- NET
        MLF --- NET
        TRF --- NET
        MNO --- NET

        TRF -->|"HTTP /ui"| WEB
        TRF -->|"HTTP /api"| API
        TRF -->|"HTTP /mlflow"| MLF

        API -->|"PIL Image"| INF
        INF -->|"tensor [1,3,224,224]"| MDL
        CKPT -.->|"usar si falla registry"| MDL
        MDL -->|"logits → softmax → top3"| INF
        INF -->|"JSON response"| API

        API <-->|"leer modelo registrado"| MLF
        API -->|"ROI + predicción\nfeedback YOLO"| MNO
        TRN -->|"registrar modelo + métricas"| MLF
        MLF -->|"persistir"| VOL
        MNO -->|"persistir"| MNOV
    end

    subgraph Training["🔬 Entrenamiento (offline)"]
        DS["CarDD dataset\n4000 imgs · 6 clases"]
        PATCH["CarDamageDataset\nparches 224×224"]
        DS --> PATCH --> TRN
    end

    SVC -->|"HTTPS https://localhost/api"| TRF
    TRF -->|"HTTPS response"| SVC
```

---

## Flujo de interacción — demo iPhone

```mermaid
sequenceDiagram
    actor Usuario
    participant App as ContentView<br/>(SwiftUI)
    participant Svc as APIService<br/>(Swift)
    participant TRF as Traefik<br/>(SSL offload · localhost)
    participant API as FastAPI<br/>POST /predecir
    participant Inf as predecir_imagen()
    participant Modelo as MobileViT-small<br/>(PyTorch · CPU)

    Usuario->>App: Selecciona foto\n(PhotosPicker)
    App->>App: onChange → carga UIImage\ncargando = true\nmuestra ProgressView
    App->>Svc: predecir(imagen: UIImage)
    Svc->>Svc: Convierte a JPEG (quality 0.8)\nArma multipart/form-data
    Svc->>TRF: HTTPS POST https://localhost/api/predecir
    TRF->>API: HTTP POST /predecir (strip /api)
    API->>API: Lee UploadFile\nconvierte bytes → PIL Image RGB
    API->>Inf: predecir_imagen(imagen, modelo, procesador, device)
    Inf->>Inf: Resize 224×224\nAplica eval transforms\ntensor [1, 3, 224, 224]
    Inf->>Modelo: forward(pixel_values=tensor)
    Modelo-->>Inf: logits [1, 7]
    Inf->>Inf: softmax → probs\nargmax → clase\ntop3 por prob desc.
    Inf-->>API: {clase, confianza, top3}
    API-->>TRF: HTTP 200 JSON
    TRF-->>Svc: HTTPS 200 JSON
    Svc->>Svc: JSONDecoder → Prediccion struct
    Svc-->>App: Prediccion(clase, confianza, top3)
    App->>App: prediccion = result\ncargando = false
    App->>Usuario: Muestra ResultadoView\n─ clase en mayúsculas\n─ barra de confianza\n─ top 3 con porcentajes
```

---

## Edge AI — Inferencia on-device (CoreML)

MobileViT-small fue exportado a `.mlpackage` mediante `scripts/exportar_coreml.py` y puede correr directamente en el Neural Engine del iPhone sin red ni servidor.

```mermaid
flowchart TD
    subgraph Train["🔬 Entrenamiento (offline, macOS)"]
        PT["best_model.pt\nMobileViT-small PyTorch\nval_f1 = 0.76"]
        SCRIPT["exportar_coreml.py\ntorch.jit.trace + coremltools 9"]
        NORM["_LogitsWrapper\nbake-in ImageNet norm\nmean/std como buffers"]
        PT --> NORM --> SCRIPT
    end

    subgraph Convert["⚙️ Conversión CoreML"]
        TRACE["TorchScript trace\nstrict=False"]
        MIL["MIL pipeline\n812 ops"]
        MLPKG[("CarDamageClassifier\n.mlpackage\n~20 MB")]
        SCRIPT --> TRACE --> MIL --> MLPKG
    end

    subgraph Xcode["🛠 Xcode"]
        GEN["Auto-genera\nCarDamageClassifierInput\nCarDamageClassifierOutput"]
        MLPKG -->|"drag & drop"| GEN
    end

    subgraph iPhone["📱 iPhone (on-device · sin red)"]
        UI["ContentView\nToggle ON → modo local"]
        LC["LocalClassifier.swift"]
        PIX["UIImage → CGImage\n→ CVPixelBuffer 224×224"]
        COREML["CoreML Runtime\nNeural Engine / CPU"]
        LOGITS["MultiArray Float32\n1×7 logits"]
        SM["Softmax manual\nSwift"]
        TOP3["Prediccion\nclase · confianza · top3"]

        UI -->|"UIImage"| LC
        LC --> PIX
        PIX -->|"pixel_values\nscale 1/255"| COREML
        COREML --> LOGITS
        LOGITS --> SM
        SM --> TOP3
        TOP3 -->|"ResultadoView"| UI
    end

    GEN -.->|"bundled en app"| COREML

    style Train fill:#1e3a5f,color:#fff
    style Convert fill:#2d4a2d,color:#fff
    style Xcode fill:#4a3a1e,color:#fff
    style iPhone fill:#3a1e4a,color:#fff
```

### Comparación de modos

| | Modo API | Modo Edge AI |
|---|---|---|
| **Dónde corre** | Docker (CPU) en Mac | Neural Engine del iPhone |
| **Red requerida** | Sí (HTTPS localhost) | No |
| **Latencia** | ~200–500 ms | ~50–150 ms |
| **RLHF / feedback** | Sí (MinIO) | No |
| **Toggle** | OFF | ON |

### Cómo generar el `.mlpackage`

```bash
conda activate car-damage-vit
python scripts/exportar_coreml.py
# → checkpoints/mobilevit_small/CarDamageClassifier.mlpackage
```

---

## Flujo de interacción — demo iPhone (modo Edge AI)

```mermaid
sequenceDiagram
    actor Usuario
    participant App as ContentView<br/>(SwiftUI)
    participant LC as LocalClassifier<br/>(Swift)
    participant CM as CoreML Runtime<br/>(Neural Engine · iPhone)
    participant SW as Softmax<br/>(Swift)

    Usuario->>App: Toggle "Modo sin conexión" ON
    Usuario->>App: Selecciona foto (PhotosPicker)
    App->>App: onChange → UIImage\ncargando = true
    App->>LC: LocalClassifier().predecir(imagen)
    LC->>LC: UIImage → CGImage\nCarDamageClassifierInput(pixel_valuesWith:)
    LC->>CM: model.prediction(input)
    Note over CM: scale 1/255 aplicado por ImageType<br/>ImageNet norm baked-in en wrapper<br/>812 ops MobileViT
    CM-->>LC: CarDamageClassifierOutput<br/>MultiArray Float32 1×7 (logits)
    LC->>SW: exp(logits) / Σexp(logits)
    SW-->>LC: probs [String: Double]\nsuman 1.0
    LC-->>App: Prediccion(clase, confianza, top3)<br/>rlhfStorage = nil
    App->>App: prediccion = result\ncargando = false
    App->>Usuario: ResultadoView\n─ clase en mayúsculas\n─ barra de confianza\n─ top 3 con porcentajes
    Note over App,Usuario: Sin red · Sin servidor · Sin latencia de red
```

---

## Flujo de corrección humana — demo iPhone

```mermaid
sequenceDiagram
    actor Usuario
    participant App as ContentView<br/>(SwiftUI)
    participant Svc as APIService<br/>(Swift)
    participant API as FastAPI<br/>POST /feedback
    participant MinIO as MinIO<br/>(bucket rlhf)

    Note over App: Modelo retornó predicción<br/>+ rlhf_storage.roi_key

    Usuario->>App: Pulsa "¿Predicción incorrecta? Corregir"
    App->>App: Muestra sheet con Picker<br/>de 7 clases
    Usuario->>App: Selecciona clase correcta\n(ej. "fondo")
    App->>App: Pulsa "Enviar"
    App->>Svc: enviarFeedback(roiKey, claseCorrecta)
    Svc->>API: POST /feedback\n{"roi_key": "...", "clase_correcta": "fondo"}
    API->>API: Valida clase\nDeriva base_key desde roi_key
    API->>MinIO: PUT <base_key>_feedback.json\n{clase_correcta, class_id, corrected_at}
    API->>MinIO: PUT <base_key>.txt\n"6 0.5 0.5 1.0 1.0" (YOLO)
    MinIO-->>API: 200 OK
    API-->>Svc: {"feedback_key": ..., "yolo_key": ...}
    Svc-->>App: success
    App->>Usuario: Muestra "✓ Corrección enviada"
```

---

## Flujo de interacción — interfaz web

```mermaid
sequenceDiagram
    actor Usuario
    participant Traefik as Traefik<br/>(SSL offload + routing)
    participant Web as Streamlit UI<br/>(clients/web/app.py)
    participant API as FastAPI<br/>(app/main.py)
    participant MLflow as MLflow Registry

    Usuario->>Traefik: HTTPS GET /
    Traefik->>Web: HTTP GET /
    Web-->>Traefik: HTTP 200 UI
    Traefik-->>Usuario: HTTPS 200 UI

    Usuario->>Traefik: HTTPS GET /api/
    Traefik->>API: HTTP GET /
    API-->>Traefik: HTTP 200 estado del modelo
    Traefik-->>Usuario: HTTPS 200 estado del modelo
    Usuario->>Web: Mostrar model_source / version / checkpoint

    opt Recargar modelo desde MLflow
        Usuario->>Traefik: HTTPS POST /api/modelo/recargar
        Traefik->>API: HTTP /modelo/recargar
        API->>MLflow: HTTP interno resolver versión registrada
        alt Carga registry OK
            MLflow-->>API: HTTP interno artefacto de modelo
            API-->>Traefik: HTTP 200 {model_source: mlflow_registry, version}
            Traefik-->>Usuario: HTTPS 200 {model_source: mlflow_registry, version}
            Usuario->>Web: Mostrar éxito de recarga
        else Falla registry
            API-->>Traefik: HTTP 200 {model_source: local_checkpoint}
            Traefik-->>Usuario: HTTPS 200 {model_source: local_checkpoint}
            Usuario->>Web: Informar fallback local
        end
    end

    Usuario->>Traefik: HTTPS POST /api/predecir (multipart image)
    Traefik->>API: HTTP /predecir
    API->>API: predecir_imagen(imagen, modelo, procesador, device)
    API->>API: forward modelo (registry o fallback)
    API->>API: softmax + top3
    API-->>Traefik: HTTP 200 JSON
    Traefik-->>Usuario: HTTPS 200 JSON
    Usuario->>Web: Mostrar predicción + top3 + confianza
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

## Cómo iniciar

Hay dos caminos según lo que querés hacer:

---

### Camino A — Correr el stack (API + UI + MLflow)

**Prerequisito:** Docker Desktop (macOS) o Docker Engine + Compose v2 (Linux).

> `mkcert` y sus dependencias los instala automáticamente `scripts/setup_dev.sh`.

#### A1. Variables de entorno (una vez)

```bash
cp .env_example .env
```

Editá `.env` y asigná credenciales para MinIO (cualquier valor sirve para desarrollo local):

```
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=admin1234
```

#### A2. Certificados TLS locales (una vez)

Genera certificados confiados por el sistema para `localhost`:

```bash
bash scripts/setup_dev.sh
```

> **Linux:** el script necesita `sudo` para instalar la CA y puede pedir contraseña.
> **Después de correr el script:** reiniciá el navegador completamente (Cmd+Q en macOS, cerrar todas las ventanas en Linux) para que tome el nuevo CA.

#### A3. Construir y levantar

```bash
docker compose build
docker compose up -d
```

Verificar que los 5 servicios estén `Up`:

```bash
docker compose ps
```

Servicios disponibles:

- Web UI (Streamlit): `https://localhost/ui/`
- API (vía Traefik): `https://localhost/api/`
- MLflow UI: `https://localhost/mlflow/`
- Traefik Dashboard: `https://localhost/dashboard/`
- MinIO Console (directo, sin Traefik): `http://localhost:9001`

---

### Camino B — Entrenar modelos

**Prerequisito:** Camino A corriendo (MLflow necesita estar activo para registrar experimentos).

#### B1. Crear el entorno local

```bash
conda env create -f environment.yml
conda activate car-damage-vit
```

#### B2. Configurar GPU

| Plataforma | Acción |
|---|---|
| macOS Apple Silicon (M1/M2/M3) | Nada extra — PyTorch usa MPS automáticamente |
| Linux con GPU NVIDIA | Ver abajo |

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verificar versión de CUDA: `nvidia-smi`

#### B3. Ejecutar pipeline end-to-end (train + eval)

`pipeline_entrenar_evaluar.py` corre en una sola corrida: prepara datos, entrena y evalúa sobre test.

```bash
python scripts/pipeline_entrenar_evaluar.py \
  --config model/mobilevit_small.yaml \
  --env dev --mlflow-uri http://localhost:6000 \
  --mlflow-train-experiment car-damage-vit-train \
  --mlflow-eval-experiment car-damage-vit-eval \
  --mlflow-register-name car-damage-mobilevit
```

Splits:
- `train` → optimizar el modelo por época
- `validation` → validar por época durante entrenamiento
- `test` → solo en evaluación final (`scripts/evaluar.py`)

Al iniciar el run, registrar el dataset en el campo **Dataset** de MLflow y subir artifacts de `data/raw/{train,validation,test,annotations}`.

#### B4. Registrar en Model Registry y asignar alias para serving

Pasar `--mlflow-register-name car-damage-mobilevit` registra el modelo en MLflow Model Registry.

Asignar alias `production` a la última versión para que la API lo cargue automáticamente:

```bash
python -c "from mlflow.tracking import MlflowClient; c=MlflowClient('http://localhost:6000'); name='car-damage-mobilevit'; v=max(c.search_model_versions(f\"name='{name}'\"), key=lambda m:int(m.version)); c.set_registered_model_alias(name, 'production', v.version); print(f'Alias production -> v{v.version}')"
```

> En la UI de MLflow → Model Registry, el modelo debe mostrar `Aliases: @ production`.

Si la carga desde Registry falla, la API usa como fallback automático `checkpoints/mobilevit_small/best_model.pt`.

---

## Docker

La UI web muestra el estado del modelo activo y ofrece el botón **Cargar última desde MLflow** (endpoint `POST /modelo/recargar`).

Si la carga desde Model Registry falla, la API usa fallback automático al checkpoint local `checkpoints/mobilevit_small/best_model.pt` e informa ese estado en la UI.

Los datos de tracking y artifacts de MLflow se persisten en el volumen `mlruns_data`.


## Desarrollo y debugging

### Correr la API fuera de Docker

Útil para iterar sobre `app/main.py` sin rebuilds:

```bash
conda activate car-damage-vit
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> Usar `python -m uvicorn` (no `uvicorn` directamente) para garantizar que se usa el Python del entorno conda y no el del sistema.

Verificar:

```bash
curl http://localhost:8000/
# {"estado":"ok","version":"0.1.0","modelo":"mobilevit-small"}
```

Probar inferencia:

```bash
curl -X POST http://localhost:8000/predecir \
  -F "archivo=@data/sample_test.jpg" | python3 -m json.tool
```

---

## Dependencias

El proyecto tiene tres archivos de dependencias según el contexto:

| Archivo | Uso | Deps incluidas |
|---|---|---|
| `requirements.txt` | Desarrollo completo (entrenamiento, notebooks, tests) | torch, transformers, datasets, fiftyone, scikit-learn, matplotlib, pytest, ... |
| `requirements-prod.txt` | API de inferencia en producción / Docker | torch CPU, transformers, fastapi, uvicorn, Pillow, python-multipart |
| `requirements-ci.txt` | Pipeline de CI (GitHub Actions) | subset para correr tests sin GPU |

Instalar según el contexto:

```bash
# Desarrollo local (entorno completo)
conda env create -f environment.yml

# Solo la API (sin conda, ej. servidor o Docker)
pip install -r requirements-prod.txt

# CI
pip install -r requirements-ci.txt
```
