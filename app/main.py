import sys
from contextlib import asynccontextmanager
from io import BytesIO
import os
from pathlib import Path
from threading import RLock
from datetime import datetime, timezone

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vit.inference.predecir import predecir_imagen
from vit.models.factory import cargar_modelo

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "apple/mobilevit-small")
NUM_CLASSES = int(os.getenv("NUM_CLASSES", "7"))

_state = {}
_model_lock = RLock()


def _resolver_checkpoint() -> Path:
    """
    Devuelve el checkpoint a cargar:
    1) `CHECKPOINT_PATH` explícito (si existe),
    2) archivo .pt más reciente en checkpoints/mobilevit_small/,
    3) fallback a checkpoints/mobilevit_small/best_model.pt
    """
    override = os.getenv("CHECKPOINT_PATH")
    if override:
        p = Path(override)
        if p.exists():
            return p
        raise FileNotFoundError(f"CHECKPOINT_PATH no existe: {p}")

    ckpt_dir = ROOT / "checkpoints" / "mobilevit_small"
    candidates = [p for p in ckpt_dir.glob("*.pt") if p.is_file()]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    fallback = ckpt_dir / "best_model.pt"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No se encontró checkpoint en: {ckpt_dir}")


def _cargar_modelo_desde_registry(force_latest: bool = False):
    """
    Intenta cargar modelo desde MLflow Model Registry.
    Prioridad:
    1) alias explícito (MLFLOW_MODEL_ALIAS),
    2) versión más alta en stage Production/Staging,
    3) versión numérica más reciente.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    model_name = os.getenv("MLFLOW_MODEL_NAME")
    model_alias = os.getenv("MLFLOW_MODEL_ALIAS")

    if not tracking_uri or not model_name:
        return None, None

    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except Exception as e:
        print(f"[startup] MLflow no disponible en runtime: {e}. Fallback a checkpoint local.")
        return None, None

    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)

        if model_alias and not force_latest:
            uri = f"models:/{model_name}@{model_alias}"
            model = mlflow.pytorch.load_model(uri)
            mv = client.get_model_version_by_alias(model_name, model_alias)
            return model, {
                "source": "mlflow_registry",
                "model_uri": uri,
                "model_name": model_name,
                "model_version": str(mv.version),
                "model_stage": (mv.current_stage or "").lower(),
            }

        versions = list(client.search_model_versions(f"name='{model_name}'"))
        if not versions:
            raise RuntimeError(f"No hay versiones para el modelo registrado '{model_name}'")

        if force_latest:
            best = max(versions, key=lambda v: int(v.version))
            uri = f"models:/{model_name}/{best.version}"
            model = mlflow.pytorch.load_model(uri)
            return model, {
                "source": "mlflow_registry",
                "model_uri": uri,
                "model_name": model_name,
                "model_version": str(best.version),
                "model_stage": (best.current_stage or "").lower(),
            }

        def _rank(v):
            stage = (v.current_stage or "").lower()
            stage_rank = {"production": 0, "staging": 1}.get(stage, 2)
            return (stage_rank, -int(v.version))

        best = sorted(versions, key=_rank)[0]
        uri = f"models:/{model_name}/{best.version}"
        model = mlflow.pytorch.load_model(uri)
        return model, {
            "source": "mlflow_registry",
            "model_uri": uri,
            "model_name": model_name,
            "model_version": str(best.version),
            "model_stage": (best.current_stage or "").lower(),
        }
    except Exception as e:
        print(f"[startup] Falló carga desde MLflow Registry: {e}. Fallback a checkpoint local.")
        return None, None


def _cargar_modelo_desde_checkpoint_local(modelo_base):
    checkpoint = _resolver_checkpoint()
    ckpt = torch.load(checkpoint, map_location="cpu")
    modelo_base.load_state_dict(ckpt["model_state_dict"])
    return modelo_base, {
        "source": "local_checkpoint",
        "checkpoint": str(checkpoint),
    }


def _aplicar_modelo_en_estado(modelo, procesador, meta: dict):
    modelo.to(DEVICE).eval()
    _state["modelo"] = modelo
    _state["procesador"] = procesador
    _state["device"] = DEVICE
    _state["model_source"] = meta.get("source")
    _state["model_uri"] = meta.get("model_uri")
    _state["model_name"] = meta.get("model_name")
    _state["model_version"] = meta.get("model_version")
    _state["model_stage"] = meta.get("model_stage")
    _state["checkpoint"] = meta.get("checkpoint")
    _state["loaded_at"] = datetime.now(timezone.utc).isoformat()


def _recargar_modelo(prefer_latest_mlflow: bool) -> dict:
    modelo_base, procesador = cargar_modelo(BASE_MODEL_NAME, num_clases=NUM_CLASSES)
    modelo, meta = _cargar_modelo_desde_registry(force_latest=prefer_latest_mlflow)

    if modelo is None:
        modelo, meta = _cargar_modelo_desde_checkpoint_local(modelo_base)

    _aplicar_modelo_en_estado(modelo, procesador, meta)
    return {
        "ok": True,
        "model_source": _state.get("model_source"),
        "model_uri": _state.get("model_uri"),
        "model_name": _state.get("model_name"),
        "model_version": _state.get("model_version"),
        "model_stage": _state.get("model_stage"),
        "checkpoint": _state.get("checkpoint"),
        "loaded_at": _state.get("loaded_at"),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    with _model_lock:
        _recargar_modelo(prefer_latest_mlflow=False)
    yield
    _state.clear()


app = FastAPI(title="car-damage-vit", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def raiz():
    return {
        "estado": "ok",
        "version": "0.1.0",
        "modelo": BASE_MODEL_NAME,
        "model_source": _state.get("model_source"),
        "model_uri": _state.get("model_uri"),
        "model_name": _state.get("model_name"),
        "model_version": _state.get("model_version"),
        "model_stage": _state.get("model_stage"),
        "checkpoint": _state.get("checkpoint"),
        "loaded_at": _state.get("loaded_at"),
    }


@app.post("/modelo/recargar")
def recargar_modelo():
    """
    Intenta recargar en caliente la última versión del modelo en MLflow Registry.
    Si falla, mantiene fallback automático a checkpoint local.
    """
    try:
        with _model_lock:
            return _recargar_modelo(prefer_latest_mlflow=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo recargar modelo: {e}")


@app.post("/predecir")
async def predecir(archivo: UploadFile = File(...)):
    if not archivo.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Se requiere una imagen")
    contenido = await archivo.read()
    imagen = Image.open(BytesIO(contenido)).convert("RGB")
    with _model_lock:
        modelo = _state["modelo"]
        procesador = _state["procesador"]
        device = _state["device"]

    resultado = predecir_imagen(
        imagen,
        modelo,
        procesador,
        device,
    )
    return resultado
