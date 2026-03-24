import sys
from contextlib import asynccontextmanager
from io import BytesIO
import os
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vit.inference.predecir import predecir_imagen
from vit.models.factory import cargar_modelo

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

_state = {}


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint = _resolver_checkpoint()
    modelo, procesador = cargar_modelo("apple/mobilevit-small", num_clases=7)
    ckpt = torch.load(checkpoint, map_location="cpu")
    modelo.load_state_dict(ckpt["model_state_dict"])
    modelo.to(DEVICE).eval()
    _state["modelo"] = modelo
    _state["procesador"] = procesador
    _state["device"] = DEVICE
    _state["checkpoint"] = str(checkpoint)
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
        "modelo": "mobilevit-small",
        "checkpoint": _state.get("checkpoint"),
    }


@app.post("/predecir")
async def predecir(archivo: UploadFile = File(...)):
    if not archivo.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Se requiere una imagen")
    contenido = await archivo.read()
    imagen = Image.open(BytesIO(contenido)).convert("RGB")
    resultado = predecir_imagen(
        imagen,
        _state["modelo"],
        _state["procesador"],
        _state["device"],
    )
    return resultado
