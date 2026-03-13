import sys
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vit.inference.predecir import predecir_imagen
from vit.models.factory import cargar_modelo

CHECKPOINT = ROOT / "checkpoints" / "mobilevit_small" / "best_model.pt"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    modelo, procesador = cargar_modelo("apple/mobilevit-small", num_clases=7)
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    modelo.load_state_dict(ckpt["model_state_dict"])
    modelo.to(DEVICE).eval()
    _state["modelo"] = modelo
    _state["procesador"] = procesador
    _state["device"] = DEVICE
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
    return {"estado": "ok", "version": "0.1.0", "modelo": "mobilevit-small"}


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
