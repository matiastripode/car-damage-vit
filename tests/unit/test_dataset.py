import json
from unittest.mock import patch, mock_open

import numpy as np
import pytest
from PIL import Image
from torchvision import transforms


# JSON COCO mínimo con 1 imagen, 2 anotaciones de daño (dent + scratch)
COCO_FAKE = {
    "images": [
        {"id": 1, "file_name": "/fake/imagen.jpg", "width": 500, "height": 500}
    ],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [50, 50, 80, 80]},
        {"id": 2, "image_id": 1, "category_id": 2, "bbox": [300, 300, 60, 60]},
    ],
    "categories": [],
}


def _crear_dataset(**kwargs):
    """Instancia CarDamageDataset con JSONs falsos, sin tocar el disco."""
    from vit.data.dataset import CarDamageDataset

    json_str = json.dumps(COCO_FAKE)
    with patch("pathlib.Path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=json_str)):
        return CarDamageDataset(split="train", **kwargs)


# ── Constantes ─────────────────────────────────────────────────────────────────

def test_clases_count():
    from vit.data.dataset import CLASES, CLASES_CON_FONDO
    assert len(CLASES) == 6
    assert len(CLASES_CON_FONDO) == 7
    assert CLASES_CON_FONDO[-1] == "fondo"


# ── Construcción de parches ─────────────────────────────────────────────────────

def test_dataset_tiene_parches():
    ds = _crear_dataset()
    # 2 positivos (dent + scratch) + 2 fondo (ratio_fondo=1.0 con semilla=42)
    assert len(ds) == 4


def test_dataset_sin_fondo():
    ds = _crear_dataset(incluir_fondo=False)
    assert len(ds) == 2


def test_etiquetas_en_rango():
    from vit.data.dataset import CLASES_CON_FONDO
    ds = _crear_dataset()
    n_clases = len(CLASES_CON_FONDO)
    for p in ds.parches:
        assert 0 <= p["etiqueta"] < n_clases


# ── __getitem__ ────────────────────────────────────────────────────────────────

def test_getitem_shape():
    ds = _crear_dataset()
    ds.transform = transforms.ToTensor()

    imagen_fake = Image.fromarray(np.zeros((500, 500, 3), dtype=np.uint8))
    with patch("PIL.Image.open", return_value=imagen_fake):
        parche, etiqueta = ds[0]

    assert parche.shape == (3, 224, 224)
    assert isinstance(etiqueta, int)


# ── pesos_clases ───────────────────────────────────────────────────────────────

def test_pesos_clases_shape():
    ds = _crear_dataset()
    pesos = ds.pesos_clases()
    assert len(pesos) == len(ds)
    assert all(w > 0 for w in pesos)
