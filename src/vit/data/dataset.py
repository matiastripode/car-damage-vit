import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

CLASES = ["dent", "scratch", "crack", "glass_shatter", "tire_flat", "lamp_broken"]
CLASES_CON_FONDO = CLASES + ["fondo"]
TAMANO_PARCHE = 224

# Mapeo COCO category_id (base 1) → índice de clase interno (base 0)
CAT_ID_A_CLASE = {i + 1: nombre for i, nombre in enumerate(CLASES)}

# Ruta por defecto a los JSONs COCO generados por exportar_anotaciones.py
_RAIZ_PROYECTO = Path(__file__).resolve().parent.parent.parent.parent
RUTA_ANOTACIONES = _RAIZ_PROYECTO / "data" / "raw" / "annotations"


class CarDamageDataset(Dataset):
    """
    Dataset de parches 224x224 extraídos de las imágenes de alta resolución del CarDD.

    Las anotaciones se leen desde los archivos COCO JSON generados por
    scripts/exportar_anotaciones.py. Cada archivo contiene rutas absolutas
    a las imágenes originales junto con sus bounding boxes y categorías.

    Estrategia de parcheo (Opción B):
    - Un parche positivo centrado en cada instancia de daño anotada.
    - `ratio_fondo` parches de fondo por cada parche positivo, elegidos
      evitando zonas con anotaciones (IoU < umbral_iou).

    Args:
        split:          "train", "validation" o "test"
        transform:      transformaciones a aplicar sobre cada parche (PIL → Tensor)
        ratio_fondo:    parches de fondo por cada parche positivo (default 1.0)
        incluir_fondo:  si True, agrega parches de fondo como clase extra
        semilla:        semilla para reproducibilidad del muestreo de fondo
        ruta_anots:     ruta alternativa al directorio de JSONs COCO (opcional)
    """

    def __init__(self, split="train", transform=None, ratio_fondo=1.0,
                 incluir_fondo=True, semilla=42, ruta_anots=None):
        self.split = split
        self.transform = transform
        self.ratio_fondo = ratio_fondo
        self.incluir_fondo = incluir_fondo
        self.tamano = TAMANO_PARCHE
        self.clases = CLASES_CON_FONDO if incluir_fondo else CLASES

        random.seed(semilla)
        np.random.seed(semilla)

        ruta = Path(ruta_anots) if ruta_anots else RUTA_ANOTACIONES
        ruta_json = ruta / f"{split}.json"

        if not ruta_json.exists():
            raise FileNotFoundError(
                f"No se encontró {ruta_json}.\n"
                "Corré primero: python scripts/exportar_anotaciones.py"
            )

        print(f"Cargando anotaciones [{split}] desde {ruta_json}...")
        with open(ruta_json) as f:
            coco = json.load(f)

        self._imagenes = {img["id"]: img for img in coco["images"]}
        self._anots_por_imagen = _agrupar_por_imagen(coco["annotations"])
        print(f"  → {len(self._imagenes)} imágenes | {len(coco['annotations'])} anotaciones")

        print("Construyendo lista de parches...")
        self.parches = self._construir_parches()
        print(f"  → {len(self.parches)} parches en total")

    def __len__(self):
        return len(self.parches)

    def __getitem__(self, idx):
        info = self.parches[idx]
        imagen = Image.open(info["ruta_imagen"]).convert("RGB")
        parche = imagen.crop(info["coordenadas"])
        etiqueta = info["etiqueta"]

        if self.transform:
            parche = self.transform(parche)

        return parche, etiqueta

    # ── Construcción de parches ────────────────────────────────────────────────

    def _construir_parches(self):
        parches = []

        for img_id, img_info in self._imagenes.items():
            anots_coco = self._anots_por_imagen.get(img_id, [])
            if not anots_coco:
                continue

            ancho = img_info["width"]
            alto = img_info["height"]
            ruta_imagen = img_info["file_name"]

            anotaciones = []
            for ann in anots_coco:
                nombre = CAT_ID_A_CLASE.get(ann["category_id"])
                if nombre is None:
                    continue
                anotaciones.append({
                    "bbox": ann["bbox"],
                    "etiqueta_idx": CLASES.index(nombre),
                })

            if not anotaciones:
                continue

            # Un parche positivo centrado en cada bounding box
            parches_pos = []
            for ann in anotaciones:
                coordenadas = self._centrar_parche(ann["bbox"], ancho, alto)
                parches_pos.append({
                    "ruta_imagen": ruta_imagen,
                    "coordenadas": coordenadas,
                    "etiqueta": ann["etiqueta_idx"],
                })

            parches.extend(parches_pos)

            # Parches de fondo proporcionales a los positivos
            if self.incluir_fondo:
                n_fondo = max(1, int(len(parches_pos) * self.ratio_fondo))
                parches_fondo = self._extraer_parches_fondo(
                    ruta_imagen, anotaciones, n_fondo, ancho, alto
                )
                parches.extend(parches_fondo)

        return parches

    def _centrar_parche(self, bbox, ancho_img, alto_img):
        """Calcula coordenadas (x1, y1, x2, y2) de un parche centrado en el bbox."""
        x, y, w, h = [int(v) for v in bbox]
        t = self.tamano

        cx = x + w // 2
        cy = y + h // 2

        x1 = cx - t // 2
        y1 = cy - t // 2

        # Ajustar si el parche se sale de los bordes de la imagen
        x1 = max(0, min(x1, ancho_img - t))
        y1 = max(0, min(y1, alto_img - t))

        return (x1, y1, x1 + t, y1 + t)

    def _extraer_parches_fondo(self, ruta_imagen, anotaciones, n_parches, ancho_img, alto_img):
        """Extrae parches aleatorios que no se superpongan significativamente con daños."""
        t = self.tamano
        MAX_INTENTOS = 50
        parches = []

        for _ in range(n_parches):
            for _ in range(MAX_INTENTOS):
                x1 = random.randint(0, max(0, ancho_img - t))
                y1 = random.randint(0, max(0, alto_img - t))

                if not self._solapa_con_daño([x1, y1, t, t], anotaciones):
                    parches.append({
                        "ruta_imagen": ruta_imagen,
                        "coordenadas": (x1, y1, x1 + t, y1 + t),
                        "etiqueta": len(CLASES),  # índice de "fondo"
                    })
                    break

        return parches

    def _solapa_con_daño(self, bbox_parche, anotaciones, umbral_iou=0.1):
        """Devuelve True si el parche se superpone con alguna anotación de daño."""
        px, py, pw, ph = bbox_parche

        for ann in anotaciones:
            ax, ay, aw, ah = [int(v) for v in ann["bbox"]]

            ix1 = max(px, ax)
            iy1 = max(py, ay)
            ix2 = min(px + pw, ax + aw)
            iy2 = min(py + ph, ay + ah)

            if ix2 <= ix1 or iy2 <= iy1:
                continue

            interseccion = (ix2 - ix1) * (iy2 - iy1)
            union = pw * ph + aw * ah - interseccion
            iou = interseccion / union if union > 0 else 0

            if iou > umbral_iou:
                return True

        return False

    # ── Utilidades ─────────────────────────────────────────────────────────────

    def distribucion_clases(self):
        """Devuelve el conteo de parches por clase. Útil para el EDA."""
        from collections import Counter
        conteo = Counter(p["etiqueta"] for p in self.parches)
        return {self.clases[k]: v for k, v in sorted(conteo.items())}

    def pesos_clases(self):
        """
        Devuelve un tensor de pesos por muestra para WeightedRandomSampler.
        Las clases con menos muestras reciben mayor peso.
        """
        import torch
        conteo = [0] * len(self.clases)
        for p in self.parches:
            conteo[p["etiqueta"]] += 1

        pesos_por_clase = [1.0 / c if c > 0 else 0.0 for c in conteo]
        return torch.tensor([pesos_por_clase[p["etiqueta"]] for p in self.parches])


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _agrupar_por_imagen(anotaciones):
    """Agrupa las anotaciones COCO por image_id."""
    grupos = {}
    for ann in anotaciones:
        grupos.setdefault(ann["image_id"], []).append(ann)
    return grupos
