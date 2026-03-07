import random

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset

CLASES = ["dent", "scratch", "crack", "glass_shatter", "tire_flat", "lamp_broken"]
CLASES_CON_FONDO = CLASES + ["fondo"]
TAMANO_PARCHE = 224

# ── Nombres de campos del dataset ─────────────────────────────────────────────
# Si la estructura del dataset es distinta, ajustar estos valores.
# Para verificar: llamar a inspeccionar_muestra() desde el notebook de EDA.
CAMPO_IMAGEN = "image"
CAMPO_OBJETOS = "objects"    # puede ser "annotations" según el dataset
CAMPO_BBOX = "bbox"          # formato esperado: [x, y, w, h]
CAMPO_CATEGORIA = "category"


class CarDamageDataset(Dataset):
    """
    Dataset de parches 224x224 extraídos de las imágenes de alta resolución del CarDD.

    Estrategia de parcheo (Opción B):
    - Un parche positivo centrado en cada instancia de daño anotada.
    - `ratio_fondo` parches de fondo por cada parche positivo, elegidos
      evitando zonas con anotaciones (IoU < umbral_iou).

    Args:
        split:          "train", "validation" o "test"
        transform:      transformaciones a aplicar sobre cada parche
        ratio_fondo:    parches de fondo por cada parche positivo (default 1.0)
        incluir_fondo:  si True, agrega parches de fondo como clase extra
        semilla:        semilla para reproducibilidad
    """

    def __init__(self, split="train", transform=None, ratio_fondo=1.0,
                 incluir_fondo=True, semilla=42):
        self.split = split
        self.transform = transform
        self.ratio_fondo = ratio_fondo
        self.incluir_fondo = incluir_fondo
        self.tamano = TAMANO_PARCHE
        self.clases = CLASES_CON_FONDO if incluir_fondo else CLASES

        random.seed(semilla)
        np.random.seed(semilla)

        print(f"Cargando CarDD [{split}]...")
        self.datos = load_dataset("harpreetsahota/CarDD", split=split)
        print(f"  → {len(self.datos)} imágenes")

        print("Construyendo lista de parches...")
        self.parches = self._construir_parches()
        print(f"  → {len(self.parches)} parches en total")

    def __len__(self):
        return len(self.parches)

    def __getitem__(self, idx):
        info = self.parches[idx]
        muestra = self.datos[info["idx_muestra"]]
        imagen = muestra[CAMPO_IMAGEN]

        parche = imagen.crop(info["coordenadas"])
        etiqueta = info["etiqueta"]

        if self.transform:
            parche = self.transform(parche)

        return parche, etiqueta

    # ── Construcción de parches ────────────────────────────────────────────────

    def _construir_parches(self):
        parches = []

        for idx_muestra, muestra in enumerate(self.datos):
            imagen = muestra[CAMPO_IMAGEN]
            ancho, alto = imagen.size

            anotaciones = self._extraer_anotaciones(muestra)
            if not anotaciones:
                continue

            # Un parche positivo por cada instancia de daño
            parches_pos = []
            for ann in anotaciones:
                coordenadas = self._centrar_parche(ann["bbox"], ancho, alto)
                parches_pos.append({
                    "idx_muestra": idx_muestra,
                    "coordenadas": coordenadas,
                    "etiqueta": ann["etiqueta_idx"],
                })

            parches.extend(parches_pos)

            # Parches de fondo proporcionales a los positivos
            if self.incluir_fondo:
                n_fondo = max(1, int(len(parches_pos) * self.ratio_fondo))
                parches_fondo = self._extraer_parches_fondo(
                    idx_muestra, anotaciones, n_fondo, ancho, alto
                )
                parches.extend(parches_fondo)

        return parches

    def _extraer_anotaciones(self, muestra):
        """
        Extrae las anotaciones en formato normalizado [{bbox, etiqueta_idx}].

        Si los campos del dataset no coinciden, ajustar las constantes
        CAMPO_OBJETOS, CAMPO_BBOX y CAMPO_CATEGORIA al inicio del archivo.
        """
        anotaciones = []
        objetos = muestra.get(CAMPO_OBJETOS, {})

        bboxes = objetos.get(CAMPO_BBOX, [])
        categorias = objetos.get(CAMPO_CATEGORIA, [])

        for bbox, categoria in zip(bboxes, categorias):
            nombre = _normalizar_categoria(categoria)
            if nombre not in CLASES:
                continue
            anotaciones.append({
                "bbox": bbox,
                "etiqueta_idx": CLASES.index(nombre),
            })

        return anotaciones

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

    def _extraer_parches_fondo(self, idx_muestra, anotaciones, n_parches, ancho_img, alto_img):
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
                        "idx_muestra": idx_muestra,
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalizar_categoria(categoria):
    """Convierte el valor de categoría del dataset al nombre interno de la clase."""
    if isinstance(categoria, int):
        return CLASES[categoria] if categoria < len(CLASES) else "desconocido"
    return str(categoria).lower().replace(" ", "_")


def inspeccionar_muestra(split="train"):
    """
    Carga una sola muestra e imprime su estructura completa.
    Llamar desde el notebook de EDA para verificar los nombres de los campos
    antes de entrenar.

    Ejemplo de uso:
        from vit.data.dataset import inspeccionar_muestra
        inspeccionar_muestra("train")
    """
    ds = load_dataset("harpreetsahota/CarDD", split=split)
    muestra = ds[0]
    print("Claves disponibles:", list(muestra.keys()))
    for clave, valor in muestra.items():
        if clave != CAMPO_IMAGEN:
            print(f"  {clave}: {valor}")
    print(f"  {CAMPO_IMAGEN}: PIL.Image de tamaño {muestra[CAMPO_IMAGEN].size}")
