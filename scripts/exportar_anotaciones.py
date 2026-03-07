"""
Exporta las anotaciones de CarDD desde FiftyOne a formato COCO JSON.

El dataset harpreetsahota/CarDD almacena bounding boxes y categorías en formato
FiftyOne, no accesibles via la API estándar de HuggingFace datasets. Este script
los extrae y genera un archivo JSON por split en data/raw/annotations/.

La partición train/validation/test usa la misma semilla que descargar_dataset.py
(default: 42) para garantizar consistencia entre equipos.

Uso:
    python scripts/exportar_anotaciones.py
    python scripts/exportar_anotaciones.py --salida otra/ruta
    python scripts/exportar_anotaciones.py --semilla 123
"""

import json
import random
import sys
import time
from pathlib import Path

RUTA_DEFAULT = Path(__file__).resolve().parent.parent / "data" / "raw" / "annotations"
DATASET_HUB_ID = "harpreetsahota/CarDD"
NOMBRE_LOCAL = "harpreetsahota/CarDD"
SPLITS = ["train", "validation", "test"]

PROP_TRAIN = 0.70
PROP_VAL = 0.15
# PROP_TEST = 1 - PROP_TRAIN - PROP_VAL = 0.15

CATEGORIAS = ["dent", "scratch", "crack", "glass_shatter", "tire_flat", "lamp_broken"]
CAT_ID = {nombre: i + 1 for i, nombre in enumerate(CATEGORIAS)}  # COCO: IDs desde 1


def exportar(ruta_salida: Path, semilla: int):
    try:
        import fiftyone as fo
        from fiftyone.utils.huggingface import load_from_hub
    except ImportError:
        print("Error: 'fiftyone' no está instalado. Corré: pip install fiftyone")
        sys.exit(1)

    if all((ruta_salida / f"{s}.json").exists() for s in SPLITS):
        print("Las anotaciones ya existen en disco.")
        _verificar(ruta_salida)
        return

    ruta_salida.mkdir(parents=True, exist_ok=True)

    print(f"Cargando dataset CarDD desde FiftyOne Hub...")
    if fo.dataset_exists(NOMBRE_LOCAL):
        dataset = fo.load_dataset(NOMBRE_LOCAL)
        print(f"  → Dataset local encontrado ({len(dataset)} muestras)")
    else:
        dataset = load_from_hub(DATASET_HUB_ID, name=NOMBRE_LOCAL)
        print(f"  → Descargado desde Hub ({len(dataset)} muestras)")

    print("Calculando dimensiones de imágenes...")
    dataset.compute_metadata()

    # Partición reproducible con semilla fija
    ids = list(dataset.values("id"))
    random.seed(semilla)
    random.shuffle(ids)

    n = len(ids)
    n_train = int(n * PROP_TRAIN)
    n_val = int(n * PROP_VAL)
    particiones = {
        "train":      ids[:n_train],
        "validation": ids[n_train:n_train + n_val],
        "test":       ids[n_train + n_val:],
    }

    inicio = time.time()

    for nombre_split, ids_split in particiones.items():
        ruta_json = ruta_salida / f"{nombre_split}.json"
        if ruta_json.exists():
            print(f"[{nombre_split}] Ya existe, salteando.")
            continue

        vista = dataset.select(ids_split)
        coco = _convertir_a_coco(vista, nombre_split)

        with open(ruta_json, "w") as f:
            json.dump(coco, f)

        print(
            f"[{nombre_split}] {len(coco['images'])} imágenes | "
            f"{len(coco['annotations'])} anotaciones → {ruta_json}"
        )

    duracion = time.time() - inicio
    print(f"\nListo en {duracion:.1f}s")
    _verificar(ruta_salida)


def _convertir_a_coco(vista, nombre_split):
    """Convierte una vista de FiftyOne a estructura COCO JSON."""
    imagenes = []
    anotaciones = []
    ann_id = 1

    for img_id, muestra in enumerate(vista, start=1):
        ancho = muestra.metadata.width if muestra.metadata else None
        alto = muestra.metadata.height if muestra.metadata else None

        if ancho is None or alto is None:
            from PIL import Image
            with Image.open(muestra.filepath) as img:
                ancho, alto = img.size

        imagenes.append({
            "id": img_id,
            "file_name": muestra.filepath,
            "width": ancho,
            "height": alto,
        })

        if muestra.detections is None:
            continue

        for det in muestra.detections.detections:
            nombre_clase = det.label.lower().replace(" ", "_")
            if nombre_clase not in CAT_ID:
                continue

            # FiftyOne guarda bbox normalizado [x, y, w, h] ∈ [0, 1]
            x_n, y_n, w_n, h_n = det.bounding_box
            x = x_n * ancho
            y = y_n * alto
            w = w_n * ancho
            h = h_n * alto

            anotaciones.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": CAT_ID[nombre_clase],
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            ann_id += 1

    categorias = [{"id": v, "name": k} for k, v in sorted(CAT_ID.items(), key=lambda x: x[1])]
    return {
        "info": {"description": f"CarDD — split {nombre_split}"},
        "categories": categorias,
        "images": imagenes,
        "annotations": anotaciones,
    }


def _verificar(ruta: Path):
    print("\nVerificación:")
    for split in SPLITS:
        ruta_json = ruta / f"{split}.json"
        if not ruta_json.exists():
            print(f"  [{split}] FALTA")
            continue
        with open(ruta_json) as f:
            coco = json.load(f)
        print(
            f"  [{split}] {len(coco['images'])} imágenes | "
            f"{len(coco['annotations'])} anotaciones"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Exporta anotaciones CarDD desde FiftyOne a COCO JSON."
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=RUTA_DEFAULT,
        help=f"Directorio de salida para los JSONs (default: {RUTA_DEFAULT})",
    )
    parser.add_argument(
        "--semilla",
        type=int,
        default=42,
        help="Semilla para la partición aleatoria (default: 42)",
    )
    args = parser.parse_args()
    exportar(args.salida, args.semilla)
