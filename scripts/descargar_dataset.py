"""
Descarga el dataset CarDD desde HuggingFace y lo guarda en data/raw/.

Uso:
    python scripts/descargar_dataset.py
    python scripts/descargar_dataset.py --salida otra/ruta
    python scripts/descargar_dataset.py --solo-splits train validation
"""

import argparse
import sys
import time
from pathlib import Path

# Asegurar que src/ esté en el path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

RUTA_DEFAULT = Path(__file__).resolve().parent.parent / "data" / "raw"
DATASET_ID = "harpreetsahota/CarDD"
SPLITS = ["train", "validation", "test"]


def descargar(ruta_salida: Path, splits: list):
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' no está instalado. Corré: pip install datasets")
        sys.exit(1)

    ruta_salida.mkdir(parents=True, exist_ok=True)

    print(f"Dataset:  {DATASET_ID}")
    print(f"Destino:  {ruta_salida}")
    print(f"Splits:   {', '.join(splits)}")
    print("-" * 50)

    inicio = time.time()

    for split in splits:
        ruta_split = ruta_salida / split

        if ruta_split.exists():
            print(f"[{split}] Ya existe en disco, salteando.")
            continue

        print(f"[{split}] Descargando...")
        ds = load_dataset(DATASET_ID, split=split)
        ds.save_to_disk(str(ruta_split))
        print(f"[{split}] {len(ds)} imágenes guardadas en {ruta_split}")

    duracion = time.time() - inicio
    print("-" * 50)
    print(f"Listo en {duracion:.1f}s")
    print()
    _verificar(ruta_salida, splits)


def _verificar(ruta: Path, splits: list):
    from datasets import load_from_disk

    print("Verificación:")
    for split in splits:
        ruta_split = ruta / split
        if not ruta_split.exists():
            print(f"  [{split}] FALTA — no se encontró en {ruta_split}")
            continue
        ds = load_from_disk(str(ruta_split))
        print(f"  [{split}] {len(ds)} imágenes | columnas: {ds.column_names}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descarga el dataset CarDD desde HuggingFace.")
    parser.add_argument(
        "--salida",
        type=Path,
        default=RUTA_DEFAULT,
        help=f"Ruta de destino (default: {RUTA_DEFAULT})",
    )
    parser.add_argument(
        "--solo-splits",
        nargs="+",
        choices=SPLITS,
        default=SPLITS,
        metavar="SPLIT",
        help="Splits a descargar: train, validation, test (default: todos)",
    )
    args = parser.parse_args()
    descargar(args.salida, args.solo_splits)
