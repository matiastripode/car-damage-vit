"""
Descarga el dataset CarDD desde HuggingFace y lo divide en train/validation/test.

El dataset original solo tiene el split 'train'. Este script lo descarga completo
y genera las particiones automáticamente (70% train, 15% validation, 15% test)
con semilla fija para reproducibilidad.

Uso:
    python scripts/descargar_dataset.py
    python scripts/descargar_dataset.py --salida otra/ruta
    python scripts/descargar_dataset.py --semilla 123
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

# Proporciones de la partición
PROP_TRAIN = 0.70
PROP_VAL = 0.15
# test = 1 - PROP_TRAIN - PROP_VAL = 0.15


def descargar(ruta_salida: Path, semilla: int):
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' no está instalado. Corré: pip install datasets")
        sys.exit(1)

    ruta_salida.mkdir(parents=True, exist_ok=True)

    # Verificar si los tres splits ya existen
    splits_existentes = [s for s in SPLITS if (ruta_salida / s).exists()]
    if len(splits_existentes) == len(SPLITS):
        print("Los tres splits ya existen en disco.")
        print()
        _verificar(ruta_salida)
        return

    print(f"Dataset:  {DATASET_ID}")
    print(f"Destino:  {ruta_salida}")
    print(f"Semilla:  {semilla}")
    print(f"División: {int(PROP_TRAIN*100)}% train / {int(PROP_VAL*100)}% val / {int((1-PROP_TRAIN-PROP_VAL)*100)}% test")
    print("-" * 50)

    inicio = time.time()

    ruta_train = ruta_salida / "train"
    if ruta_train.exists():
        print("Split 'train' encontrado en disco, cargando sin re-descargar...")
        from datasets import load_from_disk
        ds_completo = load_from_disk(str(ruta_train))
    else:
        print("Descargando split 'train' completo desde HuggingFace...")
        ds_completo = load_dataset(DATASET_ID, split="train")
    print(f"Total de imágenes: {len(ds_completo)}")
    print()

    # Dividir en train y temp (val + test)
    prop_temp = PROP_VAL + (1 - PROP_TRAIN - PROP_VAL)  # 0.30
    split_temp = ds_completo.train_test_split(test_size=prop_temp, seed=semilla)
    ds_train = split_temp["train"]
    ds_temp = split_temp["test"]

    # Dividir temp en validation y test (50/50 de ese 30%)
    prop_test_en_temp = (1 - PROP_TRAIN - PROP_VAL) / prop_temp  # 0.5
    split_val_test = ds_temp.train_test_split(test_size=prop_test_en_temp, seed=semilla)
    ds_val = split_val_test["train"]
    ds_test = split_val_test["test"]

    splits_ds = {"train": ds_train, "validation": ds_val, "test": ds_test}

    for nombre_split, ds in splits_ds.items():
        ruta_split = ruta_salida / nombre_split
        if ruta_split.exists():
            print(f"[{nombre_split}] Ya existe en disco, salteando.")
            continue
        ds.save_to_disk(str(ruta_split))
        print(f"[{nombre_split}] {len(ds)} imágenes guardadas en {ruta_split}")

    duracion = time.time() - inicio
    print("-" * 50)
    print(f"Listo en {duracion:.1f}s")
    print()
    _verificar(ruta_salida)


def _verificar(ruta: Path):
    from datasets import load_from_disk

    print("Verificación:")
    for split in SPLITS:
        ruta_split = ruta / split
        if not ruta_split.exists():
            print(f"  [{split}] FALTA — no se encontró en {ruta_split}")
            continue
        ds = load_from_disk(str(ruta_split))
        print(f"  [{split}] {len(ds)} imágenes | columnas: {ds.column_names}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Descarga CarDD desde HuggingFace y genera splits train/validation/test."
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=RUTA_DEFAULT,
        help=f"Ruta de destino (default: {RUTA_DEFAULT})",
    )
    parser.add_argument(
        "--semilla",
        type=int,
        default=42,
        help="Semilla para la partición aleatoria (default: 42)",
    )
    args = parser.parse_args()
    descargar(args.salida, args.semilla)
