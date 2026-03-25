"""
Pipeline end-to-end para:
1) preparar datos (si faltan),
2) entrenar,
3) evaluar.

Uso básico:
    python scripts/pipeline_entrenar_evaluar.py --config configs/model/mobilevit_small.yaml --env dev

Con MLflow:
    python scripts/pipeline_entrenar_evaluar.py \
      --config configs/model/mobilevit_small.yaml \
      --env dev \
      --mlflow-uri http://localhost:6000 \
      --mlflow-train-experiment car-damage-vit-train \
      --mlflow-eval-experiment car-damage-vit-eval
"""

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
ANN_DIR = DATA_RAW / "annotations"


def _run(cmd: list[str]) -> None:
    """Ejecuta un comando y corta el pipeline si falla."""
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)


def _splits_listos() -> bool:
    """
    Verifica que existan los 3 splits descargados de Hugging Face datasets.
    Comprobamos archivos mínimos para cada split guardado en disco.
    """
    required = {
        "train": ["state.json", "dataset_info.json"],
        "validation": ["state.json", "dataset_info.json"],
        "test": ["state.json", "dataset_info.json"],
    }
    for split, files in required.items():
        split_dir = DATA_RAW / split
        if not split_dir.exists():
            return False
        for name in files:
            if not (split_dir / name).exists():
                return False
    return True


def _anotaciones_listas() -> bool:
    """Verifica la existencia de los COCO JSON para train/validation/test."""
    required = ["train.json", "validation.json", "test.json"]
    return ANN_DIR.exists() and all((ANN_DIR / name).exists() for name in required)


def _cargar_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _slug_modelo(nombre: str) -> str:
    return nombre.split("/")[-1].replace("-", "_")


def _checkpoint_esperado(cfg: dict) -> Path:
    """
    Resuelve dónde debería quedar el best checkpoint según la config.
    Si no hay output_dir explícito, usa el default del script de entrenamiento.
    """
    output_dir = cfg.get("output_dir")
    if not output_dir:
        output_dir = f"checkpoints/{_slug_modelo(cfg['modelo'])}"
    return ROOT / output_dir / "best_model.pt"


def _python_has_mlflow() -> bool:
    """Detecta si el intérprete actual tiene disponible el módulo mlflow."""
    return importlib.util.find_spec("mlflow") is not None


def _ensure_mlflow_for_local_scripts(args) -> None:
    """
    Entrenar/evaluar se ejecutan con este mismo intérprete local.
    Si se pide tracking con --mlflow-uri, mlflow debe estar instalado en este venv.
    """
    if args.mlflow_uri and not _python_has_mlflow():
        raise RuntimeError(
            "Falta mlflow en el entorno actual para ejecutar entrenar/evaluar con tracking.\n"
            f"Instalalo y reintentá:\n  {sys.executable} -m pip install mlflow>=2.14.0"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Ruta al YAML de modelo")
    parser.add_argument("--env", default="dev", choices=["dev", "staging", "prod"])
    parser.add_argument("--mlflow-uri", default=None, help="URI de tracking MLflow (opcional)")
    parser.add_argument("--mlflow-train-experiment", default="car-damage-vit-train")
    parser.add_argument("--mlflow-eval-experiment", default="car-damage-vit-eval")
    parser.add_argument("--mlflow-register-name", default=None)
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    cfg = _cargar_config(config_path)

    # Paso 1: preparación de datos solo si hace falta.
    if not _splits_listos():
        print("Splits de dataset faltantes. Ejecutando descarga y partición...")
        _run([sys.executable, "scripts/descargar_dataset.py"])
    else:
        print("Splits de dataset: OK")

    if not _anotaciones_listas():
        print("Anotaciones COCO faltantes. Ejecutando exportación...")
        _run([sys.executable, "scripts/exportar_anotaciones.py"])
    else:
        print("Anotaciones: OK")

    # Paso 1.1: validación temprana para evitar fallar tarde en train/eval.
    _ensure_mlflow_for_local_scripts(args)

    # Paso 2: entrenamiento.
    train_cmd = [
        sys.executable,
        "scripts/entrenar.py",
        "--config",
        str(config_path),
        "--env",
        args.env,
    ]
    if args.mlflow_uri:
        train_cmd += [
            "--mlflow-uri",
            args.mlflow_uri,
            "--mlflow-experiment",
            args.mlflow_train_experiment,
        ]
        if args.mlflow_register_name:
            train_cmd += ["--mlflow-register-name", args.mlflow_register_name]
    _run(train_cmd)

    # Paso 3: evaluación sobre test.
    checkpoint = _checkpoint_esperado(cfg)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"No se encontró checkpoint esperado tras entrenar: {checkpoint}"
        )

    eval_cmd = [
        sys.executable,
        "scripts/evaluar.py",
        "--checkpoint",
        str(checkpoint),
        "--config",
        str(config_path),
        "--env",
        args.env,
    ]
    if args.mlflow_uri:
        eval_cmd += [
            "--mlflow-uri",
            args.mlflow_uri,
            "--mlflow-experiment",
            args.mlflow_eval_experiment,
        ]
    _run(eval_cmd)

    print("\nPipeline completado.")
