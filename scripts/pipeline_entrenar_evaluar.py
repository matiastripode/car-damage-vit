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
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
ANN_DIR = DATA_RAW / "annotations"
DATASET_VERSION_FILE = DATA_RAW / ".dataset_version.json"


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


def _dataset_fingerprint() -> str:
    """
    Fingerprint del dataset/anotaciones para detectar cambios de versión.
    Se basa en archivos de metadata y COCO JSON.
    """
    files = [
        DATA_RAW / "train" / "dataset_info.json",
        DATA_RAW / "validation" / "dataset_info.json",
        DATA_RAW / "test" / "dataset_info.json",
        ANN_DIR / "train.json",
        ANN_DIR / "validation.json",
        ANN_DIR / "test.json",
    ]
    h = hashlib.sha256()
    used = 0
    for p in files:
        if p.exists():
            h.update(p.read_bytes())
            used += 1
    if used == 0:
        return "unknown"
    return h.hexdigest()[:16]


def _dataset_changed(current: str) -> bool:
    """Compara fingerprint actual vs última versión registrada localmente."""
    if not DATASET_VERSION_FILE.exists():
        return True
    try:
        old = json.loads(DATASET_VERSION_FILE.read_text()).get("dataset_version")
    except Exception:
        return True
    return old != current


def _save_dataset_version(version: str) -> None:
    DATASET_VERSION_FILE.write_text(json.dumps({"dataset_version": version}, indent=2))


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


def _versionar_dataset_en_mlflow(args, dataset_version: str) -> None:
    """
    Ejecuta el bootstrap de versionado de dataset.
    - Si mlflow está instalado en este entorno: corre local.
    - Si no: intenta correr dentro del servicio docker `mlflow`.
    """
    env = dict(os.environ)
    env["MLFLOW_TRACKING_URI"] = args.mlflow_uri
    env["MLFLOW_DATASET_EXPERIMENT"] = args.mlflow_dataset_experiment
    env["DATASET_VERSION"] = dataset_version

    if _python_has_mlflow():
        subprocess.run(
            [sys.executable, "data/mlflow_bootstrap.py"],
            check=True,
            cwd=ROOT,
            env=env,
        )
        return

    print("mlflow no está instalado en este entorno. Intentando bootstrap en contenedor mlflow...")
    docker_cmd = [
        "docker",
        "compose",
        "exec",
        "-T",
        "mlflow",
        "sh",
        "-lc",
        (
            f"MLFLOW_TRACKING_URI='{args.mlflow_uri}' "
            f"MLFLOW_DATASET_EXPERIMENT='{args.mlflow_dataset_experiment}' "
            f"DATASET_VERSION='{dataset_version}' "
            "python /data/mlflow_bootstrap.py"
        ),
    ]
    subprocess.run(docker_cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Ruta al YAML de modelo")
    parser.add_argument("--env", default="dev", choices=["dev", "staging", "prod"])
    parser.add_argument("--mlflow-uri", default=None, help="URI de tracking MLflow (opcional)")
    parser.add_argument("--mlflow-train-experiment", default="car-damage-vit-train")
    parser.add_argument("--mlflow-eval-experiment", default="car-damage-vit-eval")
    parser.add_argument("--mlflow-dataset-experiment", default="cardd_dataset_inspection")
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

    # Paso 1.1: versionado de dataset en MLflow cuando hay cambios.
    dataset_version = _dataset_fingerprint()
    changed = _dataset_changed(dataset_version)
    if changed:
        if not args.mlflow_uri:
            raise RuntimeError(
                "El dataset cambió y debe versionarse en MLflow antes de entrenar. "
                "Pasá --mlflow-uri para continuar."
            )
        print(f"Dataset cambiado. Versionando en MLflow (version={dataset_version})...")
        _versionar_dataset_en_mlflow(args, dataset_version)
        _save_dataset_version(dataset_version)
    else:
        print(f"Dataset sin cambios (version={dataset_version}).")

    # Paso 1.2: validación temprana para evitar fallar tarde en train/eval.
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
