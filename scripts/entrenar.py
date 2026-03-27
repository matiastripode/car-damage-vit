"""
Entry point para entrenamiento.

Uso:
    python scripts/entrenar.py --config configs/model/deit_tiny.yaml --env dev
"""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch

# Asegurar que src/ esté en el path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vit.data.dataloader import get_dataloaders
from vit.models.factory import cargar_modelo
from vit.train.trainer import entrenar
from vit.utils.config import cargar_config


def _resolver_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _slug_modelo(nombre: str) -> str:
    return nombre.split("/")[-1].replace("-", "_")


def _cargar_componentes(cfg: dict):
    modelo, procesador = cargar_modelo(cfg["modelo"], num_clases=cfg["num_clases"])
    procesadores = {"train": procesador, "eval": procesador}
    return modelo, procesadores


def _construir_dataloaders(cfg: dict, env: str):
    workers_default = {"dev": 0, "staging": 2, "prod": 4}
    num_workers = cfg.get("num_workers", workers_default[env])
    ratio_fondo = cfg.get("ratio_fondo", 1.0)
    return get_dataloaders(
        procesadores=cfg["procesadores"],
        batch_size=cfg.get("batch_size", 32),
        ratio_fondo=ratio_fondo,
        num_workers=num_workers,
    )


def _armar_config_entrenamiento(cfg: dict) -> dict:
    output_dir = cfg.get("output_dir")
    if not output_dir:
        output_dir = f"checkpoints/{_slug_modelo(cfg['modelo'])}"

    return {
        "device": cfg.get("device", _resolver_device()),
        "lr": cfg.get("lr", 2e-4),
        "epochs": cfg.get("epochs", 20),
        "patience": cfg.get("patience", 5),
        "output_dir": output_dir,
    }


def _log_params_mlflow(cfg: dict, cfg_train: dict, env: str):
    import mlflow

    params = {
        "env": env,
        "modelo": cfg.get("modelo"),
        "num_clases": cfg.get("num_clases"),
        "batch_size": cfg.get("batch_size"),
        "epochs": cfg_train.get("epochs"),
        "lr": cfg_train.get("lr"),
        "patience": cfg_train.get("patience"),
        "seed": cfg.get("seed"),
        "ratio_fondo": cfg.get("ratio_fondo", 1.0),
        "device": cfg_train.get("device"),
        "output_dir": cfg_train.get("output_dir"),
    }
    for k, v in params.items():
        if v is not None:
            mlflow.log_param(k, v)


def _safe_git_commit() -> str:
    """Devuelve el commit actual si hay repo git disponible; si no, 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _dataset_fingerprint() -> str:
    """
    Fingerprint liviano del dataset local.
    Usa archivos de metadata/anotaciones para versionar corridas sin hashear imágenes.
    """
    candidates = [
        Path("data/raw/train/dataset_info.json"),
        Path("data/raw/annotations/train.json"),
    ]
    hasher = hashlib.sha256()
    used = 0
    for p in candidates:
        if p.exists():
            hasher.update(p.read_bytes())
            used += 1
    if used == 0:
        return "unknown"
    return hasher.hexdigest()[:16]


def _count_files_and_bytes(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    total_files = 0
    total_bytes = 0
    for p in path.rglob("*"):
        if p.is_file():
            total_files += 1
            total_bytes += p.stat().st_size
    return total_files, total_bytes


def _count_images(path: Path) -> int:
    if not path.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def _build_dataset_rows(data_root: Path, ann_root: Path) -> list[dict]:
    rows = []
    for split in ("train", "validation", "test"):
        split_dir = data_root / split
        ann_file = ann_root / f"{split}.json"
        files_count, bytes_count = _count_files_and_bytes(split_dir)
        image_count = _count_images(split_dir)

        ann_images = None
        ann_annotations = None
        if ann_file.exists():
            try:
                ann = json.loads(ann_file.read_text())
                ann_images = len(ann.get("images", []))
                ann_annotations = len(ann.get("annotations", []))
            except Exception:
                ann_images = None
                ann_annotations = None

        rows.append(
            {
                "split": split,
                "split_path": str(split_dir.resolve()),
                "files_count": files_count,
                "bytes_count": bytes_count,
                "image_files_count": image_count,
                "annotations_path": str(ann_file.resolve()),
                "annotations_exists": ann_file.exists(),
                "ann_images_count": ann_images,
                "ann_annotations_count": ann_annotations,
            }
        )
    return rows


def _log_dataset_to_mlflow(dataset_fingerprint: str) -> None:
    import mlflow
    import pandas as pd

    data_root = Path("data/raw")
    ann_root = data_root / "annotations"
    if not data_root.exists():
        raise FileNotFoundError(f"No existe la ruta de dataset: {data_root}")

    rows = _build_dataset_rows(data_root, ann_root)
    df = pd.DataFrame(rows)

    ds = mlflow.data.from_pandas(
        df,
        source=str(data_root.resolve()),
        name="cardd_raw_dataset",
        digest=dataset_fingerprint,
    )
    mlflow.log_input(ds, context="training")

    if ann_root.exists():
        mlflow.log_artifacts(str(ann_root), artifact_path="dataset/annotations")

    for split in ("train", "validation", "test"):
        split_dir = data_root / split
        if split_dir.exists():
            mlflow.log_artifacts(str(split_dir), artifact_path=f"dataset/raw/{split}")


def _find_registered_model_version(model_name: str, run_id: str):
    """
    Busca la versión de Model Registry creada por este run.
    Retorna string con número de versión o None.
    """
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        candidates = [v for v in versions if v.run_id == run_id]
        if not candidates:
            return None
        newest = max(candidates, key=lambda v: int(v.version))
        return str(newest.version)
    except Exception:
        return None


def _ensure_mlflow_available() -> None:
    """Valida disponibilidad de mlflow para evitar fallar tarde durante el run."""
    if importlib.util.find_spec("mlflow") is None:
        raise RuntimeError(
            "Se pasó --mlflow-uri pero mlflow no está instalado en este entorno.\n"
            f"Instalalo con: {sys.executable} -m pip install mlflow>=2.14.0"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Ruta al archivo de configuración del modelo")
    parser.add_argument("--env", default="dev", choices=["dev", "staging", "prod"])
    parser.add_argument("--mlflow-uri", default=None, help="URI de tracking de MLflow (opcional)")
    parser.add_argument("--mlflow-experiment", default="car-damage-vit-train", help="Nombre del experimento en MLflow")
    parser.add_argument("--mlflow-register-name", default=None, help="Nombre para registrar el modelo en MLflow Model Registry")
    args = parser.parse_args()

    cfg = cargar_config(args.config)

    semilla = cfg.get("seed", 42)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)

    modelo, procesadores = _cargar_componentes(cfg)
    cfg["procesadores"] = procesadores
    dl_train, dl_val, _ = _construir_dataloaders(cfg, args.env)
    cfg_train = _armar_config_entrenamiento(cfg)

    print(f"Entrenando modelo: {cfg['modelo']}")
    print(f"Entorno: {args.env} | device: {cfg_train['device']}")
    print(f"Output: {cfg_train['output_dir']}")

    output_dir = Path(cfg_train["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "best_model.pt"
    history_path = output_dir / "history.json"

    epoch_callback = None
    run_ctx = nullcontext()
    mlflow_enabled = args.mlflow_uri is not None

    if mlflow_enabled:
        _ensure_mlflow_available()
        import mlflow

        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        run_name = f"train-{_slug_modelo(cfg['modelo'])}-{args.env}"
        run_ctx = mlflow.start_run(run_name=run_name, log_system_metrics=True)

        def _cb(m):
            mlflow.log_metrics(
                {
                    "train_loss": m["train_loss"],
                    "val_loss": m["val_loss"],
                    "val_f1": m["val_f1"],
                    "val_acc": m["val_acc"],
                },
                step=m["epoch"],
            )

        epoch_callback = _cb

    with run_ctx:
        if mlflow_enabled:
            import mlflow
            dataset_fp = _dataset_fingerprint()

            # Tags para trazabilidad entre código, dataset y entorno de ejecución.
            mlflow.set_tags(
                {
                    "stage": "train",
                    "git_commit": _safe_git_commit(),
                    "dataset_fingerprint": dataset_fp,
                    "run_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            _log_dataset_to_mlflow(dataset_fp)
            _log_params_mlflow(cfg, cfg_train, args.env)

        history = entrenar(
            modelo=modelo,
            train_loader=dl_train,
            val_loader=dl_val,
            config=cfg_train,
            epoch_callback=epoch_callback,
        )

        history_path.write_text(json.dumps(history, indent=2))

        if mlflow_enabled:
            import mlflow

            # Se guarda el checkpoint generado por early-stopping como artifact.
            if ckpt_path.exists():
                mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints")
            mlflow.log_artifact(str(history_path), artifact_path="reports")

            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location="cpu")
                modelo_best, _ = cargar_modelo(cfg["modelo"], num_clases=cfg["num_clases"])
                modelo_best.load_state_dict(ckpt["model_state_dict"])
                if args.mlflow_register_name:
                    mlflow.pytorch.log_model(
                        modelo_best,
                        artifact_path="model",
                        registered_model_name=args.mlflow_register_name,
                    )
                    # Guarda en tags el nombre/versión registrada para trazabilidad.
                    run_id = mlflow.active_run().info.run_id
                    model_version = _find_registered_model_version(args.mlflow_register_name, run_id)
                    mlflow.set_tag("model_registry_name", args.mlflow_register_name)
                    if model_version is not None:
                        mlflow.set_tag("model_version", model_version)
                else:
                    mlflow.pytorch.log_model(modelo_best, artifact_path="model")

    print(f"Entrenamiento finalizado. Epochs ejecutadas: {len(history)}")
