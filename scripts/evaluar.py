"""
Evalúa un checkpoint entrenado sobre el conjunto de test.

Uso:
    python scripts/evaluar.py --checkpoint checkpoints/mobilevit_small/best_model.pt --config configs/model/mobilevit_small.yaml
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
from vit.data.dataset import CLASES_CON_FONDO
from vit.eval.metricas import calcular_metricas
from vit.eval.visualizar import visualizar_matriz_confusion
from vit.models.factory import cargar_modelo
from vit.utils.config import cargar_config


def _resolver_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _slug_modelo(nombre: str) -> str:
    return nombre.split("/")[-1].replace("-", "_")


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
        Path("data/raw/test/dataset_info.json"),
        Path("data/raw/annotations/test.json"),
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


def _ensure_mlflow_available() -> None:
    """Valida disponibilidad de mlflow para evitar fallar tarde durante el run."""
    if importlib.util.find_spec("mlflow") is None:
        raise RuntimeError(
            "Se pasó --mlflow-uri pero mlflow no está instalado en este entorno.\n"
            f"Instalalo con: {sys.executable} -m pip install mlflow>=2.14.0"
        )


def _validate_experiment_artifact_store(experiment_name: str, tracking_uri: str) -> None:
    """
    En tracking remoto, evita experimentos con artifact_location local inválido.
    """
    if not tracking_uri.startswith(("http://", "https://")):
        return

    import mlflow

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return

    loc = (exp.artifact_location or "").strip()
    if loc.startswith("/") or loc.startswith("file:"):
        raise RuntimeError(
            "El experimento de MLflow tiene artifact_location local "
            f"({loc}) y desde este host no se puede escribir ahí.\n"
            "Solución: recrear el experimento tras levantar MLflow con --serve-artifacts, "
            "o usar un nuevo nombre de experimento."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Ruta al checkpoint")
    parser.add_argument("--config", required=True, help="Ruta al archivo de configuración")
    parser.add_argument("--env", default="dev", choices=["dev", "staging", "prod"])
    parser.add_argument("--mlflow-uri", default=None, help="URI de tracking de MLflow (opcional)")
    parser.add_argument("--mlflow-experiment", default="car-damage-vit-eval", help="Nombre del experimento en MLflow")
    parser.add_argument("--train-run-id", default=None, help="run_id de entrenamiento asociado (opcional)")
    args = parser.parse_args()

    cfg = cargar_config(args.config)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"No se encontró checkpoint: {checkpoint}")

    device = _resolver_device()
    modelo, procesador = cargar_modelo(cfg["modelo"], num_clases=cfg["num_clases"])
    ckpt = torch.load(checkpoint, map_location="cpu")
    modelo.load_state_dict(ckpt["model_state_dict"])
    modelo.to(device).eval()

    workers_default = {"dev": 0, "staging": 2, "prod": 4}
    num_workers = cfg.get("num_workers", workers_default[args.env])
    _, _, dl_test = get_dataloaders(
        procesadores={"train": procesador, "eval": procesador},
        batch_size=cfg.get("batch_size", 32),
        ratio_fondo=cfg.get("ratio_fondo", 1.0),
        num_workers=num_workers,
    )

    y_true, y_pred = [], []
    with torch.no_grad():
        for pixel_values, labels in dl_test:
            pixel_values = pixel_values.to(device)
            out = modelo(pixel_values=pixel_values)
            pred = out.logits.argmax(dim=-1).cpu().tolist()
            y_pred.extend(pred)
            y_true.extend(labels.tolist())

    metricas = calcular_metricas(y_true, y_pred, CLASES_CON_FONDO)

    reports_dir = Path("reports") / "eval" / _slug_modelo(cfg["modelo"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    metricas_path = reports_dir / "metricas_test.json"
    cm_path = reports_dir / "confusion_matrix_test.png"

    metricas_serializables = {
        "accuracy": metricas["accuracy"],
        "f1_macro": metricas["f1_macro"],
        "f1_por_clase": metricas["f1_por_clase"],
        "confusion_matrix": metricas["confusion_matrix"].tolist(),
        "checkpoint": str(checkpoint),
    }
    metricas_path.write_text(json.dumps(metricas_serializables, indent=2))
    visualizar_matriz_confusion(metricas["confusion_matrix"], CLASES_CON_FONDO, cm_path)

    print(f"Test accuracy: {metricas['accuracy']:.4f}")
    print(f"Test f1_macro: {metricas['f1_macro']:.4f}")
    print(f"Reporte guardado en: {metricas_path}")
    print(f"Matriz de confusión en: {cm_path}")

    mlflow_enabled = args.mlflow_uri is not None
    run_ctx = nullcontext()
    if mlflow_enabled:
        _ensure_mlflow_available()
        import mlflow

        mlflow.set_tracking_uri(args.mlflow_uri)
        _validate_experiment_artifact_store(args.mlflow_experiment, args.mlflow_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        run_name = f"eval-{_slug_modelo(cfg['modelo'])}-{args.env}"
        run_ctx = mlflow.start_run(run_name=run_name, log_system_metrics=True)

    with run_ctx:
        if mlflow_enabled:
            import mlflow

            # Tags para trazabilidad y para vincular esta evaluación a un run de train.
            tags = {
                "stage": "eval",
                "git_commit": _safe_git_commit(),
                "dataset_fingerprint": _dataset_fingerprint(),
                "run_utc": datetime.now(timezone.utc).isoformat(),
            }
            if args.train_run_id:
                tags["train_run_id"] = args.train_run_id
            mlflow.set_tags(tags)

            mlflow.log_param("modelo", cfg.get("modelo"))
            mlflow.log_param("num_clases", cfg.get("num_clases"))
            mlflow.log_param("batch_size", cfg.get("batch_size"))
            mlflow.log_param("env", args.env)
            mlflow.log_param("checkpoint", str(checkpoint))
            mlflow.log_metric("test_accuracy", metricas["accuracy"])
            mlflow.log_metric("test_f1_macro", metricas["f1_macro"])
            for clase, f1 in metricas["f1_por_clase"].items():
                mlflow.log_metric(f"test_f1_{clase}", float(f1))

            mlflow.log_artifact(str(metricas_path), artifact_path="reports")
            mlflow.log_artifact(str(cm_path), artifact_path="reports")
