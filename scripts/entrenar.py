"""
Entry point para entrenamiento.

Uso:
    python scripts/entrenar.py --config configs/model/deit_tiny.yaml --env dev
"""
import argparse
import sys
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Ruta al archivo de configuración del modelo")
    parser.add_argument("--env", default="dev", choices=["dev", "staging", "prod"])
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

    history = entrenar(
        modelo=modelo,
        train_loader=dl_train,
        val_loader=dl_val,
        config=cfg_train,
    )

    print(f"Entrenamiento finalizado. Epochs ejecutadas: {len(history)}")
