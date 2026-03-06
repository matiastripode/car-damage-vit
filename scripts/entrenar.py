"""
Entry point para entrenamiento.

Uso:
    python scripts/entrenar.py --config configs/model/deit_tiny.yaml --env dev
"""
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Ruta al archivo de configuración del modelo")
    parser.add_argument("--env", default="dev", choices=["dev", "staging", "prod"])
    args = parser.parse_args()

    # pendiente: cargar config y llamar al trainer
    pass
