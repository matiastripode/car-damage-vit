import yaml
from pathlib import Path


def cargar_config(ruta: str) -> dict:
    """Carga un archivo YAML de configuración."""
    with open(Path(ruta), "r") as f:
        return yaml.safe_load(f)
