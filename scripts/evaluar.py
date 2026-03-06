"""
Evalúa un checkpoint entrenado sobre el conjunto de test.

Uso:
    python scripts/evaluar.py --checkpoint checkpoints/mejor_modelo.pt --config configs/model/deit_tiny.yaml
"""
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Ruta al checkpoint")
    parser.add_argument("--config", required=True, help="Ruta al archivo de configuración")
    args = parser.parse_args()

    # pendiente: cargar modelo desde checkpoint y generar reporte en reports/
    pass
