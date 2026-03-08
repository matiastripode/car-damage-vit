"""
Genera mapas de atención para una imagen dada.

Uso:
    python scripts/visualizar_atenciones.py \\
        --modelo facebook/deit-tiny-patch16-224 \\
        --imagen data/sample.jpg \\
        --salida reports/figuras/atencion_deit.png

    # Con pesos fine-tuneados:
    python scripts/visualizar_atenciones.py \\
        --modelo apple/mobilevit-small \\
        --imagen data/sample.jpg \\
        --salida reports/figuras/atencion_mobilevit.png \\
        --checkpoint checkpoints/mobilevit_best.pt \\
        --clase 0
"""

import argparse
import sys
from pathlib import Path

# Asegurar que src/ esté en el path cuando se ejecuta desde la raíz del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from PIL import Image

from vit.eval.visualizar import visualizar_attention
from vit.models.factory import cargar_modelo


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera mapas de atención (DeiT → attention rollout, MobileViT → GradCAM)"
    )
    parser.add_argument(
        "--modelo", required=True,
        help="Nombre del modelo HuggingFace (ej. 'facebook/deit-tiny-patch16-224')"
    )
    parser.add_argument(
        "--imagen", required=True,
        help="Ruta a la imagen de entrada"
    )
    parser.add_argument(
        "--salida", required=True,
        help="Ruta de salida para la figura PNG"
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Ruta a pesos fine-tuneados (.pt); opcional"
    )
    parser.add_argument(
        "--clase", type=int, default=None,
        help="Índice de clase objetivo para GradCAM; si se omite usa argmax"
    )
    parser.add_argument(
        "--num-clases", type=int, default=7,
        help="Número de clases del modelo (default: 7)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Cargando modelo: {args.modelo}")
    # 'eager' requerido para que output_attentions funcione en DeiT/ViT (sdpa no lo soporta)
    modelo, procesador = cargar_modelo(
        args.modelo, num_clases=args.num_clases, attn_implementation="eager"
    )

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        # El trainer guarda {"epoch":..., "model_state_dict":..., "val_f1":..., "history":...}
        state = ckpt.get("model_state_dict", ckpt)
        modelo.load_state_dict(state)
        epoch = ckpt.get("epoch", "?")
        val_f1 = ckpt.get("val_f1", "?")
        print(f"Pesos cargados desde: {args.checkpoint} (epoch={epoch}, val_f1={val_f1})")

    modelo.eval()

    imagen_path = Path(args.imagen)
    if not imagen_path.exists():
        print(f"Error: imagen no encontrada en {imagen_path}", file=sys.stderr)
        sys.exit(1)

    imagen = Image.open(imagen_path).convert("RGB")
    print(f"Imagen cargada: {imagen_path} ({imagen.size[0]}×{imagen.size[1]})")

    heatmap = visualizar_attention(
        modelo, imagen, procesador, ruta_salida=args.salida
    )

    print(f"Mapa de atención guardado en: {args.salida}")
    print(f"  shape={heatmap.shape}, min={heatmap.min():.4f}, max={heatmap.max():.4f}")


if __name__ == "__main__":
    main()
