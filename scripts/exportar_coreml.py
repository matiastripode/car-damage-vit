"""
Convierte el checkpoint MobileViT-small entrenado a CoreML (.mlpackage).

Uso:
    python scripts/exportar_coreml.py [--checkpoint PATH] [--salida PATH]

El .mlpackage resultante puede abrirse directamente en Xcode para
integración en la app iOS carDamageDemo.
"""

import argparse
import sys
from pathlib import Path

import torch

# Permite importar desde src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from vit.models.factory import cargar_modelo
from vit.data.dataset import CLASES_CON_FONDO

CHECKPOINT_DEFAULT = "checkpoints/mobilevit_small/best_model.pt"
SALIDA_DEFAULT = "checkpoints/mobilevit_small/CarDamageClassifier.mlpackage"
NOMBRE_MODELO = "apple/mobilevit-small"
NUM_CLASES = 7

# Estadísticas ImageNet (mismas que usa get_transforms_evaluacion)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def cargar_con_checkpoint(checkpoint: str):
    print(f"[1/4] Cargando modelo '{NOMBRE_MODELO}' desde HuggingFace...")
    modelo, procesador = cargar_modelo(NOMBRE_MODELO, num_clases=NUM_CLASES)

    print(f"[1/4] Cargando pesos desde '{checkpoint}'...")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    modelo.load_state_dict(ckpt["model_state_dict"])
    modelo.eval()
    print(f"      ✓ Checkpoint epoca={ckpt.get('epoch', '?')}, val_f1={ckpt.get('val_f1', '?'):.4f}")
    return modelo, procesador


class _LogitsWrapper(torch.nn.Module):
    """Envuelve el modelo HuggingFace para que forward() devuelva un tensor,
    no un dict — requisito de torch.jit.trace.

    Además bake-in la normalización ImageNet para que CoreML reciba la imagen
    en [0, 1] (tras scale=1/255 en ImageType) y este wrapper la normalice
    con mean/std antes de pasársela al modelo.
    """

    def __init__(self, modelo):
        super().__init__()
        self.modelo = modelo
        # Registrados como buffers para que trace los capture como constantes
        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values llega en [0, 1] — CoreML aplica scale=1/255 antes de aquí
        normalized = (pixel_values - self.mean) / self.std
        return self.modelo(pixel_values=normalized).logits


def trazar_modelo(modelo):
    print("[2/4] Trazando modelo con torch.jit.trace...")
    wrapper = _LogitsWrapper(modelo)
    wrapper.eval()
    ejemplo = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, ejemplo, strict=False)
    print("      ✓ Traza exitosa")
    return traced


def convertir_a_coreml(traced):
    import coremltools as ct

    print("[3/4] Convirtiendo a CoreML (.mlprogram)...")

    # El wrapper ya bake-in la normalización ImageNet (mean/std).
    # ImageType solo aplica scale=1/255 escalar → wrapper recibe [0,1].
    # bias=[0,0,0] evita el error de broadcasting per-canal en mlprogram.
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="pixel_values",
                shape=(1, 3, 224, 224),
                scale=1.0 / 255.0,
                bias=[0.0, 0.0, 0.0],
                color_layout=ct.colorlayout.RGB,
            )
        ],
        classifier_config=ct.ClassifierConfig(class_labels=list(CLASES_CON_FONDO)),
        minimum_deployment_target=ct.target.iOS16,
        convert_to="mlprogram",
    )

    # Metadatos del modelo
    mlmodel.author = "car-damage-vit"
    mlmodel.short_description = (
        "MobileViT-small fine-tuned en CarDD — clasifica daños en vehículos: "
        "dent, scratch, crack, glass_shatter, tire_flat, lamp_broken, fondo"
    )
    mlmodel.version = "1.0"

    print("      ✓ Conversión exitosa")
    return mlmodel


def guardar_y_verificar(mlmodel, salida: str, procesador):
    import coremltools as ct
    from PIL import Image

    print(f"[4/4] Guardando en '{salida}'...")
    Path(salida).parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(salida)
    print("      ✓ Guardado")

    # Verificación rápida con imagen sintética
    print("      Verificando con imagen de prueba (tensor aleatorio simulado como PIL)...")
    test_model = ct.models.MLModel(salida)
    img = Image.new("RGB", (224, 224), color=(128, 64, 32))

    result = test_model.predict({"pixel_values": img})
    clase = result["classLabel"]
    # El key del dict de probabilidades puede variar según coremltools
    probs_key = next((k for k in result if k != "classLabel"), None)
    top3 = sorted(result[probs_key].items(), key=lambda x: -x[1])[:3] if probs_key else []
    print(f"      Predicción: '{clase}'")
    print(f"      Top-3: {top3}")
    print(f"      (keys disponibles: {list(result.keys())})")
    print()
    print(f"✅  {salida}")
    print()
    print("Próximos pasos (iOS):")
    print("  1. Arrastrar CarDamageClassifier.mlpackage al grupo raíz de carDamageDemo en Xcode")
    print("  2. Target Membership → carDamageDemo ✓")
    print("  3. Agregar LocalClassifier.swift al proyecto")
    print("  4. Compilar y correr en iPhone")


def main():
    parser = argparse.ArgumentParser(description="Exporta MobileViT-small a CoreML")
    parser.add_argument("--checkpoint", default=CHECKPOINT_DEFAULT, help="Ruta al best_model.pt")
    parser.add_argument("--salida", default=SALIDA_DEFAULT, help="Ruta de salida .mlpackage")
    args = parser.parse_args()

    try:
        import coremltools 
    except ImportError:
        print("ERROR: coremltools no está instalado. Ejecutar:")
        print("  pip install 'coremltools>=7.2'")
        sys.exit(1)

    modelo, procesador = cargar_con_checkpoint(args.checkpoint)

    try:
        traced = trazar_modelo(modelo)
    except Exception as e:
        print(f"⚠️  torch.jit.trace falló: {e}")
        print("   Intentando exportar vía ONNX como fallback...")
        traced = exportar_via_onnx(modelo)

    mlmodel = convertir_a_coreml(traced)
    guardar_y_verificar(mlmodel, args.salida, procesador)


def exportar_via_onnx(modelo):
    """Fallback: exporta a ONNX y lo carga para coremltools."""
    import coremltools as ct
    import tempfile, os

    onnx_path = tempfile.mktemp(suffix=".onnx")
    ejemplo = torch.zeros(1, 3, 224, 224)

    torch.onnx.export(
        modelo,
        {"pixel_values": ejemplo},
        onnx_path,
        opset_version=14,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes=None,
    )
    print(f"      ✓ ONNX exportado a {onnx_path}")

    import onnx
    onnx_model = onnx.load(onnx_path)
    os.unlink(onnx_path)
    return onnx_model


if __name__ == "__main__":
    main()
