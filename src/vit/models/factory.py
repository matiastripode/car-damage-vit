from transformers import AutoImageProcessor, AutoModelForImageClassification


MODELOS_SOPORTADOS = [
    "facebook/deit-tiny-patch16-224",
    "apple/mobilevit-small",
]


def cargar_modelo(nombre: str, num_clases: int):
    """
    Carga un modelo desde HuggingFace y adapta la cabeza de clasificación.

    Returns:
        (modelo, procesador)
    """
    if nombre not in MODELOS_SOPORTADOS:
        raise ValueError(f"Modelo no soportado: {nombre}. Opciones: {MODELOS_SOPORTADOS}")

    procesador = AutoImageProcessor.from_pretrained(nombre, use_fast=True)
    modelo = AutoModelForImageClassification.from_pretrained(
        nombre,
        num_labels=num_clases,
        ignore_mismatched_sizes=True,
    )
    return modelo, procesador
