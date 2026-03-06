from transformers import AutoModelForImageClassification


MODELOS_SOPORTADOS = [
    "facebook/deit-tiny-patch16-224",
    "apple/mobilevit-small",
]


def cargar_modelo(nombre: str, num_clases: int):
    """Carga un modelo desde HuggingFace y adapta la cabeza de clasificación."""
    if nombre not in MODELOS_SOPORTADOS:
        raise ValueError(f"Modelo no soportado: {nombre}. Opciones: {MODELOS_SOPORTADOS}")

    # pendiente: cargar y devolver modelo
    pass
