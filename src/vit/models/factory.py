from transformers import AutoImageProcessor, AutoModelForImageClassification


MODELOS_SOPORTADOS = [
    "facebook/deit-tiny-patch16-224",
    "apple/mobilevit-small",
]


def cargar_modelo(nombre: str, num_clases: int, attn_implementation: str = None):
    """
    Carga un modelo desde HuggingFace y adapta la cabeza de clasificación.

    Args:
        nombre:               ID del modelo en HuggingFace
        num_clases:           número de clases de salida
        attn_implementation:  implementación de atención ('eager', 'sdpa', None=auto).
                              Usar 'eager' si se necesita output_attentions=True.

    Returns:
        (modelo, procesador)
    """
    if nombre not in MODELOS_SOPORTADOS:
        raise ValueError(f"Modelo no soportado: {nombre}. Opciones: {MODELOS_SOPORTADOS}")

    procesador = AutoImageProcessor.from_pretrained(nombre, use_fast=True)

    kwargs = dict(num_labels=num_clases, ignore_mismatched_sizes=True)
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation

    modelo = AutoModelForImageClassification.from_pretrained(nombre, **kwargs)
    return modelo, procesador
