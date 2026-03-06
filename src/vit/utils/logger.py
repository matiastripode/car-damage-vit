import logging
from pathlib import Path


def get_logger(nombre: str, archivo: str = None) -> logging.Logger:
    """Devuelve un logger configurado. Si se pasa archivo, también loguea a disco."""
    logger = logging.getLogger(nombre)
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    handler_consola = logging.StreamHandler()
    handler_consola.setFormatter(fmt)
    logger.addHandler(handler_consola)

    if archivo:
        Path(archivo).parent.mkdir(parents=True, exist_ok=True)
        handler_archivo = logging.FileHandler(archivo)
        handler_archivo.setFormatter(fmt)
        logger.addHandler(handler_archivo)

    return logger
