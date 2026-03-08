from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image


def _imagen_pil():
    return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))


def _procesador(con_stats=True):
    if con_stats:
        return SimpleNamespace(
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )
    return SimpleNamespace()  # sin image_mean/std → fallback ImageNet


# ── Shapes ─────────────────────────────────────────────────────────────────────

def test_transform_entrenamiento_shape():
    from vit.transforms.augmentaciones import get_transforms_entrenamiento
    t = get_transforms_entrenamiento(_procesador())
    tensor = t(_imagen_pil())
    assert tensor.shape == (3, 224, 224)
    assert isinstance(tensor, torch.Tensor)


def test_transform_evaluacion_shape():
    from vit.transforms.augmentaciones import get_transforms_evaluacion
    t = get_transforms_evaluacion(_procesador())
    tensor = t(_imagen_pil())
    assert tensor.shape == (3, 224, 224)
    assert isinstance(tensor, torch.Tensor)


# ── Fallback ImageNet ──────────────────────────────────────────────────────────

def test_fallback_sin_stats():
    """Si el procesador no expone image_mean/std, usa los defaults de ImageNet."""
    from vit.transforms.augmentaciones import get_transforms_evaluacion
    t = get_transforms_evaluacion(_procesador(con_stats=False))
    tensor = t(_imagen_pil())
    assert tensor.shape == (3, 224, 224)


# ── Augmentation en eval ───────────────────────────────────────────────────────

def test_evaluacion_no_tiene_augmentation():
    """El transform de evaluación no debe incluir augmentation aleatorio."""
    from torchvision.transforms import RandomErasing, RandomHorizontalFlip

    from vit.transforms.augmentaciones import get_transforms_evaluacion
    t = get_transforms_evaluacion(_procesador())
    tipos = [type(tr) for tr in t.transforms]
    assert RandomHorizontalFlip not in tipos
    assert RandomErasing not in tipos
