from torchvision import transforms

# Fallback: ImageNet mean/std (usado si el procesador no los expone)
_MEAN_DEFAULT = [0.485, 0.456, 0.406]
_STD_DEFAULT  = [0.229, 0.224, 0.225]


def _mean_std(procesador):
    mean = getattr(procesador, "image_mean", None) or _MEAN_DEFAULT
    std  = getattr(procesador, "image_std",  None) or _STD_DEFAULT
    return mean, std


def get_transforms_entrenamiento(procesador):
    """Transforms con data augmentation para el conjunto de entrenamiento."""
    mean, std = _mean_std(procesador)
    return transforms.Compose([
        # Zoom aleatorio dentro del parche 224x224: genera distintas vistas
        # del daño sin salirse del recorte ya hecho por el dataset.
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        # Borra regiones aleatorias pequeñas después de normalizar: simula
        # oclusión parcial y evita que el modelo memorice píxeles concretos.
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
    ])


def get_transforms_evaluacion(procesador):
    """Transforms sin augmentation para validación y test."""
    mean, std = _mean_std(procesador)
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
