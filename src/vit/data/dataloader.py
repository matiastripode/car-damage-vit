from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import CarDamageDataset


def get_dataloaders(procesadores, batch_size=32, ratio_fondo=1.0, num_workers=2):
    """
    Construye y devuelve los DataLoaders de train, validación y test.

    El DataLoader de train usa WeightedRandomSampler para compensar
    el desbalance de clases sin necesidad de duplicar muestras.

    Args:
        procesadores:  dict con claves "train" y "eval" (AutoImageProcessor)
        batch_size:    tamaño del batch
        ratio_fondo:   parches de fondo por cada parche positivo
        num_workers:   workers para la carga paralela de datos

    Returns:
        (dl_train, dl_val, dl_test)
    """
    from ..transforms.augmentaciones import get_transforms_entrenamiento, get_transforms_evaluacion

    ds_train = CarDamageDataset(
        split="train",
        transform=get_transforms_entrenamiento(procesadores["train"]),
        ratio_fondo=ratio_fondo,
    )
    ds_val = CarDamageDataset(
        split="validation",
        transform=get_transforms_evaluacion(procesadores["eval"]),
        ratio_fondo=ratio_fondo,
    )
    ds_test = CarDamageDataset(
        split="test",
        transform=get_transforms_evaluacion(procesadores["eval"]),
        ratio_fondo=ratio_fondo,
    )

    sampler = WeightedRandomSampler(
        weights=ds_train.pesos_clases(),
        num_samples=len(ds_train),
        replacement=True,
    )

    dl_train = DataLoader(ds_train, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,   num_workers=num_workers)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False,   num_workers=num_workers)

    return dl_train, dl_val, dl_test
