from torch.utils.data import Dataset


class CarDamageDataset(Dataset):
    """Dataset de parches extraídos de imágenes de daños en vehículos."""

    def __init__(self, split="train", transform=None):
        # pendiente: cargar anotaciones COCO y rutas de imágenes
        self.split = split
        self.transform = transform

    def __len__(self):
        # pendiente
        return 0

    def __getitem__(self, idx):
        # pendiente: cargar parche e imagen, aplicar transform
        pass
