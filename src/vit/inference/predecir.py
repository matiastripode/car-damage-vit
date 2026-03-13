import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from vit.data.dataset import CLASES_CON_FONDO
from vit.transforms.augmentaciones import get_transforms_evaluacion


def predecir_imagen(imagen: Image.Image, modelo, procesador, device) -> dict:
    """
    Toma una imagen PIL, aplica las transforms de evaluación y devuelve
    la clase predicha con su confianza y top-3.

    Retorna:
        {
            "clase": str,
            "confianza": float,
            "top3": [{"clase": str, "confianza": float}, ...]
        }
    """
    resize = transforms.Resize((224, 224))
    transform = get_transforms_evaluacion(procesador)
    tensor = transform(resize(imagen)).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    modelo.eval()
    with torch.no_grad():
        logits = modelo(pixel_values=tensor).logits  # [1, 7]
        probs = F.softmax(logits, dim=-1)[0]
        idx = logits.argmax(dim=-1).item()

    return {
        "clase": CLASES_CON_FONDO[idx],
        "confianza": round(probs[idx].item(), 4),
        "top3": [
            {"clase": CLASES_CON_FONDO[i], "confianza": round(probs[i].item(), 4)}
            for i in probs.argsort(descending=True)[:3].tolist()
        ],
    }
