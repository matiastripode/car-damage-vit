"""
Visualización de mapas de atención.

- DeiT / ViT  → Attention Rollout (Abnar & Zuidema, 2020)
- MobileViT   → GradCAM (Selvaraju et al., 2017)
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # backend no interactivo
import matplotlib.pyplot as plt


# ── Attention Rollout (DeiT / ViT) ─────────────────────────────────────────


def _attention_rollout(attentions):
    """
    Calcula el attention rollout a partir de todas las capas del transformer.

    Args:
        attentions: tupla de tensores (1, num_heads, seq_len, seq_len)

    Returns:
        ndarray (num_patches,) normalizado en [0, 1] — excluye el token CLS
    """
    device = attentions[0].device
    seq_len = attentions[0].shape[-1]
    rollout = torch.eye(seq_len, device=device)

    for attn in attentions:
        # Promedio sobre cabezas de atención
        attn_mean = attn[0].mean(dim=0)                            # (seq_len, seq_len)
        # Conexión residual: A_res = 0.5 * A + 0.5 * I
        attn_res = 0.5 * attn_mean + 0.5 * torch.eye(seq_len, device=device)
        attn_res = attn_res / attn_res.sum(dim=-1, keepdim=True)   # normalizar filas
        rollout = rollout @ attn_res

    # Fila del token CLS → todos los parches (excluye el propio CLS)
    mask = rollout[0, 1:].detach().cpu().numpy().astype(np.float32)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask


def _heatmap_deit(modelo, pixel_values):
    """
    Genera el heatmap vía attention rollout para modelos DeiT / ViT.

    Returns:
        ndarray (grid, grid) float32 en [0, 1]
    """
    with torch.no_grad():
        outputs = modelo(pixel_values, output_attentions=True)

    mask = _attention_rollout(outputs.attentions)
    grid = int(np.sqrt(len(mask)))
    return mask.reshape(grid, grid)


# ── GradCAM (MobileViT) ────────────────────────────────────────────────────


def _heatmap_mobilevit(modelo, pixel_values, clase_idx=None):
    """
    Genera el heatmap vía GradCAM para MobileViT.

    Hookea la última capa del encoder para capturar activaciones y gradientes.

    Returns:
        ndarray float32 en [0, 1]
    """
    activations: dict = {}
    gradients: dict = {}

    target_layer = modelo.mobilevit.encoder.layer[-1]

    def fwd_hook(module, inp, out):
        activations["feat"] = out[0] if isinstance(out, tuple) else out

    def bwd_hook(module, grad_in, grad_out):
        gradients["feat"] = grad_out[0] if isinstance(grad_out, tuple) else grad_out

    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    try:
        modelo.zero_grad()
        outputs = modelo(pixel_values)
        logits = outputs.logits                           # (1, num_clases)

        if clase_idx is None:
            clase_idx = int(logits.argmax(dim=-1))

        logits[0, clase_idx].backward()
    finally:
        fh.remove()
        bh.remove()

    feat = activations["feat"]    # (1, C, H, W)
    grad = gradients["feat"]      # (1, C, H, W)

    # Pesos globales por canal (GAP sobre mapa espacial de gradientes)
    weights = grad.mean(dim=(2, 3), keepdim=True)         # (1, C, 1, 1)
    cam = torch.relu((weights * feat).sum(dim=1).squeeze())  # (H, W)

    cam = cam.detach().cpu().numpy().astype(np.float32)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


# ── API pública ────────────────────────────────────────────────────────────


def visualizar_attention(modelo, imagen_pil, procesador, ruta_salida=None):
    """
    Genera el mapa de atención para una imagen, despachando por tipo de modelo.

    - DeiT / ViT  → attention rollout
    - MobileViT   → GradCAM sobre la última capa del encoder

    Args:
        modelo:       modelo cargado con cargar_modelo()
        imagen_pil:   PIL.Image.Image de entrada
        procesador:   procesador / feature extractor del modelo
        ruta_salida:  Path o str para guardar la figura; None → solo retorna

    Returns:
        ndarray (H, W) float32 con el heatmap normalizado en [0, 1]
    """
    inputs = procesador(images=imagen_pil, return_tensors="pt")
    pixel_values = inputs["pixel_values"]

    model_type = modelo.config.model_type

    # DeiT carga con AutoModel como ViT → model_type puede ser "deit" o "vit"
    if model_type in ("deit", "vit"):
        heatmap = _heatmap_deit(modelo, pixel_values)
    elif model_type == "mobilevit":
        heatmap = _heatmap_mobilevit(modelo, pixel_values)
    else:
        raise ValueError(
            f"Tipo de modelo no soportado para visualización: {model_type}. "
            "Soportados: 'deit'/'vit', 'mobilevit'"
        )

    if ruta_salida is not None:
        _guardar_figura(imagen_pil, heatmap, ruta_salida, model_type)

    return heatmap


def _guardar_figura(imagen_pil, heatmap, ruta_salida, model_type):
    """Superpone el heatmap (coloreado) sobre la imagen original y guarda la figura."""
    from pathlib import Path
    from PIL import Image

    # Redimensionar heatmap al tamaño de la imagen original con PIL (sin cv2)
    w, h = imagen_pil.size
    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
        (w, h), resample=Image.BILINEAR
    )
    heatmap_resized = np.array(heatmap_pil) / 255.0

    # Colormap jet sobre el heatmap
    heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]  # (H, W, 3) RGB

    # Superposición: 50% imagen + 50% mapa de calor
    imagen_np = np.array(imagen_pil.convert("RGB")) / 255.0
    overlay = np.clip(0.5 * imagen_np + 0.5 * heatmap_colored, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(imagen_pil)
    axes[0].set_title("Imagen original")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(f"Mapa de atención ({model_type})")
    axes[1].axis("off")
    fig.tight_layout()

    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ── Matriz de confusión ────────────────────────────────────────────────────


def visualizar_matriz_confusion(cm_array, nombres_clases, ruta_salida):
    """
    Guarda la matriz de confusión como imagen en reports/figuras/.

    Args:
        cm_array:       ndarray (N, N) de enteros
        nombres_clases: lista de N strings con los nombres de clase
        ruta_salida:    Path o str destino
    """
    import seaborn as sns
    from pathlib import Path

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_array,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=nombres_clases,
        yticklabels=nombres_clases,
        ax=ax,
    )
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión")
    fig.tight_layout()

    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida, bbox_inches="tight", dpi=150)
    plt.close(fig)
