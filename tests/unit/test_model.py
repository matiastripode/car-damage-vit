import os

import pytest
import torch


# ── Sin descarga ────────────────────────────────────────────────────────────────

def test_modelo_no_soportado():
    from vit.models.factory import cargar_modelo
    with pytest.raises(ValueError, match="Modelo no soportado"):
        cargar_modelo("modelo/inexistente", num_clases=7)


# ── Forward pass (se omiten en CI para evitar descarga de pesos) ───────────────

@pytest.mark.skipif(os.getenv("CI") == "true", reason="omite descarga de pesos en CI")
def test_forward_pass_deit():
    from vit.models.factory import cargar_modelo
    modelo, _ = cargar_modelo("facebook/deit-tiny-patch16-224", num_clases=7)
    modelo.eval()
    with torch.no_grad():
        salida = modelo(torch.randn(2, 3, 224, 224))
    assert salida.logits.shape == (2, 7)


@pytest.mark.skipif(os.getenv("CI") == "true", reason="omite descarga de pesos en CI")
def test_forward_pass_mobilevit():
    from vit.models.factory import cargar_modelo
    modelo, _ = cargar_modelo("apple/mobilevit-small", num_clases=7)
    modelo.eval()
    with torch.no_grad():
        salida = modelo(torch.randn(2, 3, 224, 224))
    assert salida.logits.shape == (2, 7)
