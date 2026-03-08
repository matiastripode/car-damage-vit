from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_raiz():
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["estado"] == "ok"
    assert "version" in data


def test_predecir_mock():
    # Imagen mínima válida (1×1 PNG) — el endpoint devuelve mock por ahora
    # TODO (#10): reemplazar por imagen real cuando se integre el modelo
    imagen_dummy = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
        b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
        b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    resp = client.post(
        "/predecir",
        files={"archivo": ("test.png", imagen_dummy, "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "clase" in data
    assert "confianza" in data
    assert isinstance(data["clase"], str)
    assert isinstance(data["confianza"], float)
