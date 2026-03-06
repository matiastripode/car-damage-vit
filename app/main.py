from fastapi import FastAPI, UploadFile, File

app = FastAPI(title="car-damage-vit", version="0.1.0")


@app.get("/")
def raiz():
    return {"estado": "ok", "version": "0.1.0"}


@app.post("/predecir")
async def predecir(archivo: UploadFile = File(...)):
    # pendiente: cargar modelo y procesar imagen
    pass
