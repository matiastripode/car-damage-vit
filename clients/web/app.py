import json
import os
from io import BytesIO

import requests
import streamlit as st
from PIL import Image, ImageDraw

ROI_SIZE = 224
DEFAULT_API_URL = os.getenv("API_BASE_URL", "http://api:8000")


st.set_page_config(page_title="Car Damage ROI Predictor", layout="wide")
st.title("Car Damage Predictor")
st.caption("Upload an image, select a 224x224 ROI, and send it to FastAPI /predecir.")

api_base_url = st.text_input("FastAPI base URL", value=DEFAULT_API_URL).rstrip("/")
endpoint = f"{api_base_url}/predecir"
status_endpoint = f"{api_base_url}/"
reload_endpoint = f"{api_base_url}/modelo/recargar"

st.subheader("Model status")
col_status, col_reload = st.columns([3, 1])

with col_status:
    try:
        status_resp = requests.get(status_endpoint, timeout=10)
        if status_resp.ok:
            status = status_resp.json()
            st.write(f"**Modelo base API:** {status.get('modelo', 'N/A')}")
            st.write(f"**Origen cargado:** {status.get('model_source', 'N/A')}")
            if status.get("model_name"):
                st.write(f"**Model Registry:** {status.get('model_name')} v{status.get('model_version', 'N/A')} ({status.get('model_stage', 'none')})")
            if status.get("model_uri"):
                st.write(f"**Model URI:** {status.get('model_uri')}")
            if status.get("checkpoint"):
                st.write(f"**Checkpoint local:** {status.get('checkpoint')}")
            if status.get("loaded_at"):
                st.write(f"**Loaded at (UTC):** {status.get('loaded_at')}")
        else:
            st.warning(f"No se pudo leer estado de API ({status_resp.status_code}).")
    except requests.RequestException as exc:
        st.warning(f"No se pudo conectar para estado de modelo: {exc}")

with col_reload:
    if st.button("Cargar ultima desde MLflow", type="secondary"):
        try:
            reload_resp = requests.post(reload_endpoint, timeout=60)
        except requests.RequestException as exc:
            st.error(f"No se pudo recargar modelo: {exc}")
        else:
            if reload_resp.ok:
                data = reload_resp.json()
                if data.get("model_source") == "mlflow_registry":
                    st.success(
                        f"Modelo recargado desde MLflow: {data.get('model_name')} v{data.get('model_version', 'N/A')}"
                    )
                else:
                    st.warning(
                        "No se pudo cargar desde MLflow. Se aplico fallback a checkpoint local."
                    )
            else:
                detail = reload_resp.text
                try:
                    detail_json = reload_resp.json()
                    detail = detail_json.get("detail", detail)
                except Exception:
                    pass
                st.error(f"Error recargando modelo ({reload_resp.status_code}): {detail}")

archivo = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"])

if archivo is not None:
    image = Image.open(archivo).convert("RGB")
    width, height = image.size

    st.write(f"Image size: **{width} x {height}**")

    if width < ROI_SIZE or height < ROI_SIZE:
        st.error(
            f"Image must be at least {ROI_SIZE}x{ROI_SIZE} pixels. "
            f"Current image is {width}x{height}."
        )
    else:
        max_x = width - ROI_SIZE
        max_y = height - ROI_SIZE

        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.subheader("1) Select ROI position")
            x = st.slider("X (left)", min_value=0, max_value=max_x, value=max_x // 2)
            y = st.slider("Y (top)", min_value=0, max_value=max_y, value=max_y // 2)

            preview = image.copy()
            draw = ImageDraw.Draw(preview)
            draw.rectangle([x, y, x + ROI_SIZE, y + ROI_SIZE], outline="red", width=4)
            st.image(preview, caption="Selected 224x224 area", use_container_width=True)

        with col_right:
            st.subheader("2) ROI preview")
            roi = image.crop((x, y, x + ROI_SIZE, y + ROI_SIZE))
            st.image(roi, caption=f"ROI {ROI_SIZE}x{ROI_SIZE}", use_container_width=True)

            if st.button("Predict ROI", type="primary"):
                img_bytes = BytesIO()
                roi.save(img_bytes, format="PNG")
                img_bytes.seek(0)

                try:
                    response = requests.post(
                        endpoint,
                        files={"archivo": ("roi.png", img_bytes, "image/png")},
                        timeout=30,
                    )
                except requests.RequestException as exc:
                    st.error(f"Could not connect to API: {exc}")
                else:
                    if response.ok:
                        data = response.json()
                        st.success("Prediction completed")

                        st.write(f"**Predicted class:** {data.get('clase', 'N/A')}")
                        st.write(f"**Confidence:** {data.get('confianza', 'N/A')}")

                        top3 = data.get("top3", [])
                        if top3:
                            st.write("**Top 3**")
                            for i, item in enumerate(top3, start=1):
                                clase = item.get("clase", "N/A")
                                confianza = item.get("confianza", "N/A")
                                st.write(f"{i}. {clase} ({confianza})")

                        with st.expander("Raw response JSON"):
                            st.code(json.dumps(data, indent=2, ensure_ascii=False), language="json")
                    else:
                        detail = response.text
                        try:
                            detail_json = response.json()
                            detail = detail_json.get("detail", detail)
                        except Exception:
                            pass
                        st.error(f"API error {response.status_code}: {detail}")
