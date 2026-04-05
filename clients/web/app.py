import json
import os
from io import BytesIO

import requests
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

ROI_SIZE = 224
DEFAULT_API_URL = os.getenv("API_BASE_URL", "http://api:8000")
MAX_APP_WIDTH_PX = 1024
CLASES_DISPONIBLES = ["dent", "scratch", "crack", "glass_shatter", "tire_flat", "lamp_broken", "fondo"]


st.set_page_config(page_title="Car Damage ROI Predictor", layout="wide")
st.title("Car Damage Predictor")
st.caption("Upload an image, select a 224x224 ROI, and send it to FastAPI /predecir.")
st.markdown(
    f"""
    <style>
    .main .block-container,
    div[data-testid="stMainBlockContainer"] {{
        max-width: {MAX_APP_WIDTH_PX}px;
        width: 100%;
        margin-left: auto;
        margin-right: auto;
        padding-left: 2rem;
        padding-right: 2rem;
        box-sizing: border-box;
    }}
    @media (min-width: 1200px) {{
        .main .block-container,
        div[data-testid="stMainBlockContainer"] {{
            max-width: {MAX_APP_WIDTH_PX}px !important;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

api_base_url = st.text_input("FastAPI base URL", value=DEFAULT_API_URL).rstrip("/")
endpoint = f"{api_base_url}/predecir"
feedback_endpoint = f"{api_base_url}/feedback"
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
        roi_x_key = "roi_x"
        roi_y_key = "roi_y"
        roi_file_key = "roi_file_name"
        roi_last_click_key = "roi_last_click_xy"
        pred_result_key = "pred_result"
        feedback_sent_key = "feedback_sent"
        feedback_class_key = "feedback_class"

        if st.session_state.get(roi_file_key) != archivo.name:
            st.session_state[roi_x_key] = max_x // 2
            st.session_state[roi_y_key] = max_y // 2
            st.session_state[roi_file_key] = archivo.name
            st.session_state[roi_last_click_key] = None
            st.session_state[pred_result_key] = None
            st.session_state[feedback_sent_key] = False
            st.session_state[feedback_class_key] = CLASES_DISPONIBLES[0]

        st.session_state[roi_x_key] = max(0, min(int(st.session_state.get(roi_x_key, max_x // 2)), max_x))
        st.session_state[roi_y_key] = max(0, min(int(st.session_state.get(roi_y_key, max_y // 2)), max_y))

        col_left, col_right = st.columns([4, 1])

        with col_left:
            st.subheader("1) Select ROI position")
            st.caption("Click sobre la imagen para centrar la ROI en ese punto.")
            fine_tune_container = st.container()

            x = int(st.session_state[roi_x_key])
            y = int(st.session_state[roi_y_key])

            preview = image.copy()
            draw = ImageDraw.Draw(preview)
            draw.rectangle([x, y, x + ROI_SIZE, y + ROI_SIZE], outline="red", width=4)
            click = streamlit_image_coordinates(
                preview,
                key=f"roi_click_{archivo.name}_{width}x{height}",
                use_column_width="always",
            )

            if click and "x" in click and "y" in click:
                click_x = int(click["x"])
                click_y = int(click["y"])
                current_click = (click_x, click_y)
                if st.session_state.get(roi_last_click_key) != current_click:
                    st.session_state[roi_last_click_key] = current_click
                    new_x = max(0, min(click_x - (ROI_SIZE // 2), max_x))
                    new_y = max(0, min(click_y - (ROI_SIZE // 2), max_y))
                    if new_x != x or new_y != y:
                        st.session_state[roi_x_key] = new_x
                        st.session_state[roi_y_key] = new_y
                        st.rerun()

            with fine_tune_container.expander("Ajuste fino (opcional)"):
                st.slider(
                    "X (left)",
                    min_value=0,
                    max_value=max_x,
                    key=roi_x_key,
                )
                st.slider(
                    "Y (top)",
                    min_value=0,
                    max_value=max_y,
                    key=roi_y_key,
                )

            x = int(st.session_state[roi_x_key])
            y = int(st.session_state[roi_y_key])

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
                        st.session_state[pred_result_key] = data
                        st.session_state[feedback_sent_key] = False
                        predicted_class = data.get("clase")
                        if predicted_class in CLASES_DISPONIBLES:
                            st.session_state[feedback_class_key] = predicted_class
                        else:
                            st.session_state[feedback_class_key] = CLASES_DISPONIBLES[0]
                        st.success("Prediction completed")
                    else:
                        detail = response.text
                        try:
                            detail_json = response.json()
                            detail = detail_json.get("detail", detail)
                        except Exception:
                            pass
                        st.error(f"API error {response.status_code}: {detail}")

            pred_result = st.session_state.get(pred_result_key)
            if pred_result:
                st.write(f"**Predicted class:** {pred_result.get('clase', 'N/A')}")
                st.write(f"**Confidence:** {pred_result.get('confianza', 'N/A')}")

                top3 = pred_result.get("top3", [])
                if top3:
                    st.write("**Top 3**")
                    for i, item in enumerate(top3, start=1):
                        clase = item.get("clase", "N/A")
                        confianza = item.get("confianza", "N/A")
                        st.write(f"{i}. {clase} ({confianza})")

        pred_result = st.session_state.get(pred_result_key)
        if pred_result:
            with st.expander("Raw response JSON"):
                st.code(json.dumps(pred_result, indent=2, ensure_ascii=False), language="json")

            rlhf_storage = pred_result.get("rlhf_storage") or {}
            roi_key = rlhf_storage.get("roi_key")
            if roi_key:
                st.markdown("### 3) Human feedback")
                st.caption("Compatible con iOS: envia roi_key + clase_correcta al endpoint /feedback.")
                st.selectbox(
                    "Correct class",
                    options=CLASES_DISPONIBLES,
                    key=feedback_class_key,
                )
                if st.button("Send feedback", type="secondary", key="send_feedback_btn"):
                    payload = {
                        "roi_key": roi_key,
                        "clase_correcta": st.session_state[feedback_class_key],
                    }
                    try:
                        feedback_resp = requests.post(
                            feedback_endpoint,
                            json=payload,
                            timeout=20,
                        )
                    except requests.RequestException as exc:
                        st.error(f"Could not send feedback: {exc}")
                    else:
                        if feedback_resp.ok:
                            st.session_state[feedback_sent_key] = True
                            st.success("Feedback stored in MinIO")
                        else:
                            detail = feedback_resp.text
                            try:
                                detail_json = feedback_resp.json()
                                detail = detail_json.get("detail", detail)
                            except Exception:
                                pass
                            st.error(f"Feedback API error {feedback_resp.status_code}: {detail}")

                if st.session_state.get(feedback_sent_key):
                    st.success("Correction sent")
            else:
                st.info("No roi_key returned by API; feedback cannot be sent for this prediction.")
