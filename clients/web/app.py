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
