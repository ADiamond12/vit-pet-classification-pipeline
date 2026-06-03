import io

import requests
import streamlit as st
from PIL import Image

DEFAULT_API_URL = "http://127.0.0.1:8000/predict"


def health_url_from(api_url: str) -> str:
    cleaned = api_url.rstrip("/")
    if cleaned.endswith("/predict"):
        return f"{cleaned.removesuffix('/predict')}/health"
    return f"{cleaned}/health"


st.set_page_config(page_title="ViT Pet Classification Pipeline", page_icon=":cat:", layout="wide")

st.markdown(
    """
    <style>
      #MainMenu,
      footer,
      header,
      [data-testid="stDeployButton"],
      [data-testid="stToolbar"],
      [data-testid="stDecoration"],
      [data-testid="stStatusWidget"],
      .stDeployButton {
        display: none !important;
      }
      .block-container {
        padding-top: 3rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("ViT Pet Classification Pipeline")
st.caption(
    "Public-safe Streamlit review UI for a cats-vs-dogs ViT delivery path. "
    "Predictions require a checkpoint-backed FastAPI service."
)

st.sidebar.header("Settings")
api_url = st.sidebar.text_input("API URL", value=DEFAULT_API_URL)
st.sidebar.info("Start FastAPI locally with: `uvicorn src.api.main:app --reload`")
st.sidebar.caption(
    "If the backend does not have a local checkpoint yet, set `VIT_PET_MODEL_REPO_ID` "
    "there to enable first-run model bootstrap."
)
if st.sidebar.button("Check backend health", use_container_width=True):
    try:
        health_response = requests.get(health_url_from(api_url), timeout=10)
        health_response.raise_for_status()
    except requests.RequestException as exc:
        st.sidebar.error(f"Backend health check failed: {exc}")
    else:
        st.sidebar.json(health_response.json())

st.info(
    "This public repo does not ship model weights or datasets. The safe screenshot shows the "
    "upload and review flow only; prediction output should be captured after using a project-owned "
    "checkpoint."
)

st.markdown(
    """
    **What this public UI proves**

    - The reviewer flow can start without private datasets or model files.
    - Backend readiness is explicit through the FastAPI health endpoint.
    - Predictions stay inactive until a checkpoint-backed service is running.
    """
)

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Image review")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Preview", use_container_width=True)
    else:
        st.caption("Upload a JPG or PNG when a local or published checkpoint-backed API is running.")

with col_right:
    st.subheader("Model response")
    st.caption("Prediction results are shown only after a checkpoint-backed API accepts the upload.")
    if uploaded_file:
        if st.button("Send to model", type="primary", use_container_width=True):
            with st.spinner("Sending to API..."):
                buf = io.BytesIO()
                image.save(buf, format="JPEG")
                buf.seek(0)

                files = {"file": ("image.jpg", buf, "image/jpeg")}
                try:
                    resp = requests.post(api_url, files=files, timeout=30)
                except requests.RequestException as exc:
                    st.error(f"Request failed: {exc}")
                else:
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(f"Prediction: {data['label']}")
                        st.metric("Confidence", f"{data['confidence']:.2%}")
                    else:
                        st.error(f"Error {resp.status_code}: {resp.text}")
    else:
        st.info("Prediction remains inactive until an image is selected and the FastAPI service is ready.")
