import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import gdown

# ========== CONFIG ========== #
MODEL_FILE = "best.pt"
FILE_ID = "1jbiP_jZbMEWMSPnmVCL_UnW6dxWmYGeD"  # 🔁 Replace with actual file ID from Google Drive

# ========== PAGE CONFIG ========== #
st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="wide",
)

# ========== STYLE ========== #
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://t4.ftcdn.net/jpg/05/64/03/09/240_F_564030929_hWqn9j34y0a0V7CPmHkQH9d992UnjmHs.jpg");
        background-attachment: fixed;
        background-size: cover;
    }
    h1 {
        text-align: center;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1>🧠 Brain Tumor Detection App</h1>", unsafe_allow_html=True)

# ========== DOWNLOAD MODEL IF NEEDED ========== #
if not os.path.exists(MODEL_FILE):
    with st.spinner("Downloading model from Google Drive..."):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_FILE, quiet=False)
    st.success("Model downloaded successfully!")

# ========== LOAD YOLO MODEL ========== #
model = YOLO(MODEL_FILE)

# ========== UPLOAD IMAGE ========== #
uploaded_image = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    img = Image.open(uploaded_image).convert("RGB")

    # Enlarge image display size
    enlarged_img = img.resize((img.width * 2, img.height * 2))

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(enlarged_img, caption="Original Uploaded Image", use_container_width=False)

    # Run inference
    with st.spinner("Detecting tumor..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_img:
            img.save(tmp_img.name)
            results = model.predict(source=tmp_img.name, save=False)

        for r in results:
            img_res = r.plot()
            img_res_pil = Image.fromarray(img_res)
            img_res_pil = img_res_pil.resize((img.width * 2, img.height * 2))

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(img_res_pil, caption="Detection Result", use_container_width=False)

    st.success("Detection completed.")

else:
    st.info("Upload a brain MRI image to start detection.")
