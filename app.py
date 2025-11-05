# app.py
import streamlit as st
import numpy as np
import os
import io
import pickle
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------------------------------------
# 🚨 AI-Powered Disaster Aid System
# -----------------------------------------------------------
st.set_page_config(page_title="AI-Powered Disaster Aid System", layout="wide")

# ---------------- USER CONFIG ----------------
IMAGE_TARGET_SIZE = (224, 224)
DEFAULT_TEXT_MAXLEN = 100
LABELS = ["Safe / No Disaster", "Disaster"]   # human-readable labels
# -----------------------------------------------------------

st.title("🌍 **AI-Powered Disaster Aid System**")
st.markdown(
    """
This intelligent system detects **potential disasters** using **image and text inputs**.  
It combines a **Convolutional Neural Network (CNN)** for image understanding and a **BiLSTM model** for text interpretation,  
then fuses both predictions to provide a reliable assessment of whether a situation represents a disaster.
"""
)

# Sidebar: model and fusion controls
st.sidebar.header("⚙️ Model Settings & Options")

IMAGE_MODEL_PATH = "disaster_cnn_mobilenet_clean.h5"
TEXT_MODEL_PATH = "disaster_text_bilstm.h5"
TOKENIZER_PATH = "tokenizer.pkl"

st.sidebar.write("Model file status:")
st.sidebar.write(f"- 🖼️ Image model: {'✅ Found' if os.path.exists(IMAGE_MODEL_PATH) else '❌ Missing'}")
st.sidebar.write(f"- 💬 Text model: {'✅ Found' if os.path.exists(TEXT_MODEL_PATH) else '❌ Missing'}")
st.sidebar.write(f"- 🔤 Tokenizer: {'✅ Found' if os.path.exists(TOKENIZER_PATH) else '❌ Missing'}")

uploaded_image_model = st.sidebar.file_uploader("Upload Image Model (.h5)", type=["h5"])
uploaded_text_model = st.sidebar.file_uploader("Upload Text Model (.h5)", type=["h5"])
uploaded_tokenizer = st.sidebar.file_uploader("Upload Tokenizer (.pkl)", type=["pkl", "pickle", "dat"])

st.sidebar.markdown("---")
text_maxlen = st.sidebar.number_input("Text max length", min_value=10, max_value=1000, value=DEFAULT_TEXT_MAXLEN, step=10)
st.sidebar.markdown("---")
st.sidebar.subheader("Fusion Weights")
img_weight = st.sidebar.slider("Image weight", 0.0, 1.0, 0.5, 0.05)
txt_weight = st.sidebar.slider("Text weight", 0.0, 1.0, 0.5, 0.05)

# ---------------- Model Loading ----------------
@st.cache_resource(show_spinner=False)
def load_model_from_path(path, uploaded=None):
    try:
        if uploaded is not None:
            tmp = f"temp_{os.path.basename(path)}"
            with open(tmp, "wb") as f:
                f.write(uploaded.getbuffer())
            model = load_model(tmp)
            os.remove(tmp)
            return model
        if os.path.exists(path):
            return load_model(path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
    return None

@st.cache_resource(show_spinner=False)
def load_tokenizer(path, uploaded=None):
    try:
        if uploaded is not None:
            uploaded.seek(0)
            return pickle.load(uploaded)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
    return None

image_model = load_model_from_path(IMAGE_MODEL_PATH, uploaded_image_model)
text_model = load_model_from_path(TEXT_MODEL_PATH, uploaded_text_model)
tokenizer = load_tokenizer(TOKENIZER_PATH, uploaded_tokenizer)

# ---------------- Helper functions ----------------
def model_is_binary(model):
    try:
        return int(model.output_shape[-1]) == 1
    except Exception:
        return None

def preprocess_image(img):
    img = img.resize(IMAGE_TARGET_SIZE)
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)

def image_predict(img, model):
    if model is None or img is None:
        return None
    arr = preprocess_image(img)
    pred = model.predict(arr)
    pred = np.array(pred).squeeze()
    if pred.size == 1:
        p = float(pred)
        return np.array([1 - p, p])
    probs = pred / np.sum(pred)
    return probs

def text_predict(text, model, tok, maxlen):
    if not text or model is None or tok is None:
        return None
    seq = tok.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
    pred = model.predict(pad)
    pred = np.array(pred).squeeze()
    if pred.size == 1:
        p = float(pred)
        return np.array([1 - p, p])
    probs = pred / np.sum(pred)
    return probs

def probs_to_label(probs):
    idx = int(np.argmax(probs))
    return LABELS[idx], float(probs[idx])

# ---------------- Input UI ----------------
col1, col2 = st.columns([1, 1])
with col1:
    st.header("🖼️ Image Input")
    img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    cam_input = st.camera_input("Or use camera")
    image_data = img_file or cam_input
    if image_data:
        image = Image.open(image_data).convert("RGB")
        st.image(image, caption="Input Image", use_column_width=True)
    else:
        image = None

with col2:
    st.header("💬 Text Input")
    user_text = st.text_area("Enter eyewitness report / tweet / message", height=200)

# ---------------- Prediction ----------------
st.markdown("---")
st.subheader("🔍 Run Disaster Detection")

if st.button("Analyze Situation"):
    with st.spinner("Analyzing using AI models..."):
        img_probs = image_predict(image, image_model) if image is not None else None
        txt_probs = text_predict(user_text, text_model, tokenizer, text_maxlen) if user_text else None

        # Normalize fusion weights
        total = img_weight + txt_weight
        w_img, w_txt = (img_weight / total, txt_weight / total) if total > 0 else (0.5, 0.5)

        # Fusion
        fused_probs = None
        if img_probs is not None and txt_probs is not None:
            min_len = min(len(img_probs), len(txt_probs))
            img_probs = img_probs[:min_len]
            txt_probs = txt_probs[:min_len]
            fused_probs = w_img * img_probs + w_txt * txt_probs
        elif img_probs is not None:
            fused_probs = img_probs
            st.info("Only image input provided — fused output equals image model output.")
        elif txt_probs is not None:
            fused_probs = txt_probs
            st.info("Only text input provided — fused output equals text model output.")

        if fused_probs is not None:
            fused_label, confidence = probs_to_label(fused_probs)
            st.success(f"✅ **Final Fused Prediction:** {fused_label}")
            st.write(f"**Confidence:** {confidence:.3f}")
            st.write("**Class Probabilities:**", np.round(fused_probs, 4))

            st.markdown("**Fusion Breakdown**")
            st.write(f"- Image Weight: {w_img:.2f}")
            st.write(f"- Text Weight: {w_txt:.2f}")
            if img_probs is not None:
                st.write("📷 Image Probabilities:", np.round(img_probs, 4))
            if txt_probs is not None:
                st.write("📝 Text Probabilities:", np.round(txt_probs, 4))
        else:
            st.warning("No valid input detected — please upload an image or enter text.")

# ---------------- Footer / Description ----------------
st.markdown("---")
st.header("📡 About the System")
st.markdown
)
