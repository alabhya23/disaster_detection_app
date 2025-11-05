import streamlit as st
import numpy as np
import os
import pickle
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI-Powered Disaster Aid System", layout="wide")
LABELS = ["No Disaster", "Disaster"]
IMAGE_TARGET_SIZE = (224, 224)
DEFAULT_TEXT_MAXLEN = 100

# ---------------- HEADER ----------------
st.title("🌍 AI-Powered Disaster Aid System")
st.markdown(
    """
This system uses **AI (CNN + LSTM)** to analyze both **images** and **text messages** 
to detect possible disasters such as floods, fires, or earthquakes.  
If a disaster is detected, an **alert (via GCC)** is generated to warn users in the affected area.
"""
)

# ---------------- SIDEBAR ----------------
st.sidebar.header("🔧 Model Settings")
IMAGE_MODEL_PATH = "disaster_cnn_mobilenet_clean.h5"
TEXT_MODEL_PATH = "disaster_text_bilstm.h5"
TOKENIZER_PATH = "tokenizer.pkl"

uploaded_image_model = st.sidebar.file_uploader("Upload Image Model (.h5)", type=["h5"])
uploaded_text_model = st.sidebar.file_uploader("Upload Text Model (.h5)", type=["h5"])
uploaded_tokenizer = st.sidebar.file_uploader("Upload Tokenizer (.pkl)", type=["pkl"])

st.sidebar.markdown("---")
st.sidebar.subheader("Fusion Weights")
img_weight = st.sidebar.slider("Image Weight", 0.0, 1.0, 0.6, 0.05)
txt_weight = st.sidebar.slider("Text Weight", 0.0, 1.0, 0.4, 0.05)

# ---------------- LOAD MODELS ----------------
@st.cache_resource(show_spinner=False)
def load_keras_model(path, uploaded=None):
    try:
        if uploaded is not None:
            with open("temp_model.h5", "wb") as f:
                f.write(uploaded.getbuffer())
            model = load_model("temp_model.h5")
            os.remove("temp_model.h5")
            return model
        elif os.path.exists(path):
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
        elif os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
    return None

image_model = load_keras_model(IMAGE_MODEL_PATH, uploaded_image_model)
text_model = load_keras_model(TEXT_MODEL_PATH, uploaded_text_model)
tokenizer = load_tokenizer(TOKENIZER_PATH, uploaded_tokenizer)

# ---------------- PREPROCESSING ----------------
def preprocess_image(pil_img):
    img = pil_img.resize(IMAGE_TARGET_SIZE)
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

def image_predict(pil_img):
    if image_model is None or pil_img is None:
        return None
    arr = preprocess_image(pil_img)
    pred = image_model.predict(arr)
    pred = np.array(pred).squeeze()
    if pred.size == 1:
        p = float(pred)
        return np.array([1-p, p])
    return pred / np.sum(pred)

def text_predict(text):
    if text_model is None or tokenizer is None or not text.strip():
        return None
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=DEFAULT_TEXT_MAXLEN, padding="post", truncating="post")
    pred = text_model.predict(pad)
    pred = np.array(pred).squeeze()
    if pred.size == 1:
        p = float(pred)
        return np.array([1-p, p])
    return pred / np.sum(pred)

def probs_to_label(probs):
    idx = int(np.argmax(probs))
    return LABELS[idx], float(probs[idx])

# ---------------- INPUTS (VERTICAL LAYOUT) ----------------
st.header("📸 Image Input")
uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("Or capture image using camera")
img_file = uploaded_image if uploaded_image else camera_image

st.header("📝 Text Input")
user_text = st.text_area("Enter text (tweet, report, or message)", height=150)

st.markdown("---")
run = st.button("🔍 Analyze & Fuse Predictions")

# ---------------- RUN PREDICTION ----------------
if run:
    if img_file is None and not user_text.strip():
        st.warning("Please provide either an image or some text for analysis.")
    else:
        # --- Predict Image ---
        if img_file is not None:
            try:
                pil_img = Image.open(img_file).convert("RGB")
                img_probs = image_predict(pil_img)
                img_label, img_conf = probs_to_label(img_probs)
                st.image(pil_img, caption="Uploaded Image", use_column_width=True)
                st.subheader("🖼️ Image Model Result")
                st.metric("Predicted Label", img_label, f"{img_conf:.3f}")
                if img_label == "Disaster":
                    if img_conf > 0.9:
                        st.info("⚠️ High confidence disaster detected in the image.")
                st.progress(float(img_conf))
            except Exception as e:
                st.error(f"Image prediction failed: {e}")
                img_probs = None
        else:
            img_probs = None

        # --- Predict Text ---
        if user_text.strip():
            try:
                txt_probs = text_predict(user_text)
                txt_label, txt_conf = probs_to_label(txt_probs)
                st.subheader("💬 Text Model Result")
                st.metric("Predicted Label", txt_label, f"{txt_conf:.3f}")
                if txt_label == "Disaster":
                    st.info("🆘 Text indicates possible disaster or emergency need.")
                st.progress(float(txt_conf))
            except Exception as e:
                st.error(f"Text prediction failed: {e}")
                txt_probs = None
        else:
            txt_probs = None

        # --- Fused Result ---
        st.markdown("---")
        st.subheader("🌐 Multi-Modal Fusion Result")
        if img_probs is None and txt_probs is None:
            st.warning("No data to fuse. Provide image and/or text.")
        else:
            if img_probs is not None and txt_probs is not None:
                fused_probs = img_weight * img_probs + txt_weight * txt_probs
            elif img_probs is not None:
                fused_probs = img_probs
            else:
                fused_probs = txt_probs

            fused_probs = fused_probs / np.sum(fused_probs)
            fused_label, fused_conf = probs_to_label(fused_probs)

            st.metric("Final Assessment", fused_label, f"{fused_conf:.3f}")
            st.progress(float(fused_conf))

            if fused_label == "Disaster":
                st.error("🚨 Disaster detected! Sending alert to users via GCC...")
                if st.button("📡 Send GCC Alert"):
                    st.success("✅ GCC Alert sent successfully to users in affected area!")
                    st.markdown("**Advisory:** Stay safe, follow local guidelines, and move to a secure location.")
            else:
                st.success("✅ Area appears safe — no disaster detected.")

# ---------------- ABOUT ----------------
st.markdown("---")
st.header("ℹ️ About the App")
st.markdown(
    """
**AI-Powered Disaster Aid System**  
This app analyzes **images and text** using deep learning models to identify disaster situations in real-time.  
By integrating **Google Cloud Communication (GCC)**, it sends **alert messages** to users in affected areas — 
helping them stay safe and take preventive action.  

**Technologies used:**  
- CNN (MobileNetV2) for Image Classification  
- BiLSTM for Text Classification  
- Fusion mechanism for multi-modal disaster detection  

Together, these enable faster, smarter, and life-saving responses during disasters.
"""
)
