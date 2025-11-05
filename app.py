import streamlit as st
import numpy as np
import pickle
import os
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from PIL import Image

# -------------------- CONFIG --------------------
st.set_page_config(page_title="AI-Powered Disaster Aid System", layout="centered")

IMAGE_MODEL_PATH = "disaster_cnn_mobilenet_clean.h5"
TEXT_MODEL_PATH = "disaster_text_bilstm.h5"
TOKENIZER_PATH = "tokenizer.pkl"
DEFAULT_TEXT_MAXLEN = 30

LABELS = ["Safe / No Disaster", "Disaster"]

# -------------------- HELPERS --------------------
def load_local_models():
    """Load models and tokenizer safely."""
    try:
        image_model = load_model(IMAGE_MODEL_PATH)
        text_model = load_model(TEXT_MODEL_PATH)
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
        st.sidebar.success("✅ All models loaded successfully!")
        return image_model, text_model, tokenizer
    except Exception as e:
        st.sidebar.error(f"❌ Error loading models: {e}")
        return None, None, None


def preprocess_image(pil_img):
    img = pil_img.convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def probs_to_label(probs):
    idx = int(np.argmax(probs))
    return LABELS[idx], float(probs[idx])


def get_precaution_message(label):
    label = label.lower()
    if "fire" in label:
        return "🔥 Fire detected! Stay away from flames and evacuate immediately."
    elif "flood" in label:
        return "🌊 Flooding reported! Move to higher ground and avoid floodwater."
    elif "earthquake" in label:
        return "🌍 Earthquake detected! Move to an open area and stay away from buildings."
    elif "hurricane" in label or "storm" in label:
        return "🌪️ Severe storm detected! Stay indoors and away from windows."
    elif "landslide" in label:
        return "🏔️ Landslide risk! Move to safer ground immediately."
    elif "safe" in label or "no disaster" in label:
        return "✅ No disaster detected. Stay alert and safe."
    else:
        return "⚠️ Potential hazard detected. Follow local alerts for safety."


# -------------------- LOAD MODELS --------------------
st.sidebar.header("Model Status")
image_model, text_model, tokenizer = load_local_models()

# -------------------- APP HEADER --------------------
st.title("🧠 AI-Powered Disaster Aid System")
st.write(
    "This system uses **AI + Deep Learning** to detect disasters from **images and text messages**. "
    "It helps authorities and users receive timely alerts and take preventive actions."
)

# -------------------- USER INPUTS --------------------
st.markdown("### 📸 Upload or Capture Image")
img_file = st.file_uploader("Upload a disaster image", type=["jpg", "jpeg", "png"])
camera_img = st.camera_input("Or capture using camera")

st.markdown("### 💬 Enter Disaster-Related Text")
user_text = st.text_area("Enter message or tweet (optional)", height=100)

# -------------------- PREDICTIONS --------------------
img_probs = None
txt_probs = None
fused_probs = None

if image_model is not None:
    if img_file or camera_img:
        try:
            image_data = Image.open(img_file if img_file else camera_img)
            arr = preprocess_image(image_data)
            img_probs = np.squeeze(image_model.predict(arr))
            if img_probs.size == 1:  # Binary output
                p = float(img_probs)
                img_probs = np.array([1 - p, p])
        except Exception as e:
            st.error(f"Image prediction failed: {e}")

if text_model is not None and tokenizer is not None and user_text.strip() != "":
    try:
        seq = tokenizer.texts_to_sequences([user_text])
        pad = pad_sequences(seq, maxlen=DEFAULT_TEXT_MAXLEN, padding="post", truncating="post")
        txt_probs = np.squeeze(text_model.predict(pad))
        if txt_probs.size == 1:
            p = float(txt_probs)
            txt_probs = np.array([1 - p, p])
    except Exception as e:
        st.error(f"Text prediction failed: {e}")

# -------------------- RESULTS --------------------
st.markdown("---")

# Show individual predictions
if img_probs is not None:
    img_label, img_conf = probs_to_label(img_probs)
    st.subheader("🖼️ Image Model Prediction")
    st.metric("Predicted", img_label, f"{img_conf:.3f}")
    st.progress(float(img_conf))
else:
    st.info("No image provided yet.")

if txt_probs is not None:
    txt_label, txt_conf = probs_to_label(txt_probs)
    st.subheader("💬 Text Model Prediction")
    st.metric("Predicted", txt_label, f"{txt_conf:.3f}")
    st.progress(float(txt_conf))
else:
    st.info("No text provided yet.")

# Fused (combined) prediction
st.markdown("---")
st.subheader("🔗 Fused Multi-Modal Prediction")

img_weight, txt_weight = 0.6, 0.4
if img_probs is not None and txt_probs is not None:
    fused_probs = img_weight * img_probs + txt_weight * txt_probs
elif img_probs is not None:
    fused_probs = img_probs
elif txt_probs is not None:
    fused_probs = txt_probs

if fused_probs is not None:
    fused_label, fused_conf = probs_to_label(fused_probs)
    st.metric("Final Assessment", fused_label, f"{fused_conf:.3f}")
    st.progress(float(fused_conf))

    # Show precaution message
    precaution = get_precaution_message(fused_label)
    st.info(precaution)

    # GCC Alert button
    if "Disaster" in fused_label:
        if st.button("🚨 Send GCC Alert"):
            st.success("✅ Alert sent to nearby authorities and users in the affected area!")
else:
    st.warning("Please provide image or text input to generate predictions.")

# -------------------- ABOUT --------------------
st.markdown("---")
st.markdown("### 🧭 About the App")
st.write(
    "The **AI-Powered Disaster Aid System** analyzes both images and text messages to detect ongoing disasters. "
    "It integrates CNN and LSTM-based models to identify patterns and provide real-time alerts. "
    "Using GCC (Global Command Center) integration, alerts are sent to users and authorities, "
    "helping them take timely action and stay safe."
)

st.caption("Developed for Smart Disaster Management and Sustainable Development Goals (SDG 11 & 13). 🌍")
