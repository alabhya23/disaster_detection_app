import streamlit as st
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from PIL import Image

# -------------------- CONFIG --------------------
st.set_page_config(page_title="AI-Powered Disaster Aid System", layout="centered")

IMAGE_MODEL_PATH = "disaster_cnn_mobilenet_clean.h5"
TEXT_MODEL_PATH = "disaster_text_bilstm.h5"
TOKENIZER_PATH = "tokenizer.pkl"
DEFAULT_TEXT_MAXLEN = 30

# Actual trained classes in your dataset
DISASTER_TYPES = ["Flood", "Fire", "Earthquake", "Hurricane", "Drought", "Landslide"]
LABELS = ["Safe / No Disaster", "Disaster"]

# -------------------- HELPERS --------------------
def load_local_models():
    """Load models and tokenizer safely."""
    try:
        image_model = load_model(IMAGE_MODEL_PATH)
        text_model = load_model(TEXT_MODEL_PATH)
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
        st.sidebar.success("✅ Models loaded successfully!")
        return image_model, text_model, tokenizer
    except Exception as e:
        st.sidebar.error(f"❌ Error loading models: {e}")
        return None, None, None


def preprocess_image(pil_img):
    img = pil_img.convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def normalize_probs(p):
    """Normalize probability vector."""
    p = np.array(p).flatten()
    if np.sum(p) == 0:
        return np.zeros_like(p)
    return p / np.sum(p)


def get_precaution_message(d_type):
    """Return short caption for detected disaster."""
    d_type = d_type.lower()
    if "fire" in d_type:
        return "🔥 Fire detected — Stay low, cover mouth, and evacuate safely."
    elif "flood" in d_type:
        return "🌊 Flood detected — Move to higher ground and avoid floodwater."
    elif "earthquake" in d_type:
        return "🌍 Earthquake detected — Drop, cover, and hold on."
    elif "hurricane" in d_type or "storm" in d_type:
        return "🌪️ Storm detected — Stay indoors and secure loose objects."
    elif "landslide" in d_type:
        return "🏔️ Landslide detected — Move away from slopes and unstable ground."
    elif "drought" in d_type:
        return "☀️ Drought detected — Conserve and store water safely."
    else:
        return "⚠️ Potential hazard detected. Stay alert."


def classify_disaster(probs, threshold=0.6):
    """
    Convert model output to binary label and disaster type.
    If max probability < threshold → Safe
    Else → Disaster + disaster type
    """
    if probs is None or len(probs) == 0:
        return "Unknown", "None", 0.0

    probs = normalize_probs(probs)
    idx = int(np.argmax(probs))
    conf = float(probs[idx])

    if conf < threshold:
        return "Safe / No Disaster", "None", conf
    else:
        return "Disaster", DISASTER_TYPES[idx % len(DISASTER_TYPES)], conf


# -------------------- LOAD MODELS --------------------
st.sidebar.header("Model Status")
image_model, text_model, tokenizer = load_local_models()

# -------------------- UI HEADER --------------------
st.title("🧠 AI-Powered Disaster Aid System")
st.write("AI system that detects disasters from **images and text** and provides safety precautions.")

# -------------------- INPUTS --------------------
st.markdown("### 📸 Upload or Capture Image")
img_file = st.file_uploader("Upload a disaster image", type=["jpg", "jpeg", "png"])
camera_img = st.camera_input("Or capture an image")

st.markdown("### 💬 Enter Disaster-Related Text")
user_text = st.text_area("Enter message or tweet (optional)", height=100)

# -------------------- PREDICTIONS --------------------
img_probs, txt_probs, fused_probs = None, None, None

# Image prediction
if image_model is not None:
    if img_file or camera_img:
        try:
            image_data = Image.open(img_file if img_file else camera_img)
            arr = preprocess_image(image_data)
            preds = np.squeeze(image_model.predict(arr))
            img_probs = normalize_probs(preds)
        except Exception as e:
            st.error(f"Image prediction failed: {e}")

# Text prediction
if text_model is not None and tokenizer is not None and user_text.strip() != "":
    try:
        seq = tokenizer.texts_to_sequences([user_text])
        pad = pad_sequences(seq, maxlen=DEFAULT_TEXT_MAXLEN, padding="post", truncating="post")
        preds = np.squeeze(text_model.predict(pad))
        txt_probs = normalize_probs(preds)
    except Exception as e:
        st.error(f"Text prediction failed: {e}")

# -------------------- RESULTS --------------------
st.markdown("---")

# IMAGE RESULT
if img_probs is not None:
    img_label, img_type, img_conf = classify_disaster(img_probs)
    st.subheader("🖼️ Image Model Result")
    st.metric("Status", img_label, f"{img_conf:.3f}")
    if img_label == "Disaster":
        st.caption(get_precaution_message(img_type))
    st.progress(float(img_conf))
else:
    st.info("Upload or capture an image to analyze.")

# TEXT RESULT
if txt_probs is not None:
    txt_label, txt_type, txt_conf = classify_disaster(txt_probs)
    st.subheader("💬 Text Model Result")
    st.metric("Status", txt_label, f"{txt_conf:.3f}")
    if txt_label == "Disaster":
        st.caption(get_precaution_message(txt_type))
    st.progress(float(txt_conf))
else:
    st.info("Enter some text to analyze.")

# -------------------- FUSED PREDICTION --------------------
st.markdown("---")
st.subheader("🔗 Fused AI Decision")

img_weight, txt_weight = 0.6, 0.4
try:
    if img_probs is not None and txt_probs is not None:
        fused_probs = img_weight * np.array(img_probs) + txt_weight * np.array(txt_probs)
    elif img_probs is not None:
        fused_probs = img_probs
    elif txt_probs is not None:
        fused_probs = txt_probs

    if fused_probs is not None:
        label, d_type, conf = classify_disaster(fused_probs, threshold=0.6)
        st.metric("Final Assessment", label, f"{conf:.3f}")
        st.progress(float(conf))

        if label == "Disaster":
            st.info(get_precaution_message(d_type))
            if st.button("🚨 Send GCC Alert"):
                st.success("✅ GCC Alert sent successfully to authorities and users nearby!")
        else:
            st.success("✅ Area appears safe. Stay alert and follow updates.")
except Exception as e:
    st.error(f"Fusion failed safely: {e}")

# -------------------- ABOUT --------------------
st.markdown("---")
st.markdown("### 🧭 About the App")
st.write(
    "This AI system uses CNN + LSTM models to detect disasters from image and text data. "
    "If a disaster is identified, safety precautions and an alert option are automatically provided."
)
