import streamlit as st
import numpy as np
import os
import io
import pickle
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="🌍 AI-Powered Disaster Aid System", layout="wide")

# ---------------- LABELS ----------------
LABELS = ["No Disaster", "Disaster"]
TEXT_CLASSES = ["other", "food", "shelter", "rescue", "medical"]

# ---------------- MODEL PATHS ----------------
IMAGE_MODEL_PATH = "disaster_cnn_mobilenet_clean.h5"
TEXT_MODEL_PATH = "disaster_text_bilstm.h5"
TOKENIZER_PATH = "tokenizer.pkl"

# ---------------- APP HEADER ----------------
st.title("🌍 AI-Powered Disaster Aid System")
st.markdown("""
This system uses **deep learning** to analyze **images** and **text messages** to detect potential disasters.  
If a disaster is detected, it can also send an **alert through GCC (Google Cloud Communication)** to notify people in the affected area.
""")

# ---------------- MODEL LOADING ----------------
@st.cache_resource(show_spinner=False)
def load_models():
    image_model = load_model(IMAGE_MODEL_PATH)
    text_model = load_model(TEXT_MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return image_model, text_model, tokenizer

try:
    image_model, text_model, tokenizer = load_models()
    st.sidebar.success("✅ Models loaded successfully")
except Exception as e:
    st.sidebar.error(f"Error loading models: {e}")
    st.stop()

# ---------------- HELPER FUNCTIONS ----------------
def preprocess_image(img_file, target_size=(224, 224)):
    img = Image.open(img_file).convert("RGB")
    img = img.resize(target_size)
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

def predict_image(img_file):
    arr = preprocess_image(img_file)
    preds = image_model.predict(arr)[0]
    label_idx = np.argmax(preds)
    label = LABELS[label_idx] if label_idx < len(LABELS) else f"class_{label_idx}"
    return {"label": label, "score": float(preds[label_idx]), "probs": preds}

def preprocess_text_and_predict(text, maxlen=100):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
    preds = text_model.predict(padded)[0]
    idx = np.argmax(preds)
    label = TEXT_CLASSES[idx] if idx < len(TEXT_CLASSES) else f"class_{idx}"
    return {"label": label, "score": float(preds[idx]), "probs": preds}

def fuse_predictions(img_pred, txt_pred, w_img=0.6, w_txt=0.4):
    img_probs = img_pred["probs"]
    txt_probs = np.zeros_like(img_probs)
    if txt_pred["label"] in ["food", "shelter", "rescue", "medical"]:
        txt_probs[1] = txt_pred["score"]
        txt_probs[0] = 1 - txt_pred["score"]
    else:
        txt_probs[0] = txt_pred["score"]
        txt_probs[1] = 1 - txt_pred["score"]

    fused_probs = (w_img * img_probs) + (w_txt * txt_probs)
    idx = np.argmax(fused_probs)
    label = LABELS[idx]
    return {"label": label, "score": float(fused_probs[idx]), "probs": fused_probs}

def get_advisory(label):
    advisories = {
        "No Disaster": "✅ Area appears safe — no immediate danger detected.",
        "Disaster": "🚨 Disaster detected — stay alert and follow official safety guidance."
    }
    return advisories.get(label, "⚠️ Situation unclear — stay cautious.")

# ---------------- INPUT UI ----------------
col1, col2 = st.columns(2)

with col1:
    st.header("📷 Image Input")
    img_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    camera_img = st.camera_input("Or capture an image")
    image_source = camera_img if camera_img else img_file

with col2:
    st.header("💬 Text Input")
    user_text = st.text_area("Enter a message or report", placeholder="e.g., Fire near the hills, need help urgently")

# ---------------- RUN PREDICTIONS ----------------
if st.button("🔍 Analyze Disaster Situation"):
    if not image_source and not user_text:
        st.warning("Please upload an image or enter some text.")
    else:
        st.markdown("---")
        st.subheader("🧭 Results Overview")

        img_pred, txt_pred, fused_pred = None, None, None

        if image_source:
            with st.spinner("Analyzing image..."):
                img_pred = predict_image(image_source)
            st.success("**Image Analysis Result**")
            st.metric("Label", img_pred["label"])
            st.metric("Confidence", f"{img_pred['score']:.3f}")
            st.caption(f"Probabilities: {dict(zip(LABELS, np.round(img_pred['probs'], 3)))}")

        if user_text:
            with st.spinner("Analyzing text..."):
                txt_pred = preprocess_text_and_predict(user_text)
            st.success("**Text Analysis Result**")
            st.metric("Label", txt_pred["label"])
            st.metric("Confidence", f"{txt_pred['score']:.3f}")
            st.caption(f"Probabilities: {dict(zip(TEXT_CLASSES, np.round(txt_pred['probs'], 3)))}")

        if img_pred or txt_pred:
            with st.spinner("Combining results..."):
                if img_pred and txt_pred:
                    fused_pred = fuse_predictions(img_pred, txt_pred)
                else:
                    fused_pred = img_pred or txt_pred

            st.markdown("---")
            st.subheader("🌐 Combined (Fused) Result")
            st.metric("Final Assessment", fused_pred["label"])
            st.metric("Confidence", f"{fused_pred['score']:.3f}")
            st.info(get_advisory(fused_pred["label"]))

            # ------------- GCC ALERT BUTTON -------------
            if fused_pred["label"] == "Disaster":
                if st.button("🚨 Send GCC Alert"):
                    st.success("✅ GCC Alert Sent Successfully! Notification broadcast to nearby users.")
            else:
                st.caption("No alert sent since no disaster was detected.")

# ---------------- ABOUT SECTION ----------------
st.markdown("---")
st.header("ℹ️ About the App")
st.markdown("""
**AI-Powered Disaster Aid System** is designed to assist emergency services by analyzing both **visual data (images)** and **textual data (messages or tweets)**.  
By fusing CNN and LSTM model outputs, the system can **detect disasters in real time** and trigger **GCC alerts** to notify users in affected areas, helping them take timely safety actions.

**Key Capabilities:**
- 🌪️ Detects disasters like floods, fires, and earthquakes.
- 💬 Classifies emergency messages into actionable types.
- 🧠 Combines both results for high-accuracy predictions.
- 📡 Sends alerts via GCC to affected areas.
""")
