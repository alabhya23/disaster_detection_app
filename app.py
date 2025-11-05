# ============================================================
# 🌍 PHASE 5: STREAMLIT REAL-TIME AI DISASTER DETECTION APP
# ============================================================

import streamlit as st
import tensorflow as tf
import numpy as np
import pickle, re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# ------------------------------------------------------------
# 1️⃣ Load Saved Models and Tokenizer
# ------------------------------------------------------------
@st.cache_resource
def load_models():
    import tensorflow as tf
    import pickle

    # ✅ Force legacy Keras 2 deserialization for .h5
    from tensorflow.keras import models
    text_model = models.load_model("disaster_text_bilstm.h5", compile=False)
    image_model = models.load_model("disaster_cnn_mobilenet_clean.h5", compile=False)

    # ✅ Load tokenizer (safe binary read)
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return text_model, image_model, tokenizer

# ------------------------------------------------------------
# 2️⃣ Helper Functions
# ------------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_text(text):
    MAX_LEN = 60
    seq = tokenizer.texts_to_sequences([clean_text(text)])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    pred = text_model.predict(pad)[0][0]
    label = "🚨 Disaster" if pred > 0.5 else "✅ Safe"
    conf = round(pred * 100, 2) if pred > 0.5 else round((1 - pred) * 100, 2)
    return label, conf

def predict_image(img):
    img = img.resize((224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    preds = image_model.predict(arr)
    idx = np.argmax(preds)
    conf = round(np.max(preds) * 100, 2)
    return idx, conf

# ------------------------------------------------------------
# 3️⃣ Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="AI Disaster Detection", page_icon="🌍", layout="wide")
st.title("🌍 AI-Powered Disaster Detection System")
st.write("Analyze text messages and images to detect potential disasters in real time.")

tab1, tab2 = st.tabs(["💬 Text Analysis", "🖼️ Image Analysis"])

# --- Text Analysis ---
with tab1:
    st.subheader("💬 Disaster Message Detection")
    user_text = st.text_area("Enter a message to analyze", placeholder="Example: Floods have destroyed homes and roads...")
    if st.button("Analyze Text"):
        if user_text.strip():
            label, conf = predict_text(user_text)
            st.markdown(f"### {label} ({conf}%)")
            if "Disaster" in label:
                st.warning("🚨 Alert Sent to Disaster Response System!")
        else:
            st.error("Please enter a message.")

# --- Image Analysis ---
with tab2:
    st.subheader("🖼️ Disaster Image Detection")
    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        img = image.load_img(uploaded_img)
        st.image(img, caption="Uploaded Image", use_container_width=True)
        if st.button("Analyze Image"):
            idx, conf = predict_image(img)
            st.markdown(f"### Detected Class Index: `{idx}` ({conf}%)")
            if conf > 70:
                st.warning("🚨 Alert Sent to Disaster Response System!")

# ------------------------------------------------------------
# 4️⃣ Footer
# ------------------------------------------------------------
st.markdown("---")
st.caption("Developed in Phase 5 — AI + Streamlit Disaster Detection System 🌍")
