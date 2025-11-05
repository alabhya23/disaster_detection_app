import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

# ------------------------------------------------------------
# 🔹 Load models and tokenizer once
# ------------------------------------------------------------
@st.cache_resource
def load_models():
    from tensorflow.keras import models
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    text_model = models.load_model("disaster_text_bilstm.h5", compile=False)
    image_model = models.load_model("disaster_cnn_mobilenet_clean.h5", compile=False)
    return text_model, image_model, tokenizer

text_model, image_model, tokenizer = load_models()

# ------------------------------------------------------------
# 🔹 Text cleaning function
# ------------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ------------------------------------------------------------
# 🔹 Prediction function
# ------------------------------------------------------------
def predict_text(text):
    MAX_LEN = 60
    seq = tokenizer.texts_to_sequences([clean_text(text)])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    pred = text_model.predict(pad)[0][0]
    label = "🚨 Disaster" if pred > 0.5 else "✅ Safe"
    conf = round(pred * 100 if pred > 0.5 else (1 - pred) * 100, 2)
    return label, conf

# ------------------------------------------------------------
# 🧠 Streamlit UI
# ------------------------------------------------------------
st.title("🌍 Disaster Detection App")

user_text = st.text_area("Enter a message to analyze", placeholder="Example: Heavy floods have destroyed the bridge...")

if st.button("Analyze Text"):
    if user_text.strip():
        label, conf = predict_text(user_text)
        st.markdown(f"### {label} ({conf}%)")
        if "Disaster" in label:
            st.warning("🚨 Alert Sent to Disaster Response System!")
    else:
        st.error("Please enter some text first!")
