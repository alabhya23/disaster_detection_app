import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from PIL import Image
import time

# ------------------- PAGE SETUP -------------------
st.set_page_config(page_title="AI Disaster Detection & Aid System", layout="centered", page_icon="🌍")

st.title("🌍 AI-Powered Disaster Aid System")
st.markdown("""
Upload a **photo** and add a **caption or short report** —  
the system will detect possible disasters and guide you with safety measures.
""")

# ------------------- LOAD MODELS -------------------
@st.cache_resource
def load_models():
    try:
        img_model = load_model("disaster_cnn_mobilenet_clean.h5")
        txt_model = load_model("disaster_text_bilstm.h5")
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        return img_model, txt_model, tokenizer
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None, None

img_model, txt_model, tokenizer = load_models()

# ------------------- LABELS -------------------
LABELS = ["no_disaster", "fire", "flood", "earthquake", "hurricane", "drought", "landslide"]

# ------------------- HELPERS -------------------
def preprocess_image(image):
    img = image.resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    return pad_sequences(seq, maxlen=100)

def probs_to_label(probs):
    idx = int(np.argmax(probs))
    return LABELS[idx], float(probs[idx])

def precaution_message(label):
    label = label.lower()
    if "fire" in label:
        return "🔥 **Fire Alert:** Move to a safe area, avoid smoke, call emergency services."
    elif "flood" in label:
        return "🌊 **Flood Alert:** Move to higher ground, avoid floodwater, keep essentials dry."
    elif "earthquake" in label:
        return "🌍 **Earthquake Alert:** Drop, cover, and hold on. Stay away from windows."
    elif "hurricane" in label:
        return "🌪️ **Hurricane Alert:** Stay indoors, keep emergency kits, and follow weather updates."
    elif "landslide" in label:
        return "🏔️ **Landslide Alert:** Avoid slopes, move to open safe ground."
    elif "drought" in label:
        return "☀️ **Drought Detected:** Conserve water, stay hydrated, avoid unnecessary water usage."
    else:
        return "✅ **No disaster detected. Stay alert and safe.**"

# ------------------- USER INPUT -------------------
st.subheader("📷 Upload Image (Optional)")
uploaded_image = st.file_uploader("Upload a disaster-related image", type=["jpg", "jpeg", "png"])
if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
else:
    image = None

st.subheader("💬 Caption or Text Description (Optional)")
user_text = st.text_area("Describe the situation", placeholder="Example: Flood water entering my house...")

# ------------------- RUN ANALYSIS -------------------
if st.button("🔍 Analyze"):
    if not any([image, user_text.strip()]):
        st.warning("Please upload an image or write a short description.")
    elif img_model is None or txt_model is None or tokenizer is None:
        st.error("Models not loaded properly. Ensure all three model files (.h5 and .pkl) are present.")
    else:
        img_probs, txt_probs = None, None

        # IMAGE MODEL PREDICTION
        if image is not None:
            with st.spinner("Analyzing image..."):
                try:
                    img_input = preprocess_image(image)
                    img_probs = img_model.predict(img_input, verbose=0)[0]
                    img_label, img_conf = probs_to_label(img_probs)
                    st.success(f"🖼️ Image Prediction: **{img_label.title()}** ({img_conf:.2f})")
                except Exception as e:
                    st.error(f"Image analysis failed: {e}")

        # TEXT MODEL PREDICTION
        if user_text.strip():
            with st.spinner("Analyzing text..."):
                try:
                    txt_input = preprocess_text(user_text)
                    txt_probs = txt_model.predict(txt_input, verbose=0)[0]
                    txt_label, txt_conf = probs_to_label(txt_probs)
                    st.success(f"💬 Text Prediction: **{txt_label.title()}** ({txt_conf:.2f})")
                except Exception as e:
                    st.error(f"Text analysis failed: {e}")

        # FUSION — only if any prediction is available
        if img_probs is not None or txt_probs is not None:
            # Assign weights (image slightly higher)
            w_img, w_txt = 0.6, 0.4
            if img_probs is None:
                img_probs = np.zeros(len(LABELS))
                w_img = 0.0
            if txt_probs is None:
                txt_probs = np.zeros(len(LABELS))
                w_txt = 0.0

            total = w_img + w_txt
            w_img /= total
            w_txt /= total

            # Ensure both vectors same length
            img_probs = np.array(img_probs)
            txt_probs = np.array(txt_probs)
            fused_probs = w_img * img_probs + w_txt * txt_probs

            fused_label, fused_conf = probs_to_label(fused_probs)

            st.markdown("---")
            st.subheader("🔗 Combined Disaster Detection Result")
            st.metric("Detected Type", fused_label.replace("_", " ").title(), f"{fused_conf:.3f}")
            st.progress(fused_conf)
            st.info(precaution_message(fused_label))

            # Alert section
            if "no" not in fused_label.lower():
                st.warning("🚨 **Potential Disaster Detected!** Authorities have been notified through GCC.")
                if st.button("📡 Simulate Sending Alert"):
                    with st.spinner("Sending aid request via GCC..."):
                        time.sleep(2)
                    st.success("✅ GCC Alert Sent Successfully. Emergency team notified!")
                    st.balloons()
        else:
            st.error("⚠️ No valid inputs for analysis.")

# ------------------- KNOWLEDGE PANEL -------------------
with st.expander("📘 Learn About Disasters"):
    st.write("""
- **Floods:** Heavy rainfall or dam breaks. Move to high ground.  
- **Fires:** Caused by heat or electrical faults. Evacuate immediately.  
- **Droughts:** Extended lack of rain. Conserve and reuse water.  
- **Earthquakes:** Sudden shaking of ground. Drop, cover, and hold.  
- **Hurricanes:** Strong rotating storms. Stay indoors and secure objects.  
- **Landslides:** Earth movement on slopes. Avoid steep areas.
    """)

# ------------------- ABOUT SECTION -------------------
st.markdown("""
---
### ℹ️ About
This app uses **AI + Deep Learning** to identify natural disasters from **user-uploaded media and text posts** —  
similar to how people share images with captions on social media during emergencies.  
It can send alerts and offer tailored **safety instructions** to assist affected users.
""")
