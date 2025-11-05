import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from PIL import Image
import io
import time
import pydeck as pdk

# ------------------- PAGE SETUP -------------------
st.set_page_config(page_title="AI-Powered Disaster Aid System", layout="wide", page_icon="🌍")

st.title("🌍 AI-Powered Disaster Aid System")
st.markdown("""
This intelligent system analyses **images** and **text messages** to detect potential disasters and assist users with
timely safety advice. It uses **CNN (MobileNetV2)** for image understanding and **BiLSTM** for text classification.
""")

# ------------------- MODEL & TOKENIZER LOAD -------------------
@st.cache_resource
def load_models():
    try:
        img_model = load_model("disaster_cnn_mobilenet_clean.h5")
        txt_model = load_model("disaster_text_bilstm.h5")
        with open("tokenizer.pkl", "rb") as handle:
            tokenizer = pickle.load(handle)
        return img_model, txt_model, tokenizer
    except Exception as e:
        st.error(f"Error loading models or tokenizer: {e}")
        return None, None, None

img_model, txt_model, tokenizer = load_models()

# ------------------- LABELS -------------------
LABELS = ["no_disaster", "fire", "flood", "earthquake", "hurricane", "drought", "landslide"]

# ------------------- HELPER FUNCTIONS -------------------
def preprocess_image(image):
    img = image.resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    return pad_sequences(seq, maxlen=100)

def probs_to_label(probs):
    idx = np.argmax(probs)
    return LABELS[idx], float(probs[idx])

def get_precaution_message(label):
    label = label.lower()
    if "fire" in label:
        return ("🔥 **Fire Detected!**\n"
                "• Evacuate the area immediately.\n"
                "• Avoid smoke inhalation — cover mouth and nose.\n"
                "• Call emergency services if trapped.")
    elif "flood" in label:
        return ("🌊 **Flood Alert!**\n"
                "• Move to higher ground immediately.\n"
                "• Avoid walking or driving through floodwaters.\n"
                "• Keep emergency kits ready.")
    elif "earthquake" in label:
        return ("🌍 **Earthquake Warning!**\n"
                "• Drop, cover, and hold on.\n"
                "• Stay away from windows and heavy objects.\n"
                "• Move to an open area once shaking stops.")
    elif "hurricane" in label or "storm" in label:
        return ("🌪️ **Hurricane Detected!**\n"
                "• Stay indoors and close windows.\n"
                "• Store drinking water and emergency supplies.\n"
                "• Stay tuned to weather updates.")
    elif "landslide" in label:
        return ("🏔️ **Landslide Risk!**\n"
                "• Move away from slopes and unstable ground.\n"
                "• Avoid driving through hilly regions.")
    elif "drought" in label:
        return ("☀️ **Drought Condition Detected!**\n"
                "• Conserve water — avoid wastage.\n"
                "• Stay hydrated and avoid prolonged sun exposure.\n"
                "• Support community water initiatives.")
    elif "no" in label or "safe" in label:
        return "✅ **No immediate disaster detected.** Continue monitoring updates and stay prepared."
    else:
        return "⚠️ **Potential hazard detected.** Stay alert and follow local safety instructions."

# ------------------- INPUT SECTION -------------------
st.header("🧠 Provide Inputs")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Upload or Capture Image")
    uploaded_file = st.file_uploader("Upload a disaster-related image", type=["jpg", "png", "jpeg"])
    use_camera = st.checkbox("Use camera")
    if use_camera:
        camera_image = st.camera_input("Capture image")
    else:
        camera_image = None

with col2:
    st.subheader("💬 Enter a Text Message")
    user_text = st.text_area("Enter a message or report (optional)", placeholder="Example: There’s heavy flooding near the river...")

# ------------------- PREDICTION SECTION -------------------
if st.button("🔍 Analyze Situation"):
    if not img_model or not txt_model or not tokenizer:
        st.error("Models not loaded properly. Ensure all three files (.h5 and .pkl) are in the same directory.")
    else:
        img_probs = None
        txt_probs = None
        fused_probs = None

        # IMAGE PREDICTION
        if uploaded_file or camera_image:
            img_input = Image.open(uploaded_file or camera_image)
            st.image(img_input, caption="Uploaded Image", width=300)
            try:
                img_array = preprocess_image(img_input)
                img_probs = img_model.predict(img_array)[0]
                img_label, img_conf = probs_to_label(img_probs)

                st.markdown("### 🖼️ Image Analysis Result")
                st.metric("Predicted Type", img_label.replace("_", " ").title(), f"{img_conf:.3f}")
                st.progress(img_conf)
                st.info(get_precaution_message(img_label))
            except Exception as e:
                st.error(f"Image prediction failed: {e}")

        # TEXT PREDICTION
        if user_text.strip():
            try:
                txt_array = preprocess_text(user_text)
                txt_probs = txt_model.predict(txt_array)[0]
                txt_label, txt_conf = probs_to_label(txt_probs)

                st.markdown("### 💬 Text Analysis Result")
                st.metric("Predicted Type", txt_label.replace("_", " ").title(), f"{txt_conf:.3f}")
                st.progress(txt_conf)
                st.info(get_precaution_message(txt_label))
            except Exception as e:
                st.error(f"Text prediction failed: {e}")

        # COMBINED FUSION RESULT
        if img_probs is not None or txt_probs is not None:
            img_weight = 0.6 if img_probs is not None else 0.0
            txt_weight = 0.4 if txt_probs is not None else 0.0

            # Normalize weights
            total = img_weight + txt_weight
            if total == 0:
                st.warning("Please provide at least one input (image or text).")
            else:
                img_weight /= total
                txt_weight /= total

                # If one of them is missing, fill zeros
                if img_probs is None:
                    img_probs = np.zeros(len(LABELS))
                if txt_probs is None:
                    txt_probs = np.zeros(len(LABELS))

                fused_probs = img_weight * img_probs + txt_weight * txt_probs
                fused_label, fused_conf = probs_to_label(fused_probs)

                st.markdown("### 🔗 Combined Multi-Modal Assessment")
                st.metric("Final Assessment", fused_label.replace("_", " ").title(), f"{fused_conf:.3f}")
                st.progress(fused_conf)
                st.success(get_precaution_message(fused_label))

                # GCC Alert Simulation
                if "no" not in fused_label.lower():
                    if st.button("🚨 Send GCC Alert"):
                        with st.spinner("Notifying nearby authorities via GCC..."):
                            time.sleep(2)
                        st.success("✅ Alert successfully sent to local authorities and nearby users!")
                        st.balloons()

# ------------------- KNOWLEDGE PANEL -------------------
with st.expander("📘 Know Your Disaster"):
    st.write("""
    **Floods:** Caused by heavy rainfall or dam failure. Stay informed, move to higher ground, and keep emergency supplies ready.  
    **Droughts:** Long dry periods lead to water shortages. Conserve water and avoid wastage.  
    **Fires:** Common in hot, dry regions. Have an evacuation plan and avoid flammable materials near homes.  
    **Earthquakes:** Sudden shaking of the ground. Drop, cover, and hold on.  
    **Hurricanes:** Intense storms with heavy wind and rain. Secure property and stay indoors.  
    **Landslides:** Triggered by rain or tremors on slopes. Avoid steep areas and move to safe zones.
    """)

# ------------------- MAP ALERT -------------------
st.subheader("🌐 GCC Alert Map")
st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=pdk.ViewState(latitude=20.5937, longitude=78.9629, zoom=4),
    layers=[
        pdk.Layer(
            "ScatterplotLayer",
            data=[{"lat": 20.5937, "lon": 78.9629}],
            get_position=["lon", "lat"],
            get_color=[255, 0, 0, 160],
            get_radius=60000,
        ),
    ],
))

# ------------------- ABOUT APP -------------------
st.markdown("""
---
### ℹ️ About the App
This AI-Powered Disaster Aid System analyzes both **visual** and **textual** data to detect potential disasters in real time.
Using **Google Cloud Communication (GCC)**, the system alerts users and authorities to ensure faster, smarter disaster response.
""")
