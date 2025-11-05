# app.py
import streamlit as st
import numpy as np
import os
import pickle
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Page config
st.set_page_config(page_title="AI-Powered Disaster Aid System", layout="wide")

# --- Styling (small CSS for spacing & card look)
st.markdown(
    """
<style>
/* container padding & card boxes */
.section-card {
  background-color: rgba(255,255,255,0.02);
  border-radius: 12px;
  padding: 18px;
  box-shadow: rgba(0,0,0,0.12) 0px 6px 18px;
  margin-bottom: 18px;
}
/* larger headers */
.title {
  font-size: 38px;
  font-weight: 700;
  margin-bottom: 6px;
}
.subtitle {
  font-size: 15px;
  color: #bfc7d6;
  margin-top: -8px;
  margin-bottom: 16px;
}
/* make labels bold */
.input-label {
  font-weight: 600;
  margin-bottom: 8px;
}
.result-card {
  background-color: rgba(255,255,255,0.02);
  border-radius: 12px;
  padding: 16px;
}
.small-muted {
  color: #98a0b3;
  font-size: 13px;
}
</style>
""",
    unsafe_allow_html=True,
)

# --- Config / Labels
IMAGE_TARGET_SIZE = (224, 224)
DEFAULT_TEXT_MAXLEN = 100
LABELS = ["Safe / No Disaster", "Disaster"]

# --- Header
st.markdown('<div class="title">🌍 AI-Powered Disaster Aid System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect disasters from images & short text, then fuse results to provide a reliable alert.</div>', unsafe_allow_html=True)

# --- Sidebar (model upload + fusion)
st.sidebar.header("Model settings & fusion")
IMAGE_MODEL_PATH = "disaster_cnn_mobilenet_clean.h5"
TEXT_MODEL_PATH = "disaster_text_bilstm.h5"
TOKENIZER_PATH = "tokenizer.pkl"

st.sidebar.write(f"Image model: {'✅' if os.path.exists(IMAGE_MODEL_PATH) else '❌ missing'}")
st.sidebar.write(f"Text model: {'✅' if os.path.exists(TEXT_MODEL_PATH) else '❌ missing'}")
st.sidebar.write(f"Tokenizer: {'✅' if os.path.exists(TOKENIZER_PATH) else '❌ missing'}")

uploaded_image_model = st.sidebar.file_uploader("Upload Image Model (.h5)", type=["h5"])
uploaded_text_model = st.sidebar.file_uploader("Upload Text Model (.h5)", type=["h5"])
uploaded_tokenizer = st.sidebar.file_uploader("Upload Tokenizer (.pkl)", type=["pkl","pickle","dat"])

st.sidebar.markdown("---")
text_maxlen = st.sidebar.number_input("Text max length", min_value=10, max_value=1000, value=DEFAULT_TEXT_MAXLEN, step=10)

st.sidebar.markdown("### Fusion weights")
img_weight = st.sidebar.slider("Image weight", 0.0, 1.0, 0.5, 0.05)
txt_weight = st.sidebar.slider("Text weight", 0.0, 1.0, 0.5, 0.05)

# --- Model loaders with caching
@st.cache_resource(show_spinner=False)
def load_keras_model(path, uploaded=None):
    try:
        if uploaded is not None:
            tmp = f"tmp_{path}"
            with open(tmp, "wb") as f:
                f.write(uploaded.getbuffer())
            m = load_model(tmp)
            os.remove(tmp)
            return m
        if os.path.exists(path):
            return load_model(path)
    except Exception as e:
        st.sidebar.error(f"Model load error: {e}")
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
        st.sidebar.error(f"Tokenizer load error: {e}")
    return None

image_model = load_keras_model(IMAGE_MODEL_PATH, uploaded_image_model)
text_model = load_keras_model(TEXT_MODEL_PATH, uploaded_text_model)
tokenizer = load_tokenizer(TOKENIZER_PATH, uploaded_tokenizer)

# --- Helper functions
def preprocess_image_for_model(pil_img, target_size=IMAGE_TARGET_SIZE):
    pil_img = ImageOps.fit(pil_img, target_size, Image.LANCZOS)
    arr = keras_image.img_to_array(pil_img)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)

def image_predict_probs(pil_img, model):
    if model is None or pil_img is None:
        return None
    x = preprocess_image_for_model(pil_img)
    pred = model.predict(x)
    pred = np.array(pred).squeeze()
    if pred.size == 1:
        p = float(pred)
        return np.array([1 - p, p])
    probs = pred / np.sum(pred) if pred.sum() > 0 else np.zeros_like(pred)
    return probs

def text_predict_probs(text, tok, model, maxlen=DEFAULT_TEXT_MAXLEN):
    if model is None or tok is None or not (text and text.strip()):
        return None
    seq = tok.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=int(maxlen), padding="post", truncating="post")
    pred = model.predict(padded)
    pred = np.array(pred).squeeze()
    if pred.size == 1:
        p = float(pred)
        return np.array([1 - p, p])
    probs = pred / np.sum(pred) if pred.sum() > 0 else np.zeros_like(pred)
    return probs

def probs_to_label(probs):
    idx = int(np.argmax(probs))
    return LABELS[idx], float(probs[idx])

# --- Input area (spacious layout)
st.markdown('<div class="section-card">', unsafe_allow_html=True)
col_img, col_text = st.columns([0.55, 0.45])

with col_img:
    st.markdown('<div class="input-label">🖼️ Image Input</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-muted">Upload JPG/PNG or take a picture (camera works in supported browsers).</div>', unsafe_allow_html=True)
    upload = st.file_uploader("Upload image", type=["jpg","jpeg","png"], key="img")
    cam = st.camera_input("Use camera (optional)", key="cam")
    # pick image data (uploaded takes precedence)
    image_file = upload if upload is not None else cam
    if image_file is not None:
        try:
            pil_image = Image.open(image_file).convert("RGB")
            st.image(pil_image, use_column_width=True, caption="Preview")
        except Exception as e:
            st.error(f"Cannot open image: {e}")
            pil_image = None
    else:
        pil_image = None
        st.info("No image selected")

with col_text:
    st.markdown('<div class="input-label">💬 Text Input</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-muted">Enter eyewitness text (tweet, SMS, short report)</div>', unsafe_allow_html=True)
    user_text = st.text_area("Text", height=220, placeholder="e.g., fire in the area, flooding on main street...")
st.markdown('</div>', unsafe_allow_html=True)

# --- Controls row
st.markdown("")  # spacer
controls_col1, controls_col2, controls_col3 = st.columns([0.32, 0.32, 0.36])
with controls_col1:
    analyze_btn = st.button("🔎 Analyze and Fuse", use_container_width=True)
with controls_col2:
    clear_btn = st.button("Reset inputs", use_container_width=True)
with controls_col3:
    st.markdown("<div class='small-muted'>Adjust fusion weights in the sidebar to prefer image or text.</div>", unsafe_allow_html=True)

if clear_btn:
    # Streamlit doesn't allow programmatic clearing of file_uploader, but we can suggest refresh
    st.experimental_rerun()

# --- Result area: clean card
st.markdown('<div class="result-card">', unsafe_allow_html=True)
if analyze_btn:
    # ensure at least one input
    if (pil_image is None) and (not user_text or user_text.strip() == ""):
        st.warning("Please provide an image or text to analyze.")
    else:
        with st.spinner("Running models..."):
            img_probs = image_predict_probs(pil_image, image_model) if pil_image is not None else None
            txt_probs = text_predict_probs(user_text, tokenizer, text_model, maxlen=text_maxlen) if user_text and user_text.strip() != "" else None

        # normalize weights
        total_w = img_weight + txt_weight
        if total_w <= 0:
            w_img, w_txt = 0.5, 0.5
        else:
            w_img, w_txt = img_weight / total_w, txt_weight / total_w

        # fusion logic: attempt to align vector sizes; default target = max length among available
        fused_probs = None
        if img_probs is not None and txt_probs is not None:
            target_len = max(len(img_probs), len(txt_probs), len(LABELS))
            def resize(p, target_len):
                p = np.array(p)
                if p.size == target_len:
                    return p
                if p.size < target_len:
                    out = np.zeros(target_len)
                    out[:p.size] = p
                    return out
                return p[:target_len]
            img_up = resize(img_probs, target_len)
            txt_up = resize(txt_probs, target_len)
            fused_probs = w_img * img_up + w_txt * txt_up
        elif img_probs is not None:
            fused_probs = img_probs
            st.info("Only image available — using image model output.")
        elif txt_probs is not None:
            fused_probs = txt_probs
            st.info("Only text available — using text model output.")
            
            if fused_probs is not None:
            # normalize fused
                if fused_probs.sum() > 0:
                    fused_probs = fused_probs / fused_probs.sum()
                    idx = int(np.argmax(fused_probs))
                    # prevent out-of-range label lookup
    label = LABELS[idx] if idx < len(LABELS) else "Disaster"
    conf = float(fused_probs[idx])

    # large result presentation
    rcol1, rcol2 = st.columns([0.6, 0.4])
    with rcol1:
        st.markdown(f"### ✅ Final Assessment: **{label}**")
        st.metric("Confidence", f"{conf:.3f}")

        # show only relevant probabilities
        st.write("**Class probabilities:**")
        for i, p in enumerate(fused_probs):
            if p < 0.01:  # skip near-zero classes
                continue
            name = LABELS[i] if i < len(LABELS) else f"Class {i+1}"
            st.write(f"- {name}: {p:.3f}")

    with rcol2:
        disaster_prob = fused_probs[1] if len(fused_probs) > 1 else 0.0
        st.progress(min(max(float(disaster_prob), 0.0), 1.0))
        st.caption("Disaster likelihood")

    # 🆕 Smart advisory message
    if label.lower().startswith("disaster") or conf > 0.7:
        advisory = "⚠️ **Potential disaster detected!** Stay alert and follow safety protocols."
        if user_text:
            text_lower = user_text.lower()
            if "fire" in text_lower or "burn" in text_lower:
                advisory = "🔥 **Fire detected!** Stay away from flames, move to open areas, and contact emergency services."
            elif "flood" in text_lower or "water" in text_lower or "rain" in text_lower:
                advisory = "🌊 **Flooding detected!** Move to higher ground and avoid walking through flood water."
            elif "earthquake" in text_lower or "shake" in text_lower:
                advisory = "🏚️ **Possible earthquake!** Take cover under sturdy furniture and move away from windows."
            elif "storm" in text_lower or "cyclone" in text_lower or "hurricane" in text_lower:
                advisory = "🌪️ **Severe storm detected!** Stay indoors, secure windows, and avoid traveling."
        st.markdown(f"<div class='section-card'>{advisory}</div>", unsafe_allow_html=True)
    else:
        st.success("✅ Area appears safe. No immediate disaster detected.")

    # show breakdown (optional but cleaner)
    st.markdown("**Fusion breakdown**")
    st.write(f"- Image weight used: {w_img:.2f}")
    st.write(f"- Text weight used: {w_txt:.2f}")
    if img_probs is not None:
        st.write("📷 Image probabilities:", np.round(img_probs[:len(LABELS)], 3))
    if txt_probs is not None:
        st.write("📝 Text probabilities:", np.round(txt_probs[:len(LABELS)], 3))
else:
    st.warning("Model outputs were not available. Check that models/tokenizer are loaded in the sidebar.")

# --- Footer description
st.markdown("---")
st.header("About this system")
st.markdown(
    """
The **AI-Powered Disaster Aid System** uses image and text deep-learning models to detect
possible disaster events (fire, flood, etc.) from user-submitted images and short messages.
The system fuses predictions from both modalities to improve reliability.

When a disaster is detected, this system can be connected to a communication pipeline (e.g., GCC or other alert channel)
to notify users and authorities in the affected area with the alert and relevant contextual information.
"""
)
