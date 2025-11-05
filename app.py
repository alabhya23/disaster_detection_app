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
import json
from datetime import datetime

# ---------------- Page config & small CSS ----------------
st.set_page_config(page_title="AI-Powered Disaster Aid System", layout="wide")
st.markdown(
    """
<style>
.section-card { background-color: rgba(255,255,255,0.02); border-radius: 12px; padding: 18px; margin-bottom: 18px; }
.title { font-size: 34px; font-weight:700; margin-bottom:6px; }
.subtitle { font-size:14px; color:#bfc7d6; margin-bottom:12px; }
.input-label { font-weight:600; margin-bottom:8px; }
.small-muted { color:#98a0b3; font-size:13px; }
.result-card { background-color: rgba(255,255,255,0.02); border-radius:12px; padding:16px; }
</style>
""", unsafe_allow_html=True
)

# ---------------- Defaults & config ----------------
IMAGE_TARGET_SIZE = (224, 224)
DEFAULT_TEXT_MAXLEN = 100

# ---------------- Header + Objectives ----------------
st.markdown('<div class="title">🌍 AI-Powered Disaster Aid System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect disasters from images & short text, fuse evidence, and alert stakeholders (GCC integration placeholder).</div>', unsafe_allow_html=True)

st.markdown("### Project Objectives")
st.markdown(
    """
1. **Disaster Image Classification** — Build a CNN to identify disaster types (floods, fires, earthquakes, hurricanes) from images for rapid situational awareness.  
2. **Emergency Message Classification** — Use NLP (LSTM/Transformer) to categorize incoming messages into actionable needs (food, shelter, rescue, medical).  
3. **Integrated Smart Response System** — Fuse image & text outputs to assist agencies and NGOs in data-driven rapid response.  
4. **Support Sustainable Development Goals (SDGs)** — Contribute to SDG 3 (Good Health), SDG 11 (Resilient Cities), and SDG 13 (Climate Action) by enabling faster aid.  
5. **Future Scalability** — Design for extension to real-time drone/satellite/IoT feeds and large-scale monitoring.
"""
)

# ---------------- Sidebar: models, labels, fusion ----------------
st.sidebar.header("Model settings & system options")
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
# Label configuration: comma-separated defaults
default_image_labels = "safe,no_disaster,flood,fire,earthquake,hurricane"
default_text_labels = "other,food,shelter,rescue,medical"
image_label_input = st.sidebar.text_input("Image class labels (comma-separated)", value=default_image_labels)
text_label_input = st.sidebar.text_input("Text category labels (comma-separated)", value=default_text_labels)

# parse label sets
IMAGE_LABELS = [s.strip() for s in image_label_input.split(",") if s.strip() != ""]
TEXT_LABELS = [s.strip() for s in text_label_input.split(",") if s.strip() != ""]

st.sidebar.markdown("---")
text_maxlen = st.sidebar.number_input("Text max length", min_value=10, max_value=1000, value=DEFAULT_TEXT_MAXLEN, step=10)
st.sidebar.markdown("Fusion weights (image vs text)")
img_weight = st.sidebar.slider("Image weight", 0.0, 1.0, 0.6, 0.05)
txt_weight = st.sidebar.slider("Text weight", 0.0, 1.0, 0.4, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("**GCC integration**: placeholder only — replace `send_gcc_alert` with real API/SDK call when available.")

# ---------------- Model loading (cached) ----------------
@st.cache_resource(show_spinner=False)
def load_keras_model(path, uploaded=None):
    try:
        if uploaded is not None:
            tmp = f"tmp_{os.path.basename(path)}"
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

# ---------------- Helpers: preprocess + predict ----------------
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
    # handle binary or multiclass
    if pred.size == 1:
        p = float(pred)
        return np.array([1 - p, p])
    probs = pred / pred.sum() if pred.sum() > 0 else np.zeros_like(pred)
    return probs

def text_predict_probs(text, tok, model, maxlen=100):
    if model is None or tok is None or not text or text.strip() == "":
        return None
    seq = tok.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=int(maxlen), padding="post", truncating="post")
    pred = model.predict(padded)
    pred = np.array(pred).squeeze()
    if pred.size == 1:
        p = float(pred)
        return np.array([1 - p, p])
    probs = pred / pred.sum() if pred.sum() > 0 else np.zeros_like(pred)
    return probs

def pretty_label_from_probs(probs, label_list):
    """
    Return (label, confidence, index).
    If label_list shorter than probs, use fallback 'Class X' names.
    """
    idx = int(np.argmax(probs))
    label = label_list[idx] if idx < len(label_list) else f"Class_{idx+1}"
    return label, float(probs[idx]), idx

# ---------------- GCC alert placeholder ----------------
def send_gcc_alert(alert_payload: dict):
    """
    Placeholder function: replace internals with real GCC integration.
    For now it logs the payload and returns a simulated success response.
    """
    # In production: POST to GCC endpoint, use API key, handle retries/logging.
    # Example (pseudocode):
    #   requests.post(GCC_URL, json=alert_payload, headers={"Authorization": "Bearer ..."})
    st.info("Simulated send to GCC — alert payload recorded in logs.")
    # write to local file for demo / logs
    try:
        logname = "gcc_alert_log.jsonl"
        with open(logname, "a") as f:
            f.write(json.dumps(alert_payload) + "\n")
    except Exception:
        pass
    return {"status": "ok", "sent_at": datetime.utcnow().isoformat() + "Z"}

# ---------------- Input UI ----------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
col_img, col_text = st.columns([0.55, 0.45])

with col_img:
    st.markdown('<div class="input-label">🖼️ Image Input</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-muted">Upload a JPG/PNG or use the camera (browser-dependent).</div>', unsafe_allow_html=True)
    img_upload = st.file_uploader("Upload image", type=["jpg","jpeg","png"], key="img_uploader")
    cam = st.camera_input("Use camera (optional)", key="cam_input")
    image_file = img_upload if img_upload is not None else cam
    if image_file is not None:
        try:
            pil_image = Image.open(image_file).convert("RGB")
            st.image(pil_image, use_column_width=True, caption="Preview")
        except Exception as e:
            st.error(f"Cannot open image: {e}")
            pil_image = None
    else:
        pil_image = None
        st.info("No image selected.")

with col_text:
    st.markdown('<div class="input-label">💬 Text Input</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-muted">Enter an eyewitness message, tweet, or short report.</div>', unsafe_allow_html=True)
    user_text = st.text_area("Text (optional)", height=220, placeholder="e.g., 'Fire near the market', 'Flooding on 4th street'")

st.markdown('</div>', unsafe_allow_html=True)

# Controls
controls_col1, controls_col2, controls_col3 = st.columns([0.32, 0.32, 0.36])
with controls_col1:
    analyze_btn = st.button("🔎 Analyze & Fuse", use_container_width=True)
with controls_col2:
    clear_btn = st.button("Reset / Reload", use_container_width=True)
with controls_col3:
    st.markdown("<div class='small-muted'>Tune fusion weights in sidebar; configure labels if your model uses custom ordering.</div>", unsafe_allow_html=True)

if clear_btn:
    st.experimental_rerun()

# ---------------- Results area ----------------
st.markdown('<div class="result-card">', unsafe_allow_html=True)

if analyze_btn:
    if (pil_image is None) and (not user_text or user_text.strip() == ""):
        st.warning("Please provide an image or some text to analyze.")
    else:
        with st.spinner("Running models..."):
            img_probs = image_predict_probs(pil_image, image_model) if pil_image is not None else None
            txt_probs = text_predict_probs(user_text, tokenizer, text_model, maxlen=text_maxlen) if user_text and user_text.strip() != "" else None

        # normalize fusion weights
        total_w = img_weight + txt_weight
        if total_w <= 0:
            w_img, w_txt = 0.5, 0.5
        else:
            w_img, w_txt = img_weight / total_w, txt_weight / total_w

        # Fuse: upcast to same length = max len among available and provided label lists
        fused_probs = None
        if img_probs is not None and txt_probs is not None:
            target_len = max(len(img_probs), len(txt_probs), len(IMAGE_LABELS), len(TEXT_LABELS))
            def upcast(p, L):
                p = np.array(p)
                out = np.zeros(L)
                out[:min(len(p), L)] = p[:min(len(p), L)]
                if out.sum() > 0:
                    out = out / out.sum()
                return out
            img_u = upcast(img_probs, target_len)
            txt_u = upcast(txt_probs, target_len)
            fused_probs = w_img * img_u + w_txt * txt_u
        elif img_probs is not None:
            fused_probs = img_probs
            st.info("Using image-only prediction (text not provided or not available).")
        elif txt_probs is not None:
            fused_probs = txt_probs
            st.info("Using text-only prediction (image not provided or not available).")

        if fused_probs is None:
            st.error("No model outputs available; ensure models/tokenizer are loaded.")
        else:
            # normalize
            if fused_probs.sum() > 0:
                fused_probs = fused_probs / fused_probs.sum()
            # derive label and confidence (prefer IMAGE_LABELS for naming if sizes match)
            # If target corresponds to image classes, use IMAGE_LABELS; else fallback
            use_labels = IMAGE_LABELS if len(IMAGE_LABELS) >= len(fused_probs) else [f"Class_{i+1}" for i in range(len(fused_probs))]
            label, conf, idx = pretty_label_from_probs(fused_probs, use_labels)

            # Show big result
            c1, c2 = st.columns([0.65, 0.35])
            with c1:
                st.markdown(f"### ✅ Final Assessment: **{label.replace('_',' ').title()}**")
                st.metric("Confidence", f"{conf:.3f}")
                st.write("**Top class probabilities (showing > 1% only):**")
                for i, p in enumerate(fused_probs):
                    if p < 0.01:
                        continue
                    name = use_labels[i] if i < len(use_labels) else f"Class_{i+1}"
                    st.write(f"- {name.replace('_',' ').title()}: {p:.3f}")
            with c2:
                disaster_index = None
                # attempt to find index labeled as 'disaster' or check second position if binary
                # We'll treat a class as 'disaster' if its label contains keywords or if binary second class is high
                keywords = ["disaster","fire","flood","earthquake","hurricane","storm"]
                for i, lbl in enumerate(use_labels):
                    if any(k in lbl.lower() for k in keywords):
                        disaster_index = i
                        break
                if disaster_index is None and len(fused_probs) >= 2:
                    disaster_index = 1  # fallback to second class
                disaster_prob = float(fused_probs[disaster_index]) if disaster_index is not None else 0.0
                st.progress(min(max(disaster_prob, 0.0), 1.0))
                st.caption("Disaster likelihood")

            # Show advisory message for disaster detection and suggest sending GCC alert
            advisory = None
            # Decide disaster if confidence high or detection label suggests disaster
            disaster_flag = False
            if any(k in label.lower() for k in ["fire","flood","earthquake","hurricane","storm","disaster"]) or conf > 0.7:
                disaster_flag = True

            if disaster_flag:
                # context from text (if present)
                contextual = ""
                if user_text:
                    text_l = user_text.lower()
                    if "fire" in text_l:
                        contextual = "🔥 Fire detected — keep distance from flames and call emergency services."
                    elif "flood" in text_l:
                        contextual = "🌊 Flooding reported — move to higher ground and avoid floodwater."
                    elif "earthquake" in text_l:
                        contextual = "🏚️ Possible earthquake — drop, cover, and hold on; move away from windows."
                    elif "storm" in text_l or "cyclone" in text_l or "hurricane" in text_l:
                        contextual = "🌪️ Severe storm — secure indoors, avoid travel, follow official advisories."
                if contextual == "":
                    # generic advice depending on label string
                    low = label.lower()
                    if "fire" in low:
                        contextual = "🔥 Fire detected — stay away from flames and evacuate if instructed."
                    elif "flood" in low:
                        contextual = "🌊 Flood detected — move to higher ground and avoid walking through water."
                    else:
                        contextual = "⚠️ Potential disaster detected — follow local safety instructions and await official guidance."

                st.markdown(f"**Advisory:** {contextual}")
                # Offer to send GCC alert
                if st.button("🚨 Send Alert via GCC (simulated)"):
                    payload = {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "label": label,
                        "confidence": conf,
                        "image_present": pil_image is not None,
                        "text_snippet": (user_text[:240] + "...") if user_text and len(user_text) > 240 else (user_text or ""),
                        "image_labels": IMAGE_LABELS,
                        "text_labels": TEXT_LABELS,
                        "fused_probs": fused_probs.tolist(),
                    }
                    resp = send_gcc_alert(payload)
                    st.success(f"Alert sent (simulated). Response: {resp.get('status')} at {resp.get('sent_at')}")
            else:
                st.success("✅ Area appears safe / no immediate disaster detected.")

            # Fusion breakdown: show image/text probs for the first few classes if available
            st.markdown("**Fusion breakdown**")
            st.write(f"- Image weight used: {w_img:.2f}")
            st.write(f"- Text weight used: {w_txt:.2f}")
            # show the first N labels from IMAGE_LABELS or TEXT_LABELS (best-effort)
            N = min(6, len(fused_probs))
            if img_probs is not None:
                show_labels = IMAGE_LABELS if len(IMAGE_LABELS) >= len(img_probs) else [f"Class_{i+1}" for i in range(len(img_probs))]
                display_img_probs = [f"{show_labels[i]}: {img_probs[i]:.3f}" for i in range(min(len(img_probs), N)) if img_probs[i] >= 0.001]
                st.write("📷 Image probabilities:", ", ".join(display_img_probs) if display_img_probs else "—")
            if txt_probs is not None:
                show_labels = TEXT_LABELS if len(TEXT_LABELS) >= len(txt_probs) else [f"Class_{i+1}" for i in range(len(txt_probs))]
                display_txt_probs = [f"{show_labels[i]}: {txt_probs[i]:.3f}" for i in range(min(len(txt_probs), N)) if txt_probs[i] >= 0.001]
                st.write("📝 Text probabilities:", ", ".join(display_txt_probs) if display_txt_probs else "—")

else:
    st.markdown("### Ready to analyze")
    st.write("Upload an image and/or enter text, then click **Analyze & Fuse**. Use the sidebar to configure labels and fusion weights.")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Footer / Deployment note ----------------
st.markdown("---")
st.header("About & Next Steps")
st.markdown(
    """
This app implements the core objectives:
- Disaster image classification, emergency message classification, and an integrated fusion-based response system.
- Aligns with SDG goals by enabling rapid aid and resilient community responses.
- Designed for scalability with future integration to drones, satellites, and IoT.

**GCC integration note:** `send_gcc_alert` is currently a placeholder. When you integrate with the actual GCC API/SDK, replace the function body with the HTTP/SDK call (securely storing API keys) and add retries/logging.

If you want, I can now:
- Add a **map** placeholder that pins the approximate location (if you provide geolocation in text), or
- Replace the GCC placeholder with a real example using (for example) an HTTP webhook or cloud messaging service (you provide the endpoint/credentials).
"""
)

# ---------------- End of app.py ----------------
