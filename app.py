# app.py
import streamlit as st
import numpy as np
import os
import io
import pickle
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Disaster Detection — Multi-Modal (Image + Text)", layout="wide")

# ---------------- USER CONFIG (edit if needed) ----------------
IMAGE_TARGET_SIZE = (224, 224)   # typical MobileNetV2 input
DEFAULT_TEXT_MAXLEN = 100        # change if your text model used different value
# Default labels (order matters for multiclass). Change to your own if different.
LABELS = ["not_disaster", "disaster"]
# ----------------------------------------------------------------

st.title("🚨 Disaster Detection — Image + Text (Multi-Modal fusion)")
st.markdown(
    """
This app classifies **images** and **short text** then **fuses** the predictions.
- Place your models in the repo root OR upload them via the sidebar.
- Supported model types: Keras `.h5` models for image & text, and a pickled tokenizer `.pkl`.
"""
)

# Sidebar: model files & options
st.sidebar.header("Model files & fusion options")
IMAGE_MODEL_PATH = "disaster_cnn_mobilenet_clean.h5"
TEXT_MODEL_PATH = "disaster_text_bilstm.h5"
TOKENIZER_PATH = "tokenizer.pkl"

st.sidebar.write("Model file locations (checked on disk):")
st.sidebar.write(f"- Image model: `{IMAGE_MODEL_PATH}` — {'FOUND' if os.path.exists(IMAGE_MODEL_PATH) else 'missing'}")
st.sidebar.write(f"- Text model: `{TEXT_MODEL_PATH}` — {'FOUND' if os.path.exists(TEXT_MODEL_PATH) else 'missing'}")
st.sidebar.write(f"- Tokenizer: `{TOKENIZER_PATH}` — {'FOUND' if os.path.exists(TOKENIZER_PATH) else 'missing'}")

st.sidebar.markdown("---")
# allow upload of models if not present / user prefers upload
uploaded_image_model = st.sidebar.file_uploader("Upload `disaster_cnn_mobilenet_clean.h5` (optional)", type=["h5"])
uploaded_text_model = st.sidebar.file_uploader("Upload `disaster_text_bilstm.h5` (optional)", type=["h5"])
uploaded_tokenizer = st.sidebar.file_uploader("Upload `tokenizer.pkl` (optional)", type=["pkl","pickle","dat"])

st.sidebar.markdown("---")
text_maxlen = st.sidebar.number_input("Text max length (pad/truncate)", min_value=10, max_value=1000, value=DEFAULT_TEXT_MAXLEN, step=10)

st.sidebar.markdown("---")
st.sidebar.subheader("Fusion weights")
img_weight = st.sidebar.slider("Image weight", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
txt_weight = st.sidebar.slider("Text weight", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
if abs(img_weight + txt_weight - 1.0) > 1e-6:
    st.sidebar.info("Weights will be normalized so they sum to 1.")

# --------------- Model loading with caching -----------------
@st.cache_resource(show_spinner=False)
def load_image_model(path=IMAGE_MODEL_PATH, uploaded=None):
    """
    If uploaded is a Streamlit UploadedFile (BytesIO-like), save to temp and load.
    """
    try:
        if uploaded is not None:
            temp = "temp_image_model.h5"
            with open(temp, "wb") as f:
                f.write(uploaded.getbuffer())
            model = load_model(temp)
            os.remove(temp)
            return model
        if os.path.exists(path):
            return load_model(path)
    except Exception as e:
        st.error(f"Error loading image model: {e}")
    return None

@st.cache_resource(show_spinner=False)
def load_text_model(path=TEXT_MODEL_PATH, uploaded=None):
    try:
        if uploaded is not None:
            temp = "temp_text_model.h5"
            with open(temp, "wb") as f:
                f.write(uploaded.getbuffer())
            model = load_model(temp)
            os.remove(temp)
            return model
        if os.path.exists(path):
            return load_model(path)
    except Exception as e:
        st.error(f"Error loading text model: {e}")
    return None

@st.cache_resource(show_spinner=False)
def load_tokenizer(path=TOKENIZER_PATH, uploaded=None):
    try:
        if uploaded is not None:
            uploaded.seek(0)
            tok = pickle.load(uploaded)
            return tok
        if os.path.exists(path):
            with open(path, "rb") as f:
                tok = pickle.load(f)
            return tok
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
    return None

# load models (from disk or uploaded files)
image_model = load_image_model(uploaded=uploaded_image_model) if uploaded_image_model is not None else load_image_model()
text_model = load_text_model(uploaded=uploaded_text_model) if uploaded_text_model is not None else load_text_model()
tokenizer = load_tokenizer(uploaded=uploaded_tokenizer) if uploaded_tokenizer is not None else load_tokenizer()

# helper to detect if a Keras model is binary (last dim == 1 -> sigmoid)
def model_is_binary(model):
    if model is None:
        return None
    try:
        out_shape = model.output_shape
        last = out_shape[-1]
        # allow possibility of (None, 1) or a scalar output
        return int(last) == 1
    except Exception:
        return None

image_is_binary = model_is_binary(image_model)
text_is_binary = model_is_binary(text_model)

# ------------------ UI: Inputs ------------------
col1, col2 = st.columns([1,1])

with col1:
    st.header("Image input")
    # camera_input works well in browser (if available)
    uploaded_image = st.file_uploader("Upload an image (jpg/png)", type=["jpg","jpeg","png"])
    camera_img = st.camera_input("Or take a picture (camera_input)")
    image_file = uploaded_image if uploaded_image is not None else camera_img
    if image_file is not None:
        try:
            pil_img = Image.open(image_file).convert("RGB")
            st.image(pil_img, caption="Input image", use_column_width=True)
        except Exception as e:
            st.error(f"Unable to open image: {e}")
            pil_img = None
    else:
        pil_img = None

with col2:
    st.header("Text input")
    user_text = st.text_area("Paste / type a short description (tweet, message, report)", height=200)
    st.caption("Text classification disabled unless tokenizer + text model are loaded.")

st.markdown("---")
# show model availability
availability_cols = st.columns(3)
with availability_cols[0]:
    st.subheader("Image model")
    if image_model is not None:
        st.success("Loaded ✅")
        st.write(f"Model type: {'binary' if image_is_binary else 'multi-class' if image_is_binary==False else 'unknown'}")
    else:
        st.error("Not loaded")

with availability_cols[1]:
    st.subheader("Text model")
    if text_model is not None and tokenizer is not None:
        st.success("Loaded ✅")
        st.write(f"Model type: {'binary' if text_is_binary else 'multi-class' if text_is_binary==False else 'unknown'}")
    elif (text_model is None) and (tokenizer is None):
        st.info("Text model & tokenizer not loaded")
    elif text_model is None:
        st.warning("Tokenizer found but model missing" if tokenizer else "Text model missing")
    elif tokenizer is None:
        st.warning("Text model found but tokenizer missing")

with availability_cols[2]:
    st.subheader("Notes")
    st.write("- If your models used different label names/order, edit LABELS at top.")
    st.write("- Adjust text maxlen & fusion weights in the sidebar.")

# ---------- Preprocessing & prediction helpers ----------
def preprocess_image_for_model(pil_img, target_size=IMAGE_TARGET_SIZE):
    img = pil_img.resize(target_size)
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

def image_predict_probs(pil_img, model):
    """
    Returns a probability vector over classes (length = num_classes).
    For binary models (sigmoid, output shape (None,1)) -> returns [1-p, p]
    For multiclass (softmax) -> returns probs vector
    """
    if model is None or pil_img is None:
        return None
    x = preprocess_image_for_model(pil_img)
    pred = model.predict(x)
    pred = np.array(pred).squeeze()
    if pred.ndim == 0:  # scalar
        p = float(pred)
        return np.array([1.0 - p, p])
    if pred.size == 1:
        p = float(pred)
        return np.array([1.0 - p, p])
    # multiclass softmax
    probs = pred / np.sum(pred)
    return probs

def text_predict_probs(text, tokenizer, model, maxlen=DEFAULT_TEXT_MAXLEN):
    if model is None or tokenizer is None or (not text or text.strip() == ""):
        return None
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=int(maxlen), padding="post", truncating="post")
    pred = model.predict(padded)
    pred = np.array(pred).squeeze()
    if pred.ndim == 0 or pred.size == 1:
        p = float(pred)
        return np.array([1.0 - p, p])
    probs = pred / np.sum(pred)
    return probs

def probs_to_label(probs, labels=LABELS):
    idx = int(np.argmax(probs))
    label = labels[idx] if idx < len(labels) else f"class_{idx}"
    score = float(probs[idx])
    return label, score, idx

# ---------- Run predictions & fusion ----------
st.markdown("### Run predictions")
run = st.button("Predict & Fuse")

result_area = st.container()

if run:
    with result_area:
        st.subheader("Results")
        # show warnings if inputs missing
        if pil_img is None and (not user_text or user_text.strip() == ""):
            st.warning("Provide at least an image or some text to predict.")
        else:
            # compute weights normalized
            total = img_weight + txt_weight
            if total == 0:
                w_img = 0.5
                w_txt = 0.5
            else:
                w_img = img_weight / total
                w_txt = txt_weight / total

            # Image prediction
            if pil_img is not None and image_model is not None:
                with st.spinner("Predicting (image)..."):
                    try:
                        img_probs = image_predict_probs(pil_img, image_model)
                        img_label, img_score, img_idx = probs_to_label(img_probs)
                        st.markdown("**Image model**")
                        st.write(f"- Predicted label: `{img_label}` (confidence {img_score:.3f})")
                        st.write(f"- Probabilities: {np.round(img_probs, 4)}")
                    except Exception as e:
                        st.error(f"Image prediction error: {e}")
                        img_probs = None
            else:
                img_probs = None
                if pil_img is None:
                    st.info("No image provided.")
                else:
                    st.warning("Image model not loaded; image prediction skipped.")

            # Text prediction
            if user_text and user_text.strip() != "" and text_model is not None and tokenizer is not None:
                with st.spinner("Predicting (text)..."):
                    try:
                        txt_probs = text_predict_probs(user_text, tokenizer, text_model, maxlen=text_maxlen)
                        txt_label, txt_score, txt_idx = probs_to_label(txt_probs)
                        st.markdown("**Text model**")
                        st.write(f"- Predicted label: `{txt_label}` (confidence {txt_score:.3f})")
                        st.write(f"- Probabilities: {np.round(txt_probs, 4)}")
                    except Exception as e:
                        st.error(f"Text prediction error: {e}")
                        txt_probs = None
            else:
                txt_probs = None
                if not user_text or user_text.strip() == "":
                    st.info("No text provided.")
                else:
                    st.warning("Text model/tokenizer not loaded; text prediction skipped.")

            # Fusion logic
            st.markdown("**Fused (multi-modal) prediction**")
            # Case handling:
            # - If both probs available: normalize lengths by expanding binary->2-vector or multiclass->as-is.
            # - If only one available: fused = that one.
            fused_probs = None
            if (img_probs is not None) and (txt_probs is not None):
                # if shapes differ (e.g., multiclass vs binary), we will attempt to align by padding or truncation.
                # Best case: both have same number of classes. If not, we will handle common-case binary vs binary or binary vs multiclass where LABELS length is 2.
                if img_probs.shape == txt_probs.shape:
                    fused_probs = w_img * img_probs + w_txt * txt_probs
                else:
                    # try to coerce both to length = len(LABELS) (default 2). This handles binary vs multiclass where multiclass==2 too.
                    target_len = max(img_probs.size, txt_probs.size, len(LABELS))
                    def upcast(p, target_len):
                        if p.size == target_len:
                            return p
                        if p.size == 1:  # single probability (rare)
                            return np.array([1 - float(p), float(p)]) if target_len == 2 else np.concatenate(([1 - float(p)], np.zeros(target_len-2), [float(p)]))
                        if p.size == 2 and target_len > 2:
                            # pad zeros for other classes
                            return np.concatenate([p, np.zeros(target_len-2)])
                        if p.size > target_len:
                            # truncate (not ideal) but pick top target_len probabilities
                            idxs = np.argsort(p)[-target_len:]
                            new = np.zeros(target_len)
                            # place these probabilities compressed (approximate)
                            new[:len(idxs)] = p[idxs]
                            new = new / np.sum(new) if new.sum() > 0 else new
                            return new
                        # fallback: normalize then pad zeros
                        res = np.zeros(target_len)
                        res[:p.size] = p
                        return res
                    img_up = upcast(img_probs, target_len)
                    txt_up = upcast(txt_probs, target_len)
                    fused_probs = w_img * img_up + w_txt * txt_up

            elif img_probs is not None:
                fused_probs = img_probs
                st.info("Only image prediction available — fused result equals image model output.")
            elif txt_probs is not None:
                fused_probs = txt_probs
                st.info("Only text prediction available — fused result equals text model output.")
            else:
                st.error("No model outputs available to fuse.")
                fused_probs = None

            if fused_probs is not None:
                fused_probs = np.array(fused_probs)
                # normalize to sum=1
                if fused_probs.sum() > 0:
                    fused_probs = fused_probs / fused_probs.sum()
                fused_label, fused_score, fused_idx = probs_to_label(fused_probs, labels=LABELS if len(LABELS) >= fused_probs.size else [f"class_{i}" for i in range(fused_probs.size)])
                st.write(f"- **Fused label:** `{fused_label}` (confidence {fused_score:.3f})")
                st.write(f"- **Fused probabilities:** {np.round(fused_probs, 4)}")
                # Show breakdown
                st.markdown("**Fusion breakdown**")
                st.write(f"- Image weight: {w_img:.2f}, Text weight: {w_txt:.2f}")
                if img_probs is not None:
                    st.write(f"- Image probs used: {np.round(img_probs,4)}")
                if txt_probs is not None:
                    st.write(f"- Text probs used: {np.round(txt_probs,4)}")

# ---------------- Deployment / requirements hint ----------------
st.markdown("---")
st.header("Deployment & requirements")
st.markdown
