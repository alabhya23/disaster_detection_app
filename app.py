import streamlit as st
import numpy as np

# ----------------------------- #
# Helper functions
# ----------------------------- #

def probs_to_label(probs, labels):
    idx = np.argmax(probs)
    return labels[idx], float(probs[idx])

def get_advisory_message(disaster_type):
    messages = {
        "fire": "🔥 Fire detected — stay away from flames and evacuate nearby areas immediately.",
        "flood": "🌊 Flooding reported — move to higher ground and avoid floodwater.",
        "earthquake": "🌍 Earthquake detected — stay in open areas and avoid tall structures.",
        "hurricane": "🌪️ Severe storm detected — take shelter and follow official alerts.",
        "no_disaster": "✅ Area is safe — no disaster detected at the moment."
    }
    return messages.get(disaster_type, "⚠️ Situation uncertain — stay alert and follow safety protocols.")

def fuse_predictions(image_probs, text_probs, image_weight=0.6, text_weight=0.4):
    image_probs = np.array(image_probs)
    text_probs = np.array(text_probs)
    fused_probs = (image_weight * image_probs) + (text_weight * text_probs)

    image_label_idx = np.argmax(image_probs)
    text_label_idx = np.argmax(text_probs)
    fused_label_idx = np.argmax(fused_probs)

    # If either predicts a disaster, mark final as disaster
    disaster_idx = 1  # assuming index 1 = disaster
    if image_label_idx == disaster_idx or text_label_idx == disaster_idx:
        fused_label_idx = disaster_idx

    return fused_label_idx, fused_probs

# ----------------------------- #
# Streamlit UI
# ----------------------------- #

st.set_page_config(page_title="AI-Powered Disaster Aid System", layout="wide")

st.title("🌍 AI-Powered Disaster Aid System")
st.markdown(
    "This intelligent system analyzes **images and text** to detect potential disasters. "
    "It combines a **CNN image model** and an **NLP text model** into a unified decision system."
)

# --- Inputs ---
col1, col2 = st.columns(2)
with col1:
    st.header("🖼️ Image Input")
    image_file = st.file_uploader("Upload a disaster-related image", type=["jpg", "jpeg", "png"])
    if image_file:
        st.image(image_file, caption="Uploaded Image", use_container_width=True)

with col2:
    st.header("💬 Text Input")
    user_text = st.text_area("Enter eyewitness report, tweet, or message", placeholder="e.g., heavy flooding in the area")

# --- Analyze Button ---
if st.button("🔍 Analyze Situation"):
    if image_file or user_text:
        # Mock probabilities for demonstration
        LABELS = ["no_disaster", "disaster"]
        image_probs = [0.05, 0.95]      # Example: CNN says 95% disaster
        text_probs = [0.80, 0.20]       # Example: NLP says 80% no disaster

        # Individual results
        img_label, img_conf = probs_to_label(image_probs, LABELS)
        txt_label, txt_conf = probs_to_label(text_probs, LABELS)

        # Fused result
        fused_label_idx, fused_probs = fuse_predictions(image_probs, text_probs)
        fused_label, fused_conf = probs_to_label(fused_probs, LABELS)

        # Decide disaster type (you can replace this with class-specific detection)
        disaster_type = "fire" if fused_label == "disaster" else "no_disaster"

        # Advisory
        advisory = get_advisory_message(disaster_type)

        # ----------------------------- #
        # Display results
        # ----------------------------- #
        st.divider()
        st.subheader("🧭 Model Assessments")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("### 🖼️ Image Model Result")
            st.metric("Prediction", img_label.replace("_", " ").title())
            st.metric("Confidence", f"{img_conf:.3f}")

        with c2:
            st.markdown("### 💬 Text Model Result")
            st.metric("Prediction", txt_label.replace("_", " ").title())
            st.metric("Confidence", f"{txt_conf:.3f}")

        with c3:
            st.markdown("### 🤖 Combined (Fused) Result")
            st.metric("Final Assessment", fused_label.replace("_", " ").title())
            st.metric("Confidence", f"{fused_conf:.3f}")

        st.divider()
        st.info(f"**Advisory:** {advisory}")

        st.markdown(
            "> 📡 Using GCC communication, alerts can be automatically sent to nearby users and authorities to ensure timely disaster response and community safety."
        )

    else:
        st.warning("Please upload an image or enter text for analysis.")
