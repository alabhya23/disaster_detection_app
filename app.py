import streamlit as st
import numpy as np
from PIL import Image

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
        "no_disaster": "✅ Area appears safe — no signs of disaster detected."
    }
    return messages.get(disaster_type, "⚠️ Situation uncertain — stay alert and follow safety protocols.")

def fuse_predictions(image_probs, text_probs, image_weight=0.6, text_weight=0.4):
    image_probs = np.array(image_probs)
    text_probs = np.array(text_probs)
    fused_probs = (image_weight * image_probs) + (text_weight * text_probs)

    image_label_idx = np.argmax(image_probs)
    text_label_idx = np.argmax(text_probs)
    fused_label_idx = np.argmax(fused_probs)

    # If either model predicts disaster, mark final as disaster
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
    """
    This intelligent multi-modal system analyzes **images** and **text** to detect potential disasters.  
    It uses a **CNN image model** and an **NLP text model** together to provide quick and reliable disaster detection.
    """
)

# --- Inputs Section ---
col1, col2 = st.columns(2)
with col1:
    st.header("🖼️ Image Input")
    uploaded_image = st.file_uploader("Upload a disaster-related image", type=["jpg", "jpeg", "png"])
    camera_image = st.camera_input("Or capture live image")

    # Prioritize camera input if available
    if camera_image is not None:
        image_file = camera_image
    else:
        image_file = uploaded_image

    if image_file:
        st.image(image_file, caption="Input Image", use_container_width=True)

with col2:
    st.header("💬 Text Input")
    user_text = st.text_area("Enter eyewitness report, tweet, or message", placeholder="e.g., Heavy flooding in the area...")

# --- Analyze Button ---
if st.button("🔍 Analyze Situation"):
    if not image_file and not user_text.strip():
        st.warning("Please upload an image or enter text for analysis.")
    else:
        # Placeholder model predictions (replace with your model outputs)
        LABELS = ["no_disaster", "disaster"]
        disaster_types = ["no_disaster", "fire", "flood", "earthquake", "hurricane"]

        # Simulated model outputs (replace with actual predictions)
        if image_file:
            image_probs = [0.10, 0.90]  # 90% disaster (e.g., flood/fire)
        else:
            image_probs = [1.00, 0.00]

        if user_text.strip():
            text_probs = [0.70, 0.30]  # 30% disaster
        else:
            text_probs = [1.00, 0.00]

        # Compute labels
        img_label, img_conf = probs_to_label(image_probs, LABELS)
        txt_label, txt_conf = probs_to_label(text_probs, LABELS)

        # Fusion
        fused_label_idx, fused_probs = fuse_predictions(image_probs, text_probs)
        fused_label, fused_conf = probs_to_label(fused_probs, LABELS)

        # Choose disaster type (mock logic — replace with real classification)
        disaster_type = "no_disaster"
        if fused_label == "disaster":
            disaster_type = np.random.choice(["fire", "flood", "earthquake", "hurricane"])

        advisory = get_advisory_message(disaster_type)

        # ----------------------------- #
        # Display results
        # ----------------------------- #
        st.divider()
        st.subheader("🧭 Model Assessments")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("### 🖼️ Image Model Result")
            if image_file:
                st.metric("Prediction", img_label.replace("_", " ").title())
                st.metric("Confidence", f"{img_conf:.3f}")
            else:
                st.markdown("_No image provided_")

        with c2:
            st.markdown("### 💬 Text Model Result")
            if user_text.strip():
                st.metric("Prediction", txt_label.replace("_", " ").title())
                st.metric("Confidence", f"{txt_conf:.3f}")
            else:
                st.markdown("_No text provided_")

        with c3:
            st.markdown("### 🤖 Combined (Fused) Result")
            st.metric("Final Assessment", fused_label.replace("_", " ").title())
            st.metric("Confidence", f"{fused_conf:.3f}")

        st.divider()
        st.info(f"**Advisory:** {advisory}")

        # --- GCC Alert Message ---
        st.markdown(
            """
            > 📡 **GCC (Global Communication Channel):**  
            In case of a detected disaster, the system immediately transmits alerts through the GCC network — 
            notifying nearby users, emergency response teams, and authorities to ensure rapid and coordinated action.
            """
        )

# --- About Section ---
st.divider()
st.markdown(
    """
    ### ℹ️ About the App  
    The **AI-Powered Disaster Aid System** integrates computer vision (CNN) and natural language processing (LSTM)  
    to analyze real-time images and text reports from the field. By combining both data sources, it detects disasters early  
    and sends timely alerts using the **Global Communication Channel (GCC)** — helping people stay safe and enabling  
    faster government and NGO responses to critical events.
    """
)
