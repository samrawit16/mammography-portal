# Mammography Diagnostic Portal - coursework prototype (NOT a medical device)
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import time

st.set_page_config(
    page_title="Mammography Triage Prototype",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    #MainMenu, footer {visibility: hidden;}

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .stApp {
        background: radial-gradient(circle at 10% 0%, #14161f 0%, #0d0e14 55%, #0a0b10 100%);
    }

    .hero {
        padding: 2rem 2.2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(124,58,237,0.18), rgba(6,182,212,0.10));
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1.6rem;
    }
    .hero h1 {
        font-size: 2.1rem;
        margin: 0 0 0.4rem 0;
        background: linear-gradient(90deg, #a78bfa, #67e8f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        color: #b8bcc8;
        font-size: 0.98rem;
        margin: 0.15rem 0;
    }

    .disclaimer {
        border-radius: 14px;
        padding: 0.9rem 1.2rem;
        background: rgba(234,179,8,0.10);
        border: 1px solid rgba(234,179,8,0.35);
        color: #fde68a;
        font-size: 0.88rem;
        margin-bottom: 1.4rem;
    }

    .card {
        border-radius: 18px;
        padding: 1.4rem 1.5rem;
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        backdrop-filter: blur(6px);
        margin-bottom: 1rem;
    }

    .badge {
        display: inline-block;
        padding: 0.45rem 1.1rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: 0.03em;
    }
    .badge-benign {
        background: rgba(16,185,129,0.15);
        color: #34d399;
        border: 1px solid rgba(16,185,129,0.4);
    }
    .badge-malignant {
        background: rgba(239,68,68,0.15);
        color: #f87171;
        border: 1px solid rgba(239,68,68,0.4);
    }
    .badge-unsure {
        background: rgba(234,179,8,0.15);
        color: #fbbf24;
        border: 1px solid rgba(234,179,8,0.4);
    }

    .conf-track {
        width: 100%;
        height: 14px;
        border-radius: 999px;
        background: rgba(255,255,255,0.07);
        overflow: hidden;
        margin: 0.6rem 0 0.2rem 0;
    }
    .conf-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 1s ease-out;
    }

    .stButton>button {
        width: 100%;
        border-radius: 12px;
        padding: 0.7rem 0;
        font-weight: 700;
        font-size: 1.02rem;
        background: linear-gradient(90deg, #7c3aed, #06b6d4);
        color: white;
        border: none;
        box-shadow: 0 4px 18px rgba(124,58,237,0.35);
        transition: transform 0.15s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(124,58,237,0.5);
    }

    section[data-testid="stSidebar"] {
        background: #0e0f16;
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    [data-testid="stFileUploaderDropzone"] {
        border-radius: 16px !important;
        border: 1.5px dashed rgba(167,139,250,0.45) !important;
        background: rgba(124,58,237,0.05) !important;
    }

    div[data-testid="stImage"] img {
        border-radius: 14px;
        max-height: 420px;
        width: auto !important;
        display: block;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="hero">
    <h1>Breast Mammography Triage Prototype</h1>
    <p><b>What this does:</b> classifies a cropped mammography finding (breast X-ray) as <b>BENIGN</b> or <b>MALIGNANT</b>.</p>
    <p><b>Scope:</b> mammography images of breast tissue (CBIS-DDSM-style cropped lesion views), resized to 224×224.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Model")
    st.markdown(
        "**Architecture:** MobileNetV2 backbone (frozen) + trainable classification head\n\n"
        "**Input:** 224×224 RGB mammography crop\n\n"
        "**Output:** BENIGN / MALIGNANT probability\n\n"
        "**Trained on:** CBIS-DDSM"
    )
    st.markdown("---")
    st.markdown("### How to use")
    st.markdown(
        "1. Upload a mammography image\n"
        "2. Click **Analyze Image**\n"
        "3. Review the prediction & confidence"
    )

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "mammography_model.keras",
        custom_objects={"preprocess_input": preprocess_input}
    )

model = load_model()
class_names = ["BENIGN", "MALIGNANT"]

col_left, col_right = st.columns([1, 1.15], gap="large")

with col_left:
    st.markdown("####  Upload")
    uploaded = st.file_uploader(
        "Upload a mammography image (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    img = None
    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded image")
        analyze_clicked = st.button("🔍  Analyze Image", type="primary")
    else:
        analyze_clicked = False
        st.session_state.pop("last_result", None)
        st.info("Upload an image to enable analysis.")

with col_right:
    st.markdown("#### Result")

    if uploaded is None:
        st.markdown(
            '<div class="card" style="text-align:center; color:#6b7280; padding:3rem 1rem;">'
            'Waiting for an image…</div>',
            unsafe_allow_html=True,
        )
    elif not analyze_clicked and "last_result" not in st.session_state:
        st.markdown(
            '<div class="card" style="text-align:center; color:#9ca3af; padding:3rem 1rem;">'
            'Image ready — click <b>Analyze Image</b> to run the model.</div>',
            unsafe_allow_html=True,
        )

    if analyze_clicked:
        with st.spinner("Running model inference…"):
            arr = np.array(img.resize((224, 224)), dtype=np.float32)
            batch = np.expand_dims(arr, axis=0)
            prob_malignant = float(model.predict(batch, verbose=0)[0][0])
            time.sleep(0.3)
        st.session_state["last_result"] = prob_malignant

    if "last_result" in st.session_state and uploaded is not None:
        prob_malignant = st.session_state["last_result"]
        pred = 1 if prob_malignant > 0.5 else 0
        conf = max(prob_malignant, 1 - prob_malignant) * 100
        unsure = 0.4 < prob_malignant < 0.6

        if unsure:
            badge_class, badge_icon = "badge-unsure", ""
            bar_color = "#fbbf24"
        elif pred == 1:
            badge_class, badge_icon = "badge-malignant", "🔴"
            bar_color = "#f87171"
        else:
            badge_class, badge_icon = "badge-benign", "🟢"
            bar_color = "#34d399"

        st.markdown(f"""
        <div class="card">
            <span class="badge {badge_class}">{badge_icon} {class_names[pred]}</span>
            <div style="margin-top:1rem; color:#d1d5db; font-size:0.95rem;">Confidence</div>
            <div class="conf-track">
                <div class="conf-fill" style="width:{conf:.1f}%; background:{bar_color};"></div>
            </div>
            <div style="text-align:right; color:#9ca3af; font-size:0.85rem;">{conf:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        if pred == 1:
            meaning = (
                f"The model predicts this mammogram shows signs that may be **malignant** "
                f"({prob_malignant * 100:.0f}% confidence). This means only that the pattern "
                f"resembles cancerous lesions in the training data — it is **NOT a diagnosis**."
            )
        else:
            meaning = (
                f"The model predicts this finding looks **benign** "
                f"({(1 - prob_malignant) * 100:.0f}% confidence), i.e. it does not resemble the "
                f"malignant patterns it was trained on. This is **NOT a diagnosis**."
            )
        st.markdown(f'<div class="card">{meaning}</div>', unsafe_allow_html=True)

        if unsure:
            st.markdown("""
            <div class="disclaimer">
                The model is <b>unsure</b> on this image (probability close to 0.5).
                The image may be unlike anything in its training data, or the finding may be
                genuinely borderline.Seek expert review.
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Raw model output"):
            st.write({
                "P(malignant)": round(prob_malignant, 4),
                "P(benign)": round(1 - prob_malignant, 4),
                "predicted_class": class_names[pred],
            })