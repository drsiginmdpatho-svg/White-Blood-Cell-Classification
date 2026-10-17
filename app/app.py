"""
app/app.py

Streamlit Web Application for Peripheral Blood Smear White Blood Cell (WBC) Classification.
Loads trained deep learning models without retraining, provides real-time inference,
confidence score visualizations, morphological descriptions, and curated test samples.
"""

import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from utils import load_model, load_class_metadata, predict_single_image

# Page configuration
st.set_page_config(
    page_title="WBC Classification AI - Capstone Prototype",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3799;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #4b6584;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #1e3799;
        margin-bottom: 1rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .badge-primary { background-color: #e8f0fe; color: #1967d2; }
    .badge-success { background-color: #e6f4ea; color: #137333; }
    .badge-warning { background-color: #fef7e0; color: #b06000; }
</style>
""", unsafe_allow_html=True)

# Cache model loading
@st.cache_resource
def get_cached_model(model_key: str):
    device = torch.device("cpu")
    return load_model(model_key, device=device)

metadata = load_class_metadata()
classes = metadata["classes"]
descriptions = metadata.get("descriptions", {})

CLASS_COLORS = {
    'Basophil': '#5c2d91',
    'Eosinophil': '#e056fd',
    'Lymphocyte': '#0984e3',
    'Monocyte': '#00b894',
    'Neutrophil': '#e17055'
}

# Session state initialization for sample selection
if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = None
if "is_failure_case" not in st.session_state:
    st.session_state.is_failure_case = False

# Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/microscope.png", width=64)
    st.title("Control Panel")
    
    st.subheader("1. Model Selection")
    model_choice = st.selectbox(
        "Choose Deep Learning Architecture:",
        options=["MobileNetV2 (Transfer Learning)", "Custom Baseline CNN"],
        index=0,
        help="MobileNetV2 leverages ImageNet pretrained features with fine-tuning. Baseline CNN is a 4-stage convolutional model trained from scratch."
    )
    model_key = "mobilenet_v2" if "MobileNetV2" in model_choice else "baseline_cnn"
    
    # Model Specs Callout
    if model_key == "mobilenet_v2":
        st.markdown("""
        <div class="metric-card">
            <b>Selected Model: MobileNetV2</b><br/>
            <span class="badge badge-success">Top Benchmark</span><br/>
            • Held-Out Accuracy: <b>91.20%</b><br/>
            • Macro F1-Score: <b>89.77%</b><br/>
            • Parameters: <b>2.39M</b><br/>
            • Inference: <b>~38 ms/image</b>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card">
            <b>Selected Model: Baseline CNN</b><br/>
            <span class="badge badge-primary">Custom Architecture</span><br/>
            • Held-Out Accuracy: <b>89.40%</b><br/>
            • Macro F1-Score: <b>84.47%</b><br/>
            • Parameters: <b>1.44M</b><br/>
            • Inference: <b>~38 ms/image</b>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("2. Curated Sample Gallery")
    st.caption("Click any button below to test immediate inference on held-out blood smear cells:")
    
    samples_dir = os.path.join(os.path.dirname(__file__), "samples")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("🧪 Basophil", use_container_width=True):
            st.session_state.selected_sample = os.path.join(samples_dir, "sample_basophil.jpg")
            st.session_state.is_failure_case = False
        if st.button("🛡️ Lymphocyte", use_container_width=True):
            st.session_state.selected_sample = os.path.join(samples_dir, "sample_lymphocyte.jpg")
            st.session_state.is_failure_case = False
        if st.button("⚡ Neutrophil", use_container_width=True):
            st.session_state.selected_sample = os.path.join(samples_dir, "sample_neutrophil.jpg")
            st.session_state.is_failure_case = False

    with col_s2:
        if st.button("🩸 Eosinophil", use_container_width=True):
            st.session_state.selected_sample = os.path.join(samples_dir, "sample_eosinophil.jpg")
            st.session_state.is_failure_case = False
        if st.button("🔬 Monocyte", use_container_width=True):
            st.session_state.selected_sample = os.path.join(samples_dir, "sample_monocyte.jpg")
            st.session_state.is_failure_case = False
        if st.button("⚠️ Failure Case", use_container_width=True, help="Authentic edge-case image misclassified by the model"):
            st.session_state.selected_sample = os.path.join(samples_dir, "known_failure_case.jpg")
            st.session_state.is_failure_case = True

    if st.session_state.selected_sample is not None:
        if st.button("🔄 Clear Active Sample", use_container_width=True):
            st.session_state.selected_sample = None
            st.session_state.is_failure_case = False
            st.rerun()

    st.markdown("---")
    st.markdown("""
    **Project Info:**  
    AI/ML Specialisation Capstone  
    *Peripheral Blood Smear WBC Classification*  
    Framework: **PyTorch 2.14** | **Streamlit**
    """)

# Main Content Area
st.markdown('<div class="main-title">Peripheral Blood Smear White Blood Cell Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI/ML Specialisation Capstone Project Prototype | Automated Five-Class Leukocyte Recognition</div>', unsafe_allow_html=True)

# Image Input Handling
uploaded_file = st.file_uploader(
    "Upload a cropped peripheral blood smear WBC image (JPG, JPEG, PNG):",
    type=["jpg", "jpeg", "png"],
    help="Select a cropped single-cell microscopy image to classify into one of the 5 WBC subtypes."
)

active_image = None
image_source = None

if uploaded_file is not None:
    try:
        active_image = Image.open(uploaded_file).convert("RGB")
        image_source = f"Uploaded File: {uploaded_file.name}"
        st.session_state.is_failure_case = False
    except Exception as e:
        st.error(f"Error loading uploaded image: {e}")
elif st.session_state.selected_sample is not None:
    if os.path.exists(st.session_state.selected_sample):
        active_image = Image.open(st.session_state.selected_sample).convert("RGB")
        image_source = f"Sample Image: {os.path.basename(st.session_state.selected_sample)}"

# Inference & Display
if active_image is not None:
    # Load model and run inference
    try:
        model = get_cached_model(model_key)
        results = predict_single_image(active_image, model)
    except Exception as e:
        st.error(f"Inference error: {e}")
        st.stop()

    pred_class = results["predicted_class"]
    confidence = results["confidence"] * 100
    latency = results["latency_ms"]
    probs = results["probabilities"]

    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.markdown(f"#### 📷 Input Cell Image (`{image_source}`)")
        st.image(active_image, use_container_width=True, caption=f"Resolution: {active_image.size[0]} x {active_image.size[1]} px | Mode: {active_image.mode}")
        
        if st.session_state.is_failure_case:
            st.warning("""
            **⚠️ Documented Known Failure Case Analysis:**  
            - **Ground Truth:** Eosinophil  
            - **Predicted:** Monocyte (Conf: 99.4%)  
            - **Morphological Rationale:** In this sample, the cytoplasm lacks the typical prominent coarse orange granules, exhibiting pale ground-glass staining and a non-bilobed indented nucleus that closely mimics a monocyte. This illustrates the known clinical challenge of staining variability.
            """)

    with col2:
        st.markdown("#### 🎯 Model Classification Results")
        
        # Primary prediction callout
        card_color = CLASS_COLORS.get(pred_class, "#1e3799")
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {card_color}15, #ffffff); border-left: 6px solid {card_color}; padding: 1.2rem; border-radius: 8px; margin-bottom: 1rem;">
            <div style="font-size: 0.9rem; text-transform: uppercase; color: #555; font-weight: 600;">Predicted Category</div>
            <div style="font-size: 2rem; font-weight: 800; color: {card_color}; margin: 0.2rem 0;">{pred_class}</div>
            <div style="display: flex; gap: 1rem; margin-top: 0.5rem;">
                <span class="badge badge-success">Confidence: {confidence:.2f}%</span>
                <span class="badge badge-primary">Latency: {latency:.1f} ms</span>
                <span class="badge badge-warning">{model_choice}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Softmax Probability Distribution Chart
        st.markdown("##### Softmax Probability Distribution across 5 Classes")
        df_probs = pd.DataFrame({
            'WBC Class': list(probs.keys()),
            'Probability (%)': [v * 100 for v in probs.values()]
        }).sort_values('Probability (%)', ascending=True)

        fig, ax = plt.subplots(figsize=(7, 3.2))
        bar_colors = [CLASS_COLORS.get(c, '#3498db') for c in df_probs['WBC Class']]
        bars = ax.barh(df_probs['WBC Class'], df_probs['Probability (%)'], color=bar_colors, height=0.55)
        ax.set_xlim(0, 105)
        ax.set_xlabel("Predicted Probability (%)", fontsize=9, fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 1.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%",
                    va='center', fontsize=9, fontweight='bold', color='#2c3e50')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Morphological and Diagnostic Description Card
        st.markdown("##### 🔬 Morphological & Diagnostic Features")
        desc = descriptions.get(pred_class, "Standard white blood cell subtype.")
        st.info(f"**{pred_class} Characteristics:** {desc}")

else:
    st.info("👆 Please upload an image using the file uploader above or select one of the curated test samples from the sidebar gallery.")

# Detailed Educational & Technical Sections
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["🔬 WBC Morphology Guide", "📊 Model Benchmark Comparison", "⚖️ Clinical Governance & Limitations"])

with tab1:
    st.markdown("""
    ### Microscopic Hallmarks of the 5 WBC Subtypes
    Peripheral blood smears are routinely examined to identify distinct leukocyte populations:
    """)
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("""
        - **Neutrophil (50–70% of normal blood):**
          - *Nucleus:* Segmented with 3 to 5 distinct lobes connected by fine chromatin threads.
          - *Cytoplasm:* Neutral lilac-pink with fine azurophilic and specific granules.
          - *Clinical Role:* Primary acute response against bacterial infections.
        - **Lymphocyte (20–40% of normal blood):**
          - *Nucleus:* Compact, round or slightly indented, dense dark purple chromatin.
          - *Cytoplasm:* Very thin rim of clear pale blue cytoplasm (high N:C ratio).
          - *Clinical Role:* Adaptive immunity, antibody production (B-cells), cell-mediated defense (T-cells, NK-cells).
        - **Basophil (0.5–1.0% of normal blood):**
          - *Nucleus:* Segmented into 2–3 lobes, but frequently obscured.
          - *Cytoplasm:* Heavily packed with coarse, water-soluble, dark purple-black granules.
          - *Clinical Role:* Mediates hypersensitivity, anaphylaxis, and histamine release.
        """)
    with col_t2:
        st.markdown("""
        - **Eosinophil (1–4% of normal blood):**
          - *Nucleus:* Characteristically bilobed (spectacle-shaped).
          - *Cytoplasm:* Densely packed with large, uniform, refractive bright orange-red granules.
          - *Clinical Role:* Defense against helminthic parasites and mediation of allergic asthma.
        - **Monocyte (2–8% of normal blood):**
          - *Nucleus:* Large, folded or indented (kidney bean / horseshoe shaped) with spongy chromatin.
          - *Cytoplasm:* Abundant grayish-blue ("ground glass") appearance, often containing vacuoles.
          - *Clinical Role:* Phagocytosis, antigen presentation, tissue macrophage precursor.
        """)

with tab2:
    st.markdown("""
    ### Held-Out Test-A Benchmark Evaluation (1,000 Images)
    """)
    eval_comp = pd.DataFrame({
        'Model': ['Custom Baseline CNN', 'MobileNetV2 (Transfer Learning)'],
        'Overall Accuracy': ['89.40%', '91.20%'],
        'Macro F1-Score': ['84.47%', '89.77%'],
        'Macro Recall': ['89.68%', '95.67%'],
        'Macro Precision': ['81.74%', '87.51%'],
        'Basophil F1': ['97.60%', '100.00%'],
        'Monocyte F1': ['61.70%', '92.70%'],
        'Parameters': ['1.44M', '2.39M'],
        'Inference Latency': ['38.2 ms', '38.0 ms']
    })
    st.dataframe(eval_comp, use_container_width=True)

with tab3:
    st.markdown("""
    ### Responsible AI & Clinical Use Policy
    1. **Research & Educational Proof of Concept:**
       - This application is developed as part of an AI/ML Specialisation Capstone Project.
       - **It is not certified as a medical device (SaMD) and must never be used to make diagnostic, prognostic, or therapeutic decisions.**
    2. **Class Imbalance & Staining Sensitivity:**
       - The dataset exhibits natural physiological imbalance (Neutrophils ~61% vs Basophils ~2.1%).
       - Images stained with atypical staining protocols, non-standard illuminations, or taken from whole smears without proper cell cropping may produce erroneous classifications.
    3. **Expert In-The-Loop:**
       - Automated hematological image analysis tools must always serve as assistive decision-support tools under the supervision of a certified pathologist.
    """)
