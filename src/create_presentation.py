"""
src/create_presentation.py

Generates a professional 8-slide academic presentation (.pptx) based on the Capstone Project:
"Peripheral Blood Smear White Blood Cell Classification Using Deep Learning"
Following the 8 specific required slides and incorporating high-res project figures and tables.
"""

import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

OUTPUT_PPTX = os.path.join("reports", "WBC_Classification_Academic_Presentation.pptx")
FIG_DIR = os.path.join("reports", "figures")
SAMPLE_DIR = os.path.join("app", "samples")

# Academic Color Palette
NAVY = RGBColor(15, 43, 92)          # Primary brand & titles
SLATE = RGBColor(51, 65, 85)         # Body text
LIGHT_BG = RGBColor(248, 250, 252)   # Card background
ACCENT_BLUE = RGBColor(30, 64, 175)  # Key callouts & headers
TEAL = RGBColor(13, 148, 136)        # Highlights & badges
BORDER_COLOR = RGBColor(203, 213, 225)
WHITE = RGBColor(255, 255, 255)
LIGHT_RED = RGBColor(254, 242, 242)
DARK_RED = RGBColor(185, 28, 28)


def create_presentation():
    prs = Presentation()
    # Set 16:9 widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_header(slide, title_text, category_text="AI/ML Specialisation Capstone Project"):
        # Header banner
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.733), Inches(0.9))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p0 = tf.paragraphs[0]
        p0.text = category_text.upper()
        p0.font.size = Pt(10)
        p0.font.bold = True
        p0.font.color.rgb = TEAL

        p1 = tf.add_paragraph()
        p1.text = title_text
        p1.font.size = Pt(22)
        p1.font.bold = True
        p1.font.color.rgb = NAVY

        # Subtle bottom line
        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.32), Inches(11.733), Inches(0.02))
        line.fill.solid()
        line.fill.fore_color.rgb = BORDER_COLOR
        line.line.color.rgb = BORDER_COLOR

    def add_card(slide, left, top, width, height, bg_color=LIGHT_BG, border_color=BORDER_COLOR):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1)
        return shape

    # =========================================================================
    # SLIDE 1: Title, Team Members, and Problem Context
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)
    
    # Hero background card
    hero = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(0.8), Inches(11.733), Inches(5.9))
    hero.fill.solid()
    hero.fill.fore_color.rgb = NAVY
    hero.line.fill.background()

    tb1 = slide1.shapes.add_textbox(Inches(1.3), Inches(1.2), Inches(10.733), Inches(3.2))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "AI / ML SPECIALISATION CAPSTONE PROJECT"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = TEAL

    p2 = tf1.add_paragraph()
    p2.text = "Peripheral Blood Smear White Blood Cell\nClassification Using Deep Learning"
    p2.font.size = Pt(30)
    p2.font.bold = True
    p2.font.color.rgb = WHITE
    p2.space_before = Pt(12)

    p3 = tf1.add_paragraph()
    p3.text = "Automated Morphological Categorization of Leukocyte Subtypes via Custom CNN & Transfer Learning"
    p3.font.size = Pt(14)
    p3.font.color.rgb = RGBColor(203, 213, 225)
    p3.space_before = Pt(8)

    # Context Card inside Slide 1
    c1 = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.3), Inches(4.2), Inches(6.8), Inches(2.1))
    c1.fill.solid()
    c1.fill.fore_color.rgb = RGBColor(26, 58, 110)
    c1.line.color.rgb = RGBColor(40, 80, 140)

    tb_ctx = slide1.shapes.add_textbox(Inches(1.5), Inches(4.3), Inches(6.4), Inches(1.9))
    tf_ctx = tb_ctx.text_frame
    tf_ctx.word_wrap = True

    p_ctx_h = tf_ctx.paragraphs[0]
    p_ctx_h.text = "Clinical Problem Statement & Context:"
    p_ctx_h.font.size = Pt(12)
    p_ctx_h.font.bold = True
    p_ctx_h.font.color.rgb = WHITE

    points = [
        "White blood cell (WBC) differential analysis is a cornerstone of hematological diagnostics.",
        "Manual microscopy is labor-intensive, operator-dependent, and prone to diagnostic fatigue.",
        "Objective: Build an end-to-end computer vision pipeline to classify cropped WBCs into 5 subtypes: Neutrophils, Lymphocytes, Monocytes, Eosinophils, and Basophils.",
        "Clinical Goal: Provide rapid, reproducible decision-support with sub-40ms CPU inference."
    ]
    for pt in points:
        p_pt = tf_ctx.add_paragraph()
        p_pt.text = f"• {pt}"
        p_pt.font.size = Pt(10)
        p_pt.font.color.rgb = RGBColor(226, 232, 240)
        p_pt.space_before = Pt(3)

    # Metadata card (Author & Details)
    c2 = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.3), Inches(4.2), Inches(3.7), Inches(2.1))
    c2.fill.solid()
    c2.fill.fore_color.rgb = RGBColor(26, 58, 110)
    c2.line.color.rgb = RGBColor(40, 80, 140)

    tb_meta = slide1.shapes.add_textbox(Inches(8.5), Inches(4.3), Inches(3.3), Inches(1.9))
    tf_meta = tb_meta.text_frame
    tf_meta.word_wrap = True

    p_m_h = tf_meta.paragraphs[0]
    p_m_h.text = "Project Metadata:"
    p_m_h.font.size = Pt(12)
    p_m_h.font.bold = True
    p_m_h.font.color.rgb = WHITE

    meta_items = [
        ("Domain:", "Healthcare AI / Computer Vision"),
        ("Programme:", "Hero Vired AI/ML Specialisation"),
        ("Framework:", "PyTorch 2.14 & Streamlit 1.66"),
        ("Hardware:", "AMD Ryzen 7 5700U (8-Core CPU)"),
        ("Project Type:", "Individual Capstone Project")
    ]
    for k, v in meta_items:
        pm = tf_meta.add_paragraph()
        pm.text = f"{k} {v}"
        pm.font.size = Pt(10)
        pm.font.color.rgb = RGBColor(226, 232, 240)
        pm.space_before = Pt(3)

    # =========================================================================
    # SLIDE 2: Dataset Source, Access Method, Data Size, and Sample Records
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_layout)
    add_header(slide2, "Slide 2: Dataset Source, Access Method, Data Size & Sample Images")

    # Left Column: Dataset Details Card
    add_card(slide2, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.4))
    tb2 = slide2.shapes.add_textbox(Inches(1.0), Inches(1.65), Inches(5.2), Inches(5.1))
    tf2 = tb2.text_frame
    tf2.word_wrap = True

    p = tf2.paragraphs[0]
    p.text = "Dataset Specification & Structure"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = NAVY

    ds_bullets = [
        ("Source & Access:", "White Blood Cells Dataset by Masoud Nickparvar (Kaggle), derived from the Raabin-WBC repository (Tavakoli et al., 2021)."),
        ("Total Verified Volume:", "16,633 RGB images in JPG format (resolutions 492x487 to 575x575 px). Zero file corruption detected."),
        ("Split Verification:", "Three distinct directories with 0 cross-split filename leakage:"),
        ("  • Train Set:", "10,175 images across all 5 classes (used for training & validation)."),
        ("  • Test-A Set:", "4,339 images across all 5 classes (held-out evaluation benchmark)."),
        ("  • Test-B Set:", "2,119 images (Lymphocyte & Neutrophil only; external hardware generalizability test)."),
        ("5 Target Categories:", "Neutrophil (61.2%), Lymphocyte (23.9%), Eosinophil (7.3%), Monocyte (5.5%), Basophil (2.1%)."),
        ("Severe Imbalance:", "Majority:minority ratio is ~29.4 : 1 (Neutrophils vs. Basophils), requiring specialized loss weighting.")
    ]
    for title_b, desc_b in ds_bullets:
        p_b = tf2.add_paragraph()
        p_b.text = f"• {title_b} {desc_b}"
        p_b.font.size = Pt(10)
        p_b.font.color.rgb = SLATE
        p_b.space_before = Pt(4)

    # Right Column: Visual Samples Grid Figure
    samples_fig = os.path.join(FIG_DIR, "representative_wbc_samples.png")
    if os.path.exists(samples_fig):
        add_card(slide2, Inches(6.7), Inches(1.5), Inches(5.833), Inches(5.4))
        slide2.shapes.add_picture(samples_fig, Inches(6.85), Inches(1.65), width=Inches(5.533))

    # =========================================================================
    # SLIDE 3: EDA and Preprocessing Decisions
    # =========================================================================
    slide3 = prs.slides.add_slide(blank_layout)
    add_header(slide3, "Slide 3: Exploratory Data Analysis & Preprocessing Pipeline")

    # Left: Distribution Chart
    dist_fig = os.path.join(FIG_DIR, "class_distribution_comparison.png")
    if os.path.exists(dist_fig):
        add_card(slide3, Inches(0.8), Inches(1.5), Inches(6.2), Inches(5.4))
        slide3.shapes.add_picture(dist_fig, Inches(0.95), Inches(1.8), width=Inches(5.9))

    # Right: Preprocessing Decisions Card
    add_card(slide3, Inches(7.3), Inches(1.5), Inches(5.233), Inches(5.4))
    tb3 = slide3.shapes.add_textbox(Inches(7.5), Inches(1.65), Inches(4.8), Inches(5.1))
    tf3 = tb3.text_frame
    tf3.word_wrap = True

    p = tf3.paragraphs[0]
    p.text = "Key EDA Insights & Pipeline Decisions"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = NAVY

    eda_points = [
        ("Leakage-Safe Partitioning:", "Preserved Test-A (4,339 imgs) strictly untouched. Partitioned Train (10,175 imgs) via stratified 80/20 split (8,140 train, 2,035 val, seed 42)."),
        ("Inverse Class Weights:", "Computed loss weights w_c = N / (C * N_c) to counter extreme imbalance: Basophil (2.80), Monocyte (1.06), Eosinophil (0.80), Lymphocyte (0.24), Neutrophil (0.09)."),
        ("Spatial Standardization:", "All cropped images resized to 224 x 224 pixels with bilinear interpolation to match CNN receptive fields."),
        ("Microscopy-Specific Augmentation:", "Peripheral blood smears have no canonical orientation. Applied random rotations (±15°), horizontal/vertical flips (p=0.5), and color jitter (±10% brightness/contrast)."),
        ("Strict Leakage Boundary:", "Augmentation is applied solely to training mini-batches; validation and test sets receive only deterministic resizing and ImageNet normalization.")
    ]
    for t_e, d_e in eda_points:
        pe = tf3.add_paragraph()
        pe.text = f"• {t_e} {d_e}"
        pe.font.size = Pt(10)
        pe.font.color.rgb = SLATE
        pe.space_before = Pt(6)

    # =========================================================================
    # SLIDE 4: Baseline Model and Advanced Model Architecture
    # =========================================================================
    slide4 = prs.slides.add_slide(blank_layout)
    add_header(slide4, "Slide 4: Baseline Model vs. Transfer Learning Architecture")

    # Column 1: Baseline CNN Architecture
    add_card(slide4, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.4))
    tb4a = slide4.shapes.add_textbox(Inches(1.0), Inches(1.65), Inches(5.2), Inches(5.1))
    tf4a = tb4a.text_frame
    tf4a.word_wrap = True

    p = tf4a.paragraphs[0]
    p.text = "1. Custom Baseline CNN (From Scratch)"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = NAVY

    b_specs = [
        ("Total Parameters:", "1,439,269 (1.44 Million parameters)"),
        ("Conv Block 1:", "Conv2D (3->32, 3x3), BatchNorm, ReLU, MaxPool(2x2) -> 112x112"),
        ("Conv Block 2:", "Conv2D (32->64, 3x3), BatchNorm, ReLU, MaxPool(2x2) -> 56x56"),
        ("Conv Block 3:", "Conv2D (64->128, 3x3), BatchNorm, ReLU, MaxPool(2x2) -> 28x28"),
        ("Conv Block 4:", "Conv2D (128->256, 3x3), BatchNorm, ReLU, AdaptiveAvgPool(4x4)"),
        ("Classifier Head:", "Flatten (4096) -> Linear(256), BatchNorm, ReLU, Dropout (p=0.4) -> Linear(5)"),
        ("Design Rationale:", "Lightweight spatial feature extractor designed to establish an empirical benchmark without external pretraining.")
    ]
    for k, v in b_specs:
        pb = tf4a.add_paragraph()
        pb.text = f"• {k} {v}"
        pb.font.size = Pt(10)
        pb.font.color.rgb = SLATE
        pb.space_before = Pt(5)

    # Column 2: MobileNetV2 Transfer Learning Architecture
    add_card(slide4, Inches(6.8), Inches(1.5), Inches(5.733), Inches(5.4))
    tb4b = slide4.shapes.add_textbox(Inches(7.0), Inches(1.65), Inches(5.3), Inches(5.1))
    tf4b = tb4b.text_frame
    tf4b.word_wrap = True

    p = tf4b.paragraphs[0]
    p.text = "2. MobileNetV2 (Transfer Learning)"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = NAVY

    m_specs = [
        ("Total Parameters:", "2,388,485 (2.39 Million parameters)"),
        ("Pretrained Backbone:", "ImageNet-1k pretrained weights; 18 Inverted Residual Bottleneck blocks with Depthwise Separable Convolutions."),
        ("Computational Efficiency:", "Depthwise separable convolutions drastically reduce FLOPs, enabling sub-40ms inference on standard CPUs."),
        ("Custom Head:", "Dropout (p=0.3) -> Linear (1280->128) -> ReLU -> Dropout (p=0.2) -> Linear (128->5)."),
        ("Transfer Strategy:", "Fine-tuned end-to-end using differential learning rates (10^-4 for backbone, 10^-3 for classification head)."),
        ("Advantage:", "Pretrained hierarchical filters immediately recognize complex cellular contours, nuclear folding, and chromatin texture.")
    ]
    for k, v in m_specs:
        pm = tf4b.add_paragraph()
        pm.text = f"• {k} {v}"
        pm.font.size = Pt(10)
        pm.font.color.rgb = SLATE
        pm.space_before = Pt(5)

    # =========================================================================
    # SLIDE 5: Experiment Tracking and Hyperparameter Choices
    # =========================================================================
    slide5 = prs.slides.add_slide(blank_layout)
    add_header(slide5, "Slide 5: Experiment Tracking & Hyperparameter Configuration")

    # Left: Training Curves Figure
    curves_fig = os.path.join(FIG_DIR, "mobilenet_v2_training_curves.png")
    if os.path.exists(curves_fig):
        add_card(slide5, Inches(0.8), Inches(1.5), Inches(6.5), Inches(5.4))
        slide5.shapes.add_picture(curves_fig, Inches(0.95), Inches(2.1), width=Inches(6.2))

    # Right: Hyperparameters Table Card
    add_card(slide5, Inches(7.6), Inches(1.5), Inches(4.933), Inches(5.4))
    tb5 = slide5.shapes.add_textbox(Inches(7.8), Inches(1.65), Inches(4.5), Inches(5.1))
    tf5 = tb5.text_frame
    tf5.word_wrap = True

    p = tf5.paragraphs[0]
    p.text = "Experimental Setup & Hyperparameters"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = NAVY

    exp_params = [
        ("Optimization Algorithm:", "AdamW (weight decay = 1e-4) for decoupled regularization."),
        ("Differential Learning Rate:", "Backbone features: 1e-4 | Classification Head: 1e-3."),
        ("LR Scheduler:", "ReduceLROnPlateau (factor=0.5, patience=1, metric=Val Loss)."),
        ("Batch Size & Epochs:", "Batch size: 32 | Training Epochs: 6 epochs with early checkpointing."),
        ("Loss Function:", "Class-Weighted Cross-Entropy (weights: 2.80, 0.80, 0.24, 1.06, 0.09)."),
        ("Regularization:", "Dropout (0.3 & 0.2), Gradient Clipping (max_norm=2.0), Weight Decay."),
        ("Checkpoint Criterion:", "Optimal model weights saved based on highest Validation Macro F1."),
        ("Experiment Tracking:", "Logged epoch loss, accuracy, macro F1, and learning rates to JSON.")
    ]
    for k, v in exp_params:
        pe = tf5.add_paragraph()
        pe.text = f"• {k} {v}"
        pe.font.size = Pt(9.5)
        pe.font.color.rgb = SLATE
        pe.space_before = Pt(4)

    # =========================================================================
    # SLIDE 6: Evaluation Results and Error Analysis
    # =========================================================================
    slide6 = prs.slides.add_slide(blank_layout)
    add_header(slide6, "Slide 6: Quantitative Evaluation Results & In-Depth Error Analysis")

    # Left: Benchmark Comparison Graphic
    bench_fig = os.path.join(FIG_DIR, "model_comparison_benchmark.png")
    if os.path.exists(bench_fig):
        add_card(slide6, Inches(0.8), Inches(1.5), Inches(6.0), Inches(5.4))
        slide6.shapes.add_picture(bench_fig, Inches(0.9), Inches(1.8), width=Inches(5.8))

    # Right: Error Analysis and Findings
    add_card(slide6, Inches(7.1), Inches(1.5), Inches(5.433), Inches(5.4))
    tb6 = slide6.shapes.add_textbox(Inches(7.3), Inches(1.65), Inches(5.0), Inches(5.1))
    tf6 = tb6.text_frame
    tf6.word_wrap = True

    p = tf6.paragraphs[0]
    p.text = "Held-Out Test-A Benchmark (1,000 Imgs)"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = NAVY

    res_bullets = [
        ("Overall Test Accuracy:", "MobileNetV2: 91.20% vs. Baseline CNN: 89.40% (+1.80% gain)."),
        ("Macro F1-Score:", "MobileNetV2: 89.77% vs. Baseline CNN: 84.47% (+5.30% gain)."),
        ("Macro Recall & Precision:", "Recall: 95.67% (+5.99%) | Precision: 87.51% (+5.77%)."),
        ("Monocyte Breakthrough (+31% F1):", "Baseline F1 was 61.7% due to confusion with lymphocytes and band neutrophils. MobileNetV2 lifted F1 to 92.7% (precision 91.1%, recall 94.4%)."),
        ("Flawless Basophil Detection:", "100.0% precision and 100.0% recall achieved on the rarest class (~2.1% support), verifying loss weight efficacy."),
        ("Known Failure Mode:", "Eosinophil achieved 98.6% recall but 48.3% precision, as neutrophils with dense granular cytoplasm were occasionally flagged as eosinophils."),
        ("Sub-40ms Inference:", "Latency on standard CPU is 37.99 ms/image, ideal for real-time app.")
    ]
    for k, v in res_bullets:
        pr = tf6.add_paragraph()
        pr.text = f"• {k} {v}"
        pr.font.size = Pt(9.5)
        pr.font.color.rgb = SLATE
        pr.space_before = Pt(4)

    # =========================================================================
    # SLIDE 7: Deployment Demo and Inference Workflow
    # =========================================================================
    slide7 = prs.slides.add_slide(blank_layout)
    add_header(slide7, "Slide 7: Interactive Streamlit Demonstration & Inference Pipeline")

    # Left: Architecture / Workflow Card
    add_card(slide7, Inches(0.8), Inches(1.5), Inches(5.8), Inches(5.4))
    tb7 = slide7.shapes.add_textbox(Inches(1.0), Inches(1.65), Inches(5.4), Inches(5.1))
    tf7 = tb7.text_frame
    tf7.word_wrap = True

    p = tf7.paragraphs[0]
    p.text = "Streamlit Application Architecture"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = NAVY

    dep_bullets = [
        ("Deployment Target:", "Interactive Streamlit web application (app/app.py) loading saved checkpoints without retraining."),
        ("Dual Model Selector:", "Seamlessly toggle between Pretrained MobileNetV2 (Recommended) and Custom Baseline CNN."),
        ("File Drag-and-Drop:", "Accepts cropped JPG, JPEG, and PNG smear images with instant resolution and color-space validation."),
        ("Real-Time Probabilities:", "Renders horizontal Softmax probability distributions across all 5 classes with class-specific color coding."),
        ("Clinical Feature Cards:", "Displays cell morphology hallmarks (nuclear segments, granules, N:C ratio) for the predicted leukocyte."),
        ("Curated Sample Gallery:", "Includes 5 high-confidence test buttons for all classes plus 1 documented authentic failure case for transparent auditing."),
        ("1-Click Launchers:", "Packaged with run_app.bat and run_app.ps1 for frictionless execution on Windows environments.")
    ]
    for k, v in dep_bullets:
        pd_item = tf7.add_paragraph()
        pd_item.text = f"• {k} {v}"
        pd_item.font.size = Pt(10)
        pd_item.font.color.rgb = SLATE
        pd_item.space_before = Pt(5)

    # Right: Confusion Matrix Figure
    cm_fig = os.path.join(FIG_DIR, "mobilenet_v2_test_cm.png")
    if os.path.exists(cm_fig):
        add_card(slide7, Inches(6.9), Inches(1.5), Inches(5.633), Inches(5.4))
        slide7.shapes.add_picture(cm_fig, Inches(7.1), Inches(1.75), width=Inches(5.233))

    # =========================================================================
    # SLIDE 8: Risks, Limitations, Ethical Considerations & Future Enhancements
    # =========================================================================
    slide8 = prs.slides.add_slide(blank_layout)
    add_header(slide8, "Slide 8: Risks, Limitations, Ethical Governance & Future Roadmap")

    # Column 1: Risks, Limitations & Ethics
    add_card(slide8, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.4))
    tb8a = slide8.shapes.add_textbox(Inches(1.0), Inches(1.65), Inches(5.2), Inches(5.1))
    tf8a = tb8a.text_frame
    tf8a.word_wrap = True

    p = tf8a.paragraphs[0]
    p.text = "Risks, Limitations & Clinical Governance"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = NAVY

    r_points = [
        ("Non-Diagnostic Status:", "Educational & research proof-of-concept. Not approved for clinical diagnosis or patient management. Pathologist oversight is mandatory."),
        ("Generalizability Bottleneck (Test-B):", "External Test-B showed dramatic accuracy drops (<9%), validating Tavakoli et al. (2021) that optical sensor and stain variance impede naive cross-domain deployment."),
        ("Pre-Cropped Input Dependency:", "Pipeline requires pre-segmented individual cells; cannot process whole-slide gigapixel smears directly."),
        ("Stain & Illumination Sensitivity:", "Optimized for standard Wright-Giemsa Romanowsky staining; atypical lab protocols require domain recalibration.")
    ]
    for k, v in r_points:
        pr_item = tf8a.add_paragraph()
        pr_item.text = f"• {k} {v}"
        pr_item.font.size = Pt(10)
        pr_item.font.color.rgb = SLATE
        pr_item.space_before = Pt(6)

    # Column 2: Future Roadmap & Technical Enhancements
    add_card(slide8, Inches(6.8), Inches(1.5), Inches(5.733), Inches(5.4))
    tb8b = slide8.shapes.add_textbox(Inches(7.0), Inches(1.65), Inches(5.3), Inches(5.1))
    tf8b = tb8b.text_frame
    tf8b.word_wrap = True

    p = tf8b.paragraphs[0]
    p.text = "Future Technical & Clinical Roadmap"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = NAVY

    f_points = [
        ("Optical Stain Harmonization:", "Integrate Macenko or Reinhard stain normalization as an initial preprocessing stage to resolve cross-camera domain shifts."),
        ("End-to-End Whole-Slide Detection:", "Couple YOLOv8 or Faster R-CNN for automated leukocyte detection and cropping directly from full peripheral blood smear images."),
        ("Explainable AI (Grad-CAM):", "Incorporate Grad-CAM heatmaps in the Streamlit interface to visualize cellular regions (nuclear lobes vs. granules) driving predictions."),
        ("Hematologic Malignancy Expansion:", "Extend classification from 5 normal mature cell types to blast cells, immature granulocytes, and atypical lymphoid cells (e.g. leukemia screening)."),
        ("Clinical Prospective Validation:", "Multi-center clinical trials across diverse hospital hematology analyzers.")
    ]
    for k, v in f_points:
        pf_item = tf8b.add_paragraph()
        pf_item.text = f"• {k} {v}"
        pf_item.font.size = Pt(10)
        pf_item.font.color.rgb = SLATE
        pf_item.space_before = Pt(6)

    # Save presentation
    os.makedirs(os.path.dirname(OUTPUT_PPTX), exist_ok=True)
    prs.save(OUTPUT_PPTX)
    print(f"Academic presentation successfully generated and saved to: {OUTPUT_PPTX}")


if __name__ == "__main__":
    create_presentation()
