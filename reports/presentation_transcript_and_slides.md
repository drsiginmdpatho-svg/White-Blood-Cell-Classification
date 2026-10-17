# Academic Presentation Guide & Speaker Notes
## Peripheral Blood Smear White Blood Cell Classification Using Deep Learning

**Course:** AI / ML Specialisation Capstone Project  
**Domain:** Healthcare AI & Computer Vision  
**Presentation File:** [`reports/WBC_Classification_Academic_Presentation.pptx`](file:///c:/Users/sigin/Desktop/Hero%20Vired/Data%20Science%20AI%20&%20ML/Capstone%20Project/reports/WBC_Classification_Academic_Presentation.pptx)  
**Layout:** 16:9 Widescreen | 8 Academic Slides  

---

### Slide 1: Project Title, Team Members & Problem Context
- **Title:** Peripheral Blood Smear White Blood Cell Classification Using Deep Learning
- **Subtitle:** Automated Morphological Categorization of Leukocyte Subtypes via Custom CNN & Transfer Learning
- **Problem Context:**
  - Manual differential leukocyte counting under light microscopy is the diagnostic standard for hematological disorders, systemic infections, and immunodeficiencies.
  - Manual morphological examination requires highly trained hematopathologists, is labor-intensive, and is prone to observer variability and diagnostic fatigue.
  - Objective: Construct an automated, deep-learning-based classification pipeline that classifies cropped WBC images into five primary categories: Neutrophils, Lymphocytes, Monocytes, Eosinophils, and Basophils.
  - Clinical Goal: Assist medical laboratories with rapid, reproducible decision support with sub-40ms CPU inference.
- **Speaker Notes:**
  > *"Good morning/afternoon. Today I present our AI/ML Capstone Project on automated White Blood Cell classification from peripheral blood smears. White blood cells are essential components of our immune system. In hematology clinics worldwide, manual morphological review of blood smears is critical but remains slow and operator-dependent. Our objective was to develop and compare supervised deep learning models that automate the classification of five distinct WBC subtypes, remediate severe natural class imbalance, and deliver a deployable, real-time clinical demonstration."*

---

### Slide 2: Dataset Source, Access Method, Data Size & Sample Images
- **Source & Provenance:** Masoud Nickparvar's White Blood Cells Dataset on Kaggle, originating from the Raabin-WBC repository (*Tavakoli et al., 2021*).
- **Total Volume:** **16,633 RGB images** in `.jpg` format, with high native resolutions (492x487 to 575x575 pixels).
- **Split Breakdown & Verification:**
  - **`Train/`:** 10,175 images across all 5 classes.
  - **`Test-A/`:** 4,339 images across all 5 classes (held-out benchmark matching training proportions).
  - **`Test-B/`:** 2,119 images (Lymphocyte and Neutrophil only; external hardware generalizability test).
  - **Leakage Safety:** Confirmed **0 overlapping filenames** across splits.
- **Class Distribution:** Neutrophils (61.2%), Lymphocytes (23.9%), Eosinophils (7.3%), Monocytes (5.5%), Basophils (2.1%).
- **Visual Grid:** Shows representative morphology: Basophil (dark granules obscuring nucleus), Eosinophil (bilobed with pink/orange granules), Lymphocyte (compact spherical nucleus, high N:C ratio), Monocyte (folded/kidney nucleus), Neutrophil (3–5 segmented lobes).
- **Speaker Notes:**
  > *"Our dataset consists of 16,633 verified RGB images across three dedicated splits. A critical audit finding is the natural physiological class imbalance: Neutrophils represent over 61% of samples, whereas Basophils comprise only 2.1%—an imbalance ratio of approximately 29 to 1. Test-A contains 4,339 images with all 5 classes and serves as our untouched evaluation benchmark, while Test-B provides an external cross-device evaluation set."*

---

### Slide 3: EDA & Preprocessing Decisions
- **Leakage-Safe Partitioning:**
  - Preserved `Test-A` strictly untouched.
  - Created a stratified 80/20 train/validation split on `Train` (8,140 train, 2,035 validation, seed 42) preserving exact class proportions.
- **Inverse-Frequency Loss Weighting:**
  $$w_c = \frac{N}{C \cdot N_c}$$
  Weights: Basophil = 2.80, Monocyte = 1.06, Eosinophil = 0.80, Lymphocyte = 0.24, Neutrophil = 0.09.
- **Image Preprocessing:** Uniform bilinear resizing to 224x224 pixels and ImageNet channel standardization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
- **Domain-Specific Augmentation:** Random rotations ($\pm 15^\circ$), horizontal and vertical reflections ($p=0.5$), and subtle color jitter ($\pm 10\%$). Applied **strictly to training mini-batches** to prevent leakage.
- **Speaker Notes:**
  > *"Because microscopy images possess no canonical upright orientation, rotation and reflection augmentations naturally expand visual diversity without altering biological validity. To handle class imbalance, we implemented inverse-frequency loss weighting, preventing majority classes like Neutrophils from dominating the gradient updates while ensuring sensitive detection of rare Basophils."*

---

### Slide 4: Baseline Model vs. Transfer Learning Architecture
- **1. Custom Baseline CNN (From Scratch):**
  - **1,439,269 parameters** (1.44M).
  - 4 convolutional blocks (32, 64, 128, 256 filters) with `3x3` kernels, Batch Normalization, and ReLU activations.
  - Spatial downsampling via Max-Pooling (`2x2`) and final Adaptive Average Pooling (`4x4`).
  - Dense classification head: `Flatten (4096) -> Linear(256) -> BatchNorm -> ReLU -> Dropout(0.4) -> Linear(5)`.
- **2. MobileNetV2 (Transfer Learning):**
  - **2,388,485 parameters** (2.39M).
  - Pretrained on ImageNet-1k; 18 Inverted Residual Bottleneck blocks with Depthwise Separable Convolutions.
  - Custom classification head: `Dropout(0.3) -> Linear(1280->128) -> ReLU -> Dropout(0.2) -> Linear(128->5)`.
  - Fine-tuned with differential learning rates ($10^{-4}$ for backbone, $10^{-3}$ for classifier).
- **Speaker Notes:**
  > *"We formulated two distinct architectures: a 4-stage convolutional baseline trained from scratch to set an empirical lower bound, and an advanced MobileNetV2 model leveraging ImageNet representations. MobileNetV2 utilizes depthwise separable convolutions, which dramatically lower mathematical FLOPs, making it ideally suited for fast, low-power CPU inference."*

---

### Slide 5: Experiment Tracking & Hyperparameter Choices
- **Optimizer:** AdamW with weight decay $10^{-4}$ for decoupled weight regularization.
- **Learning Rate Strategy:**
  - Baseline CNN: Initial $lr = 10^{-3}$.
  - MobileNetV2: Differential learning rates ($10^{-4}$ for backbone, $10^{-3}$ for classification head).
  - Scheduler: `ReduceLROnPlateau` (factor=0.5, patience=1 monitoring validation loss).
- **Batch Size & Epochs:** Batch size 32; 6 epochs with early checkpointing.
- **Checkpointing Protocol:** Best model checkpoints saved based on highest **Validation Macro F1-Score**.
- **Metrics Tracked:** Epoch loss, overall accuracy, Macro F1, Macro Recall, and learning rate schedules recorded to JSON.
- **Speaker Notes:**
  > *"We tracked experiments using validation loss and Macro F1 rather than overall accuracy, since accuracy is misleading under heavy class imbalance. Gradient clipping with a maximum norm of 2.0 ensured training stability, while ReduceLROnPlateau facilitated fine convergence as loss plateaus were reached."*

---

### Slide 6: Quantitative Evaluation Results & In-Depth Error Analysis
- **Held-Out Test-A Benchmark (1,000 Images):**
  - Overall Accuracy: MobileNetV2 **91.20%** vs. Baseline **89.40%** (+1.80%).
  - Macro F1-Score: MobileNetV2 **89.77%** vs. Baseline **84.47%** (**+5.30% gain**).
  - Macro Recall: MobileNetV2 **95.67%** vs. Baseline **89.68%** (+5.99%).
  - Macro Precision: MobileNetV2 **87.51%** vs. Baseline **81.74%** (+5.77%).
  - Inference Latency: **37.99 ms per sample** on CPU.
- **Per-Class Breakdown:**
  - **Basophil:** **100.0% Precision, 100.0% Recall, 100.0% F1** (flawless rare-class detection).
  - **Lymphocyte:** **98.3% Precision, 98.3% Recall, 98.3% F1**.
  - **Monocyte Breakthrough (+31.0% F1):** Precision jumped from 48.4% in Baseline to **91.1%** in MobileNetV2 ($F_1 = 92.7\%$). Pretrained features effectively isolated folded nuclear contours.
  - **Neutrophil:** **99.8% Precision, 86.9% Recall, 92.9% F1**.
  - **Eosinophil:** **98.6% Recall**, 48.3% Precision (some granulocytic neutrophils flagged as eosinophils).
- **Speaker Notes:**
  > *"Evaluating on held-out Test-A demonstrates the clear superiority of transfer learning. MobileNetV2 achieved a Macro F1 of 89.77%, with our most dramatic improvement occurring in Monocytes—where F1 jumped by 31 percentage points. Notably, Basophil detection was flawless at 100% precision and recall, demonstrating that rare clinical subtypes can be mastered when proper loss weighting is enforced."*

---

### Slide 7: Deployment Demo & Inference Workflow
- **Streamlit Prototype (`app/app.py`):**
  - Real-time inference application running without retraining overhead.
  - Image drag-and-drop supporting JPG, JPEG, and PNG formats.
  - Interactive Softmax probability bar chart across all 5 classes.
  - Morphological reference cards detailing nuclear shape, cytoplasmic granules, and clinical role.
  - Dual model toggle between MobileNetV2 and Baseline CNN.
  - Curated Sample Gallery: 5 high-confidence class buttons plus 1 authentic documented failure case.
  - 1-Click Launchers: `run_app.bat` and `run_app.ps1`.
- **Speaker Notes:**
  > *"To demonstrate practical deployment, we packaged the winning MobileNetV2 model into an interactive Streamlit application. Clinicians or researchers can drag and drop any cropped smear image, view the predicted category, inspect the softmax confidence distribution, and review cell morphology hallmarks with sub-40ms latency."*

---

### Slide 8: Risks, Limitations, Ethical Governance & Future Roadmap
- **Risks & Clinical Limitations:**
  - **Non-Diagnostic Proof of Concept:** Tool is intended strictly for research and education. Must never replace a certified hematopathologist.
  - **Cross-Domain Hardware Shift (*Tavakoli et al., 2021*):** External testing on `Test-B` (2,119 images) showed dramatic accuracy degradation (<9%), confirming that variations in microscope sensors and illumination require domain harmonization.
  - **Pre-Cropped Input Dependency:** Relies on pre-cropped single cells rather than gigapixel whole-slide scans.
- **Future Roadmap:**
  - Integration of optical stain harmonization (e.g., Reinhard or Macenko normalization).
  - End-to-end detection coupling with YOLOv8 for automated leukocyte localization from whole slides.
  - Explainable AI heatmaps (Grad-CAM) to visualize cellular regions driving classification.
  - Expansion to abnormal malignant leukocytes (e.g., myeloblasts and lymphoblasts for acute leukemia triage).
- **Speaker Notes:**
  > *"Finally, responsible AI requires transparent appraisal of limitations. When we tested on external Test-B, performance dropped, demonstrating the classic medical imaging domain shift. Moving forward, integrating stain normalization algorithms and extending the model to detect whole-slide smears and malignant blast cells will bridge the gap between benchtop AI and clinical hematology workflows. Thank you, and I look forward to your questions."*
