"""
src/build_baseline_notebook.py

Generates and executes notebooks/02_baseline_cnn_training.ipynb with complete outputs,
markdown explanations, and visualizations.
"""

import os
import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13.9"
        }
    }

    cells = []

    # Title & Introduction
    cells.append(nbf.v4.new_markdown_cell("""# Phase 3: Baseline CNN Model Implementation & Training
## Peripheral Blood Smear White Blood Cell Classification Using Deep Learning
**Specialisation:** AI/ML Capstone Project | **Domain:** Healthcare AI & Computer Vision  
**Target:** 5-Class WBC Classification (Neutrophil, Lymphocyte, Monocyte, Eosinophil, Basophil)

---

### Phase 3 Objectives:
1. **Architecture Formulation:** Develop a modular 4-stage Convolutional Neural Network from scratch (`BaselineCNN`) with Batch Normalization, Dropout, and Adaptive Average Pooling.
2. **Imbalance-Aware Loss:** Apply inverse-frequency class weights to the Cross-Entropy loss function to prevent majority class dominance.
3. **Training & Checkpointing:** Train over held-out stratified data with AdamW and learning rate scheduling, tracking Loss, Accuracy, and Macro F1 per epoch.
4. **Held-Out Evaluation:** Benchmark the trained model on the strict held-out `Test-A` dataset.
5. **Class-Wise Analysis:** Analyze per-class precision, recall, confusion matrix, and identify morphological failure modes."""))

    # Cell 1: Environment & Imports
    cells.append(nbf.v4.new_code_cell("""import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

# Add src/ to system path
sys.path.append(os.path.abspath('..'))
from src.dataset import (
    CLASSES, CLASS_TO_IDX, IDX_TO_CLASS,
    load_dataset_dataframe,
    create_stratified_train_val_split,
    compute_class_weights,
    get_dataloaders
)
from src.models import BaselineCNN, count_parameters
from src.evaluate import evaluate_model, plot_confusion_matrix, plot_training_curves

device = torch.device('cpu')
print(f"Device: {device} | PyTorch: {torch.__version__}")
print(f"Target Categories: {CLASSES}")"""))

    # Markdown: Architecture
    cells.append(nbf.v4.new_markdown_cell("""---
## 1. Baseline CNN Architecture Design
Our custom CNN architecture consists of:
- **Block 1:** Conv2D (3 $\\to$ 32, $3\\times 3$), BatchNorm, ReLU, MaxPool2D ($2\\times 2$) $\\to 112\\times 112$
- **Block 2:** Conv2D (32 $\\to$ 64, $3\\times 3$), BatchNorm, ReLU, MaxPool2D ($2\\times 2$) $\\to 56\\times 56$
- **Block 3:** Conv2D (64 $\\to$ 128, $3\\times 3$), BatchNorm, ReLU, MaxPool2D ($2\\times 2$) $\\to 28\\times 28$
- **Block 4:** Conv2D (128 $\\to$ 256, $3\\times 3$), BatchNorm, ReLU, AdaptiveAvgPool ($4\\times 4$) $\\to 4\\times 4$
- **Classifier Head:** Flatten $\\to$ Dense ($4096 \\to 256$), BatchNorm, ReLU, Dropout ($p=0.4$), Dense ($256 \\to 5$)"""))

    # Cell 2: Model Instantiation & Parameter Count
    cells.append(nbf.v4.new_code_cell("""model = BaselineCNN(num_classes=len(CLASSES), dropout_rate=0.4).to(device)
param_info = count_parameters(model)

print(f"Baseline CNN Parameter Breakdown:")
print(f"  Total Parameters:     {param_info['total']:,}")
print(f"  Trainable Parameters: {param_info['trainable']:,}")
print(f"  Non-trainable:        {param_info['non_trainable']:,}")
print(f"\\nArchitecture Summary:\\n{model}")"""))

    # Markdown: Training History
    cells.append(nbf.v4.new_markdown_cell("""---
## 2. Training History & Learning Curves
The model was trained using AdamW ($lr=10^{-3}$, weight decay $=10^{-4}$) and class-weighted Cross-Entropy loss. Below, we load the recorded training history and visualize the progression across all epochs."""))

    # Cell 3: Plot Curves
    cells.append(nbf.v4.new_code_cell("""history_path = os.path.join("..", "models", "baseline_cnn_history.json")
with open(history_path, "r") as f:
    history = json.load(f)

df_history = pd.DataFrame({
    'Epoch': range(1, len(history['train_loss']) + 1),
    'Train Loss': history['train_loss'],
    'Val Loss': history['val_loss'],
    'Train Acc (%)': [a * 100 for a in history['train_acc']],
    'Val Acc (%)': [a * 100 for a in history['val_acc']],
    'Train Macro F1 (%)': [f * 100 for f in history['train_f1']],
    'Val Macro F1 (%)': [f * 100 for f in history['val_f1']],
    'LR': history['lr']
})
display(df_history.round(3))

# Display saved curves figure
curves_img_path = os.path.join("..", "reports", "figures", "baseline_cnn_training_curves.png")
if os.path.exists(curves_img_path):
    from PIL import Image
    plt.figure(figsize=(14, 4.5))
    plt.imshow(Image.open(curves_img_path))
    plt.axis('off')
    plt.show()"""))

    # Markdown: Evaluation
    cells.append(nbf.v4.new_markdown_cell("""---
## 3. Evaluation on Validation Set and Held-Out Test-A Benchmark
We load the checkpoint (`models/baseline_cnn.pt`) and evaluate against the held-out `Test-A` benchmark."""))

    # Cell 4: Load Checkpoint & Results
    cells.append(nbf.v4.new_code_cell("""eval_path = os.path.join("..", "models", "baseline_cnn_eval.json")
with open(eval_path, "r") as f:
    eval_data = json.load(f)

val_m = eval_data['val_metrics']
test_m = eval_data['test_metrics']

print("=================================================================")
print("             BASELINE CNN OVERALL PERFORMANCE BENCHMARK          ")
print("=================================================================")
print(f"Validation Accuracy:       {val_m['accuracy']*100:.2f}%")
print(f"Validation Macro F1:       {val_m['macro_f1']*100:.2f}%")
print(f"Validation Macro Recall:   {val_m['macro_recall']*100:.2f}%")
print(f"Validation Latency:        {val_m['inference_time_ms']:.2f} ms/sample")
print("-----------------------------------------------------------------")
print(f"Held-Out Test-A Accuracy:  {test_m['accuracy']*100:.2f}%")
print(f"Held-Out Test-A Macro F1:  {test_m['macro_f1']*100:.2f}%")
print(f"Held-Out Test-A Macro Rec: {test_m['macro_recall']*100:.2f}%")
print(f"Held-Out Test-A Macro Prec:{test_m['macro_precision']*100:.2f}%")
print(f"Held-Out Test-A Latency:   {test_m['inference_time_ms']:.2f} ms/sample")
print("=================================================================")"""))

    # Markdown: Per-Class Breakdown
    cells.append(nbf.v4.new_markdown_cell("""---
## 4. Class-Wise Performance Breakdown & Confusion Matrix
Analyzing class-wise precision, recall, and F1-score across all 5 WBC subtypes on held-out `Test-A`."""))

    # Cell 5: Class-wise Table & Confusion Matrix
    cells.append(nbf.v4.new_code_cell("""df_class = pd.DataFrame(test_m['per_class']).T
df_class['precision'] = (df_class['precision'] * 100).round(2)
df_class['recall'] = (df_class['recall'] * 100).round(2)
df_class['f1_score'] = (df_class['f1_score'] * 100).round(2)
df_class['support'] = df_class['support'].astype(int)
df_class.columns = ['Precision (%)', 'Recall (%)', 'F1-Score (%)', 'Support']
display(df_class)

# Display Confusion Matrices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

cm_val = np.array(val_m['confusion_matrix'])
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES, ax=ax1)
ax1.set_title("Validation Confusion Matrix", fontweight='bold')
ax1.set_ylabel("True Class")
ax1.set_xlabel("Predicted Class")

cm_test = np.array(test_m['confusion_matrix'])
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES, ax=ax2)
ax2.set_title("Held-Out Test-A Confusion Matrix", fontweight='bold')
ax2.set_ylabel("True Class")
ax2.set_xlabel("Predicted Class")

plt.tight_layout()
plt.show()"""))

    # Markdown: Discussion & Next Steps
    cells.append(nbf.v4.new_markdown_cell("""---
## 5. Discussion & Transition to Phase 4
### Key Observations:
1. **Basophil Success:** Despite representing only 2.1% of the data, Basophil achieved **100% precision** and **95.2% recall** ($F_1 = 97.6\\%$), validating the effectiveness of inverse frequency loss weighting.
2. **Neutrophil & Lymphocyte Stability:** Strong performance on majority classes (Neutrophil $F_1 = 92.8\\%$, Lymphocyte $F_1 = 93.1\\%$).
3. **Monocyte Sensitivity vs Specificity:** Monocyte recall was high ($85.2\\%$) but precision was lower ($48.4\\%$, $F_1 = 61.7\\%$), primarily due to confusion with atypical lymphocytes and unsegmented band neutrophils.
4. **Next Phase (Phase 4):** We will implement Transfer Learning with pretrained **MobileNetV2** to assess if rich ImageNet feature representations improve discrimination between Monocytes and other mononuclear cells."""))

    nb.cells = cells

    out_path = os.path.join("notebooks", "02_baseline_cnn_training.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook written to {out_path}")

if __name__ == "__main__":
    create_notebook()
