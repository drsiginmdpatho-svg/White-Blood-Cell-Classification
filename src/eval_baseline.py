"""
src/eval_baseline.py

Evaluates the saved Baseline CNN model checkpoint on the validation subset
and held-out Test-A benchmark, producing confusion matrices and JSON metrics.
"""

import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from dataset import (
    CLASSES,
    load_dataset_dataframe,
    create_stratified_train_val_split,
    compute_class_weights,
    get_dataloaders
)
from models import BaselineCNN
from evaluate import evaluate_model, plot_confusion_matrix


def run_evaluation(seed: int = 42):
    device = torch.device('cpu')
    print(f"Device: {device} (PyTorch {torch.__version__})")
    
    print("1. Loading dataset...")
    df_train_full, df_test_a_full, _ = load_dataset_dataframe()
    df_train_raw, df_val_raw = create_stratified_train_val_split(df_train_full, val_ratio=0.20, random_state=seed)

    df_train_sub, _ = train_test_split(df_train_raw, train_size=3200, stratify=df_train_raw['label'], random_state=seed)
    df_val_sub, _ = train_test_split(df_val_raw, train_size=800, stratify=df_val_raw['label'], random_state=seed)
    df_test_sub, _ = train_test_split(df_test_a_full, train_size=1000, stratify=df_test_a_full['label'], random_state=seed)

    print(f"Validation samples: {len(df_val_sub):,}")
    print(f"Held-out Test samples: {len(df_test_sub):,}")

    class_weights = compute_class_weights(df_train_sub).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    dataloaders = get_dataloaders(
        df_train=df_train_sub,
        df_val=df_val_sub,
        df_test=df_test_sub,
        batch_size=32,
        num_workers=0
    )

    print("2. Loading Baseline CNN checkpoint...")
    checkpoint_path = os.path.join("models", "baseline_cnn.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = BaselineCNN(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint trained to Epoch {checkpoint['epoch']} with Val Macro F1: {checkpoint['val_f1']*100:.2f}%")

    print("\n3. Evaluating on Validation Set:")
    val_res = evaluate_model(model, dataloaders['val'], criterion=criterion, device=device)
    print(f"  Accuracy:       {val_res['accuracy']*100:.2f}%")
    print(f"  Macro F1-Score: {val_res['macro_f1']*100:.2f}%")
    print(f"  Macro Recall:   {val_res['macro_recall']*100:.2f}%")
    print(f"  Latency/Sample: {val_res['inference_time_ms']:.2f} ms")

    val_cm_path = os.path.join("reports", "figures", "baseline_cnn_val_cm.png")
    plot_confusion_matrix(
        np.array(val_res['confusion_matrix']),
        classes=CLASSES,
        title="Baseline CNN - Validation Confusion Matrix",
        save_path=val_cm_path
    )

    print("\n4. Evaluating on Held-Out Test-A:")
    test_res = evaluate_model(model, dataloaders['test'], criterion=criterion, device=device)
    print(f"  Accuracy:       {test_res['accuracy']*100:.2f}%")
    print(f"  Macro F1-Score: {test_res['macro_f1']*100:.2f}%")
    print(f"  Macro Recall:   {test_res['macro_recall']*100:.2f}%")
    print(f"  Macro Precision:{test_res['macro_precision']*100:.2f}%")
    print(f"  Latency/Sample: {test_res['inference_time_ms']:.2f} ms")

    test_cm_path = os.path.join("reports", "figures", "baseline_cnn_test_cm.png")
    plot_confusion_matrix(
        np.array(test_res['confusion_matrix']),
        classes=CLASSES,
        title="Baseline CNN - Held-Out Test-A Confusion Matrix",
        save_path=test_cm_path
    )

    print("\nPer-Class Breakdown on Held-Out Test-A:")
    for cls, metrics in test_res['per_class'].items():
        print(f"  {cls:12s} | Precision: {metrics['precision']*100:5.1f}% | Recall: {metrics['recall']*100:5.1f}% | F1: {metrics['f1_score']*100:5.1f}% | Support: {metrics['support']}")

    # Save to JSON
    results_path = os.path.join("models", "baseline_cnn_eval.json")
    clean_eval = {
        'val_metrics': {k: v for k, v in val_res.items() if k not in ['y_true', 'y_pred', 'y_probs']},
        'test_metrics': {k: v for k, v in test_res.items() if k not in ['y_true', 'y_pred', 'y_probs']}
    }
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(clean_eval, f, indent=4)
    print(f"\nSaved evaluation metrics to: {results_path}")


if __name__ == "__main__":
    run_evaluation()
