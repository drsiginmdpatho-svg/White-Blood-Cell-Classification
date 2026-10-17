"""
src/evaluate.py

Evaluation metrics, confusion matrix visualization, error analysis,
and performance benchmarking for WBC classification models.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

try:
    from src.dataset import CLASSES, IDX_TO_CLASS, denormalize_image
except ImportError:
    from dataset import CLASSES, IDX_TO_CLASS, denormalize_image


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module = None,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    """
    Evaluates a model on a given DataLoader, calculating comprehensive metrics:
    - Loss, Accuracy, Macro Precision, Macro Recall, Macro F1
    - Weighted Precision, Weighted Recall, Weighted F1
    - Per-class precision, recall, and F1
    - Confusion Matrix
    - Average inference latency (ms per sample)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            if criterion is not None:
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    total_time = time.time() - start_time
    total_samples = len(all_targets)
    avg_latency_ms = (total_time / total_samples) * 1000.0 if total_samples > 0 else 0.0
    avg_loss = (total_loss / total_samples) if total_samples > 0 else 0.0
    
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    per_class_p, per_class_r, per_class_f1, per_class_sup = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    
    class_metrics = {}
    for i, cls in enumerate(CLASSES):
        class_metrics[cls] = {
            'precision': float(per_class_p[i]) if i < len(per_class_p) else 0.0,
            'recall': float(per_class_r[i]) if i < len(per_class_r) else 0.0,
            'f1_score': float(per_class_f1[i]) if i < len(per_class_f1) else 0.0,
            'support': int(per_class_sup[i]) if i < len(per_class_sup) else 0
        }
        
    return {
        'loss': float(avg_loss),
        'accuracy': float(acc),
        'macro_precision': float(macro_p),
        'macro_recall': float(macro_r),
        'macro_f1': float(macro_f1),
        'weighted_precision': float(weighted_p),
        'weighted_recall': float(weighted_r),
        'weighted_f1': float(weighted_f1),
        'per_class': class_metrics,
        'confusion_matrix': cm.tolist(),
        'inference_time_ms': float(avg_latency_ms),
        'total_samples': total_samples,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_probs': np.array(all_probs)
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str] = CLASSES,
    title: str = "Confusion Matrix",
    save_path: str = None,
    normalize: bool = False
):
    """Plots and optionally saves a high-resolution confusion matrix heatmap."""
    if normalize:
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
        fmt = '.2%'
        data_to_plot = cm_norm
    else:
        fmt = 'd'
        data_to_plot = cm

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        data_to_plot,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title(title, fontsize=13, fontweight='bold', pad=12)
    plt.ylabel('True Class', fontsize=11, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=11, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def plot_training_curves(history: Dict[str, List[float]], title_prefix: str, save_path: str = None):
    """Plots loss, accuracy, and macro F1 curves over epochs."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    
    # 1. Loss Curve
    axes[0].plot(epochs, history['train_loss'], 'o-', label='Train Loss', color='#e74c3c', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 's--', label='Val Loss', color='#c0392b', linewidth=2)
    axes[0].set_title(f'{title_prefix} - Cross-Entropy Loss', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # 2. Accuracy Curve
    axes[1].plot(epochs, [a * 100 for a in history['train_acc']], 'o-', label='Train Acc', color='#3498db', linewidth=2)
    axes[1].plot(epochs, [a * 100 for a in history['val_acc']], 's--', label='Val Acc', color='#2980b9', linewidth=2)
    axes[1].set_title(f'{title_prefix} - Accuracy (%)', fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    # 3. Macro F1 Curve
    axes[2].plot(epochs, [f * 100 for f in history['train_f1']], 'o-', label='Train Macro F1', color='#2ecc71', linewidth=2)
    axes[2].plot(epochs, [f * 100 for f in history['val_f1']], 's--', label='Val Macro F1', color='#27ae60', linewidth=2)
    axes[2].set_title(f'{title_prefix} - Macro F1-Score (%)', fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Macro F1 (%)')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    plt.close()
