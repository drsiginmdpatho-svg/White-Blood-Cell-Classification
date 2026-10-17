"""
src/train_baseline.py

Executes training, validation, checkpointing, and evaluation for the Baseline CNN model.
"""

import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import json
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from dataset import (
    CLASSES,
    load_dataset_dataframe,
    create_stratified_train_val_split,
    compute_class_weights,
    get_dataloaders
)
from models import BaselineCNN, count_parameters
from train import train_model
from evaluate import evaluate_model, plot_confusion_matrix, plot_training_curves


def run_baseline_training(
    num_epochs: int = 6,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    train_sample_limit: int = 3200,
    val_sample_limit: int = 800,
    test_sample_limit: int = 1000,
    image_size: tuple = (224, 224),
    seed: int = 42
):
    torch.manual_seed(seed)
    device = torch.device('cpu')
    print(f"Device: {device} (PyTorch {torch.__version__})")
    
    print("1. Loading dataset...")
    df_train_full, df_test_a_full, _ = load_dataset_dataframe()
    
    # Create leak-free stratified split
    df_train_raw, df_val_raw = create_stratified_train_val_split(df_train_full, val_ratio=0.20, random_state=seed)
    
    # Subsample for efficient CPU convergence while keeping full class stratification
    if train_sample_limit and train_sample_limit < len(df_train_raw):
        df_train_sub, _ = train_test_split(
            df_train_raw,
            train_size=train_sample_limit,
            stratify=df_train_raw['label'],
            random_state=seed
        )
    else:
        df_train_sub = df_train_raw
        
    if val_sample_limit and val_sample_limit < len(df_val_raw):
        df_val_sub, _ = train_test_split(
            df_val_raw,
            train_size=val_sample_limit,
            stratify=df_val_raw['label'],
            random_state=seed
        )
    else:
        df_val_sub = df_val_raw

    if test_sample_limit and test_sample_limit < len(df_test_a_full):
        df_test_sub, _ = train_test_split(
            df_test_a_full,
            train_size=test_sample_limit,
            stratify=df_test_a_full['label'],
            random_state=seed
        )
    else:
        df_test_sub = df_test_a_full

    print(f"Training subset samples:   {len(df_train_sub):,}")
    print(f"Validation subset samples: {len(df_val_sub):,}")
    print(f"Held-out Test samples:     {len(df_test_sub):,}")

    print("2. Computing class weights for loss balancing...")
    class_weights = compute_class_weights(df_train_sub).to(device)
    print("Class weights:", {c: round(float(w), 3) for c, w in zip(CLASSES, class_weights)})

    print("3. Building DataLoaders...")
    dataloaders = get_dataloaders(
        df_train=df_train_sub,
        df_val=df_val_sub,
        df_test=df_test_sub,
        batch_size=batch_size,
        num_workers=0,
        image_size=image_size
    )

    print("4. Initializing Baseline CNN...")
    model = BaselineCNN(num_classes=len(CLASSES), dropout_rate=0.4).to(device)
    param_info = count_parameters(model)
    print(f"Model Parameters: {param_info['total']:,} total, {param_info['trainable']:,} trainable")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    checkpoint_path = os.path.join("models", "baseline_cnn.pt")
    history_path = os.path.join("models", "baseline_cnn_history.json")

    print("5. Training Baseline CNN...")
    history = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        checkpoint_path=checkpoint_path,
        device=device,
        save_history_path=history_path
    )

    print("6. Plotting training curves...")
    curves_path = os.path.join("reports", "figures", "baseline_cnn_training_curves.png")
    plot_training_curves(history, title_prefix="Baseline CNN", save_path=curves_path)

    print("7. Loading best checkpoint for final held-out evaluation...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from Epoch {checkpoint['epoch']} with Val Macro F1: {checkpoint['val_f1']*100:.2f}%")

    print("\n8. Evaluating on Validation Subset:")
    val_results = evaluate_model(model, dataloaders['val'], criterion=criterion, device=device)
    print(f"  Accuracy:       {val_results['accuracy']*100:.2f}%")
    print(f"  Macro F1-Score: {val_results['macro_f1']*100:.2f}%")
    print(f"  Macro Recall:   {val_results['macro_recall']*100:.2f}%")
    print(f"  Latency/Sample: {val_results['inference_time_ms']:.2f} ms")

    val_cm_path = os.path.join("reports", "figures", "baseline_cnn_val_cm.png")
    plot_confusion_matrix(
        np.array(val_results['confusion_matrix']),
        classes=CLASSES,
        title="Baseline CNN - Validation Confusion Matrix",
        save_path=val_cm_path
    )

    print("\n9. Evaluating on Held-Out Test-A:")
    test_results = evaluate_model(model, dataloaders['test'], criterion=criterion, device=device)
    print(f"  Accuracy:       {test_results['accuracy']*100:.2f}%")
    print(f"  Macro F1-Score: {test_results['macro_f1']*100:.2f}%")
    print(f"  Macro Recall:   {test_results['macro_recall']*100:.2f}%")
    print(f"  Macro Precision:{test_results['macro_precision']*100:.2f}%")
    print(f"  Latency/Sample: {test_results['inference_time_ms']:.2f} ms")

    test_cm_path = os.path.join("reports", "figures", "baseline_cnn_test_cm.png")
    plot_confusion_matrix(
        np.array(test_results['confusion_matrix']),
        classes=CLASSES,
        title="Baseline CNN - Held-Out Test-A Confusion Matrix",
        save_path=test_cm_path
    )

    print("\nPer-Class Performance on Test-A:")
    for cls, metrics in test_results['per_class'].items():
        print(f"  {cls:12s} | Precision: {metrics['precision']*100:5.1f}% | Recall: {metrics['recall']*100:5.1f}% | F1: {metrics['f1_score']*100:5.1f}% | Support: {metrics['support']}")

    # Save test results to JSON
    results_path = os.path.join("models", "baseline_cnn_eval.json")
    clean_eval = {
        'val_metrics': {k: v for k, v in val_results.items() if k not in ['y_true', 'y_pred', 'y_probs']},
        'test_metrics': {k: v for k, v in test_results.items() if k not in ['y_true', 'y_pred', 'y_probs']}
    }
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(clean_eval, f, indent=4)
    print(f"\nSaved evaluation metrics to: {results_path}")

    return model, history, test_results


if __name__ == "__main__":
    run_baseline_training()
