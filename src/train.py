"""
src/train.py

Training engine and validation loop for WBC classification models.
Supports class-weighted loss, gradient clipping, learning rate scheduling,
and best-model checkpointing based on Validation Macro F1.
"""

import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import time
import json
from typing import Dict, List, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

try:
    from src.dataset import CLASSES
except ImportError:
    from dataset import CLASSES


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """Runs one training epoch with gradient updates."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
    epoch_loss = running_loss / len(all_targets)
    epoch_acc = accuracy_score(all_targets, all_preds)
    _, _, epoch_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0
    )
    
    return {
        'loss': float(epoch_loss),
        'acc': float(epoch_acc),
        'f1': float(epoch_f1)
    }


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Runs one evaluation epoch without gradient updates."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    epoch_loss = running_loss / len(all_targets)
    epoch_acc = accuracy_score(all_targets, all_preds)
    _, _, epoch_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0
    )
    
    return {
        'loss': float(epoch_loss),
        'acc': float(epoch_acc),
        'f1': float(epoch_f1)
    }


def train_model(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    num_epochs: int = 10,
    checkpoint_path: str = "models/best_model.pt",
    device: torch.device = torch.device('cpu'),
    save_history_path: Optional[str] = None
) -> Dict[str, List[float]]:
    """
    Complete training pipeline orchestrating training epochs, validation,
    learning rate scheduling, and best checkpoint saving.
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    history: Dict[str, List[float]] = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'lr': []
    }
    
    best_val_f1 = -1.0
    total_start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"Starting Training: {num_epochs} Epochs on Device: {device}")
    print(f"{'='*70}")
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # 1. Training Phase
        train_metrics = train_one_epoch(
            model=model,
            dataloader=dataloaders['train'],
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )
        
        # 2. Validation Phase
        val_metrics = validate_one_epoch(
            model=model,
            dataloader=dataloaders['val'],
            criterion=criterion,
            device=device
        )
        
        # 3. Learning Rate Adjustment
        curr_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
                
        # 4. Record History
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['acc'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['acc'])
        history['val_f1'].append(val_metrics['f1'])
        history['lr'].append(curr_lr)
        
        epoch_dur = time.time() - epoch_start
        
        # 5. Checkpointing on Best Validation Macro F1
        is_best = val_metrics['f1'] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'val_acc': val_metrics['acc'],
                'classes': CLASSES
            }, checkpoint_path)
            best_marker = "[BEST CHECKPOINT]"
        else:
            best_marker = ""
            
        print(f"Epoch {epoch:02d}/{num_epochs:02d} [{epoch_dur:.1f}s] | "
              f"Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['acc']*100:.1f}% | F1: {train_metrics['f1']*100:.1f}% || "
              f"Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['acc']*100:.1f}% | F1: {val_metrics['f1']*100:.1f}% | "
              f"LR: {curr_lr:.2e} {best_marker}")
              
    total_duration = time.time() - total_start_time
    print(f"{'='*70}")
    print(f"Training Complete in {total_duration/60:.2f} min | Best Validation Macro F1: {best_val_f1*100:.2f}%")
    print(f"Best checkpoint saved to: {checkpoint_path}")
    print(f"{'='*70}\n")
    
    if save_history_path:
        os.makedirs(os.path.dirname(save_history_path), exist_ok=True)
        with open(save_history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4)
        print(f"Training history saved to: {save_history_path}")
        
    return history
