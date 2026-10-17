"""
src/dataset.py

Dataset loading, stratified splitting, augmentation, and PyTorch DataLoader pipeline
for White Blood Cell (WBC) classification.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import glob
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Standard 5 WBC classes defined in the capstone problem statement
CLASSES: List[str] = ['Basophil', 'Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {i: c for i, c in enumerate(CLASSES)}

# ImageNet normalization standard (used for MobileNetV2, ResNet, and baseline CNN)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_dataset_dataframe(base_dir: str = "WBC archive") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Scans the WBC dataset folder structure and builds Pandas DataFrames for Train, Test-A, and Test-B.
    
    Args:
        base_dir: Root directory containing 'Train', 'Test-A', and 'Test-B' subfolders.
        
    Returns:
        Tuple of DataFrames: (df_train, df_test_a, df_test_b)
    """
    def _scan_split(split_name: str) -> pd.DataFrame:
        split_dir = os.path.join(base_dir, split_name)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Directory not found: {split_dir}")
            
        records = []
        for class_name in os.listdir(split_dir):
            class_folder = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_folder):
                continue
            if class_name not in CLASS_TO_IDX:
                continue
                
            files = glob.glob(os.path.join(class_folder, "*.*"))
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png']:
                    records.append({
                        'filepath': os.path.abspath(f),
                        'filename': os.path.basename(f),
                        'class_name': class_name,
                        'label': CLASS_TO_IDX[class_name],
                        'split': split_name
                    })
        return pd.DataFrame(records)

    df_train = _scan_split("Train")
    df_test_a = _scan_split("Test-A")
    df_test_b = _scan_split("Test-B")

    return df_train, df_test_a, df_test_b


def create_stratified_train_val_split(
    df_train: pd.DataFrame, 
    val_ratio: float = 0.20, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates a leak-free stratified train/validation split from the training dataset.
    
    Args:
        df_train: DataFrame of the full training set.
        val_ratio: Fraction of training samples allocated to validation (default 0.20).
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of DataFrames: (df_train_split, df_val_split)
    """
    train_idx, val_idx = train_test_split(
        df_train.index,
        test_size=val_ratio,
        stratify=df_train['label'],
        random_state=random_state
    )
    df_train_split = df_train.loc[train_idx].copy().reset_index(drop=True)
    df_val_split = df_train.loc[val_idx].copy().reset_index(drop=True)
    
    df_train_split['subset'] = 'train'
    df_val_split['subset'] = 'validation'
    
    return df_train_split, df_val_split


def get_train_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Returns data augmentation and preprocessing pipeline for training.
    Rotations and flips simulate microscopic viewing angles without canonical orientation.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_eval_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Returns deterministic preprocessing pipeline for validation and test sets.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


class WBCDataset(Dataset):
    """
    PyTorch Dataset for White Blood Cell classification.
    """
    def __init__(self, df: pd.DataFrame, transform: Optional[transforms.Compose] = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.filepaths = self.df['filepath'].values
        self.labels = self.df['label'].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.filepaths[idx]
        image = Image.open(img_path).convert("RGB")
        label = int(self.labels[idx])
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label


def compute_class_weights(df: pd.DataFrame, num_classes: int = 5) -> torch.Tensor:
    """
    Computes inverse-frequency class weights to remediate dataset imbalance:
    weight[c] = total_samples / (num_classes * count[c])
    
    Args:
        df: Training DataFrame containing 'label' column.
        num_classes: Total number of classes (5).
        
    Returns:
        torch.FloatTensor of shape (num_classes,)
    """
    total_samples = len(df)
    class_counts = df['label'].value_counts().to_dict()
    weights = np.zeros(num_classes, dtype=np.float32)
    
    for c in range(num_classes):
        count = class_counts.get(c, 0)
        if count > 0:
            weights[c] = total_samples / (num_classes * count)
        else:
            weights[c] = 1.0
            
    # Normalize so mean weight is 1.0
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def get_dataloaders(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: Optional[pd.DataFrame] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: Tuple[int, int] = (224, 224)
) -> Dict[str, DataLoader]:
    """
    Creates PyTorch DataLoaders for train, val, and optional test sets.
    """
    train_transform = get_train_transforms(image_size)
    eval_transform = get_eval_transforms(image_size)
    
    train_ds = WBCDataset(df_train, transform=train_transform)
    val_ds = WBCDataset(df_val, transform=eval_transform)
    
    dataloaders = {
        'train': DataLoader(
            train_ds, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=False
        ),
        'val': DataLoader(
            val_ds, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=False
        )
    }
    
    if df_test is not None:
        test_ds = WBCDataset(df_test, transform=eval_transform)
        dataloaders['test'] = DataLoader(
            test_ds, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=False
        )
        
    return dataloaders


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverses ImageNet normalization to return a displayable numpy array [H, W, 3] in range [0, 1].
    """
    img = tensor.clone().detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img = std * img + mean
    return np.clip(img, 0, 1)
