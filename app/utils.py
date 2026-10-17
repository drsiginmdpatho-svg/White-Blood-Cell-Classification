"""
app/utils.py

Helper utilities for loading models, image preprocessing, and inference in the Streamlit application.
"""

import os
import sys
import time
import json
from typing import Dict, Any, Tuple
from PIL import Image
import torch
import torchvision.transforms as transforms

# Add project root and src to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from src.models import get_transfer_model, BaselineCNN
from src.dataset import IMAGENET_MEAN, IMAGENET_STD, CLASSES


def get_eval_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Deterministic image transform for inference."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def load_class_metadata() -> Dict[str, Any]:
    """Loads class definitions, descriptions, and color palettes."""
    json_path = os.path.join(PROJECT_ROOT, "models", "class_mapping.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "classes": CLASSES,
        "descriptions": {c: f"Leukocyte subtype: {c}" for c in CLASSES}
    }


def load_model(model_name: str = "mobilenet_v2", device: torch.device = torch.device("cpu")) -> torch.nn.Module:
    """Loads the designated checkpoint without retraining."""
    if model_name == "mobilenet_v2":
        model = get_transfer_model("mobilenet_v2", num_classes=len(CLASSES), pretrained=False).to(device)
        ckpt_path = os.path.join(PROJECT_ROOT, "models", "mobilenet_v2_wbc.pt")
    elif model_name == "baseline_cnn":
        model = BaselineCNN(num_classes=len(CLASSES)).to(device)
        ckpt_path = os.path.join(PROJECT_ROOT, "models", "baseline_cnn.pt")
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_single_image(
    image: Image.Image,
    model: torch.nn.Module,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """
    Runs inference on a PIL image and returns predicted class, confidence,
    all class probabilities, and execution latency.
    """
    transform = get_eval_transforms()
    img_rgb = image.convert("RGB")
    tensor = transform(img_rgb).unsqueeze(0).to(device)

    start_time = time.time()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    latency_ms = (time.time() - start_time) * 1000.0

    pred_idx = int(probs.argmax())
    pred_class = CLASSES[pred_idx]
    confidence = float(probs[pred_idx])

    prob_dict = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

    return {
        "predicted_class": pred_class,
        "predicted_index": pred_idx,
        "confidence": confidence,
        "probabilities": prob_dict,
        "latency_ms": latency_ms
    }
