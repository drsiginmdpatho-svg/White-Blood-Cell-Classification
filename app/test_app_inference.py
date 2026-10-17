"""
app/test_app_inference.py

Sanity check script for Streamlit backend functions.
"""

import os
import sys
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils import load_model, load_class_metadata, predict_single_image

def main():
    print("Testing metadata loading...")
    meta = load_class_metadata()
    print("Classes:", meta['classes'])

    print("Testing MobileNetV2 inference...")
    model_mn = load_model("mobilenet_v2")
    img_baso = Image.open(os.path.join("app", "samples", "sample_basophil.jpg"))
    res_baso = predict_single_image(img_baso, model_mn)
    print(f"Sample Basophil -> Predicted: {res_baso['predicted_class']} ({res_baso['confidence']*100:.2f}%) in {res_baso['latency_ms']:.1f}ms")

    img_fail = Image.open(os.path.join("app", "samples", "known_failure_case.jpg"))
    res_fail = predict_single_image(img_fail, model_mn)
    print(f"Known Failure Case -> Predicted: {res_fail['predicted_class']} ({res_fail['confidence']*100:.2f}%)")

    print("Testing Baseline CNN inference...")
    model_base = load_model("baseline_cnn")
    res_base = predict_single_image(img_baso, model_base)
    print(f"Baseline CNN on Basophil -> Predicted: {res_base['predicted_class']} ({res_base['confidence']*100:.2f}%)")

    print("App backend verification completely successful!")

if __name__ == "__main__":
    main()
