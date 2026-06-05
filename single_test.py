#!/usr/bin/env python3
"""
single_text.py

Tests the image-to-ASCII ML model on a single image file path.
Supports both ONNX (.onnx) and Keras (.keras) model formats.

Usage:
    python single_text.py path/to/image.jpg --cols 120
    python single_text.py path/to/image.jpg --invert
    python single_text.py path/to/image.jpg --model ascii_cam_model/model.onnx
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from PIL import Image

# Setup default paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(SCRIPT_DIR, "ascii_cam_model")
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")

def load_ascii_chars(config_path):
    """Loads the custom character ramp from config.json."""
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)["ASCII_CHARS"]
        except Exception as e:
            print(f"⚠️ Error reading config file {config_path}: {e}")
    
    # Fallback default character ramp from low to high density
    print("⚠️ Using fallback ASCII character ramp.")
    return [" ", ".", ",", "-", "~", ":", "i", "r", "s", "t", "l", "C", "O", "Z", "w", "m", "#", "8", "%", "@"]

def preprocess_image(image_path, cols):
    """
    Loads an image, converts it to grayscale, and resizes it.
    Accounts for the aspect ratio difference in character cells (typically 1:2 width:height).
    
    Returns:
        X (np.ndarray): Tensor of shape (rows * cols, 16, 8, 1) normalized to [0, 1]
        rows (int): Number of character rows in the output
        cols (int): Number of character columns in the output
        w_orig (int): Original image width
        h_orig (int): Original image height
    """
    if not os.path.exists(image_path):
        print(f"❌ Image path not found: {image_path}")
        sys.exit(1)
        
    try:
        img = Image.open(image_path).convert("L")
    except Exception as e:
        print(f"❌ Error loading image {image_path}: {e}")
        sys.exit(1)
        
    w_orig, h_orig = img.size
    aspect_ratio = w_orig / h_orig
    
    # Character cells are 8px wide and 16px high (1:2 aspect ratio).
    # To keep physical dimensions correct on output:
    # (cols * 8) / (rows * 16) ≈ aspect_ratio
    # rows ≈ cols * 0.5 / aspect_ratio
    rows = int(round(cols * 0.5 / aspect_ratio))
    rows = max(1, rows)
    
    target_w = cols * 8
    target_h = rows * 16
    
    # Resize image to fit the patch grid exactly
    img_resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    
    # Segment image into patches of size 16x8
    patches = []
    for r in range(rows):
        for c in range(cols):
            y0 = r * 16
            x0 = c * 8
            patch = img_np[y0 : y0 + 16, x0 : x0 + 8]
            patches.append(patch)
            
    # Format into Keras/ONNX input shape (Batch, Height, Width, Channels)
    X = np.expand_dims(np.array(patches, dtype=np.float32), axis=-1)
    return X, rows, cols, w_orig, h_orig

def predict_onnx(model_path, X):
    """Runs batch inference using ONNX Runtime."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("❌ 'onnxruntime' is not installed. Run 'uv pip install onnxruntime' or equivalent.")
        sys.exit(1)
        
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Run prediction
    outputs = session.run([output_name], {input_name: X})[0]
    return np.argmax(outputs, axis=1)

def predict_keras(model_path, X):
    """Runs batch inference using Keras (configured to PyTorch backend)."""
    # Set backend to torch to match the training scripts
    os.environ["KERAS_BACKEND"] = "torch"
    try:
        import keras
    except ImportError:
        print("❌ 'keras' is not installed. Run 'uv pip install keras' or equivalent.")
        sys.exit(1)
        
    try:
        model = keras.models.load_model(model_path)
    except Exception as e:
        print(f"❌ Failed to load Keras model from {model_path}: {e}")
        sys.exit(1)
        
    outputs = model.predict(X, verbose=0)
    return np.argmax(outputs, axis=1)

def main():
    parser = argparse.ArgumentParser(description="Test single images with the ASCII Cam ML model.")
    parser.add_argument("image_path", help="Path to the input image file.")
    parser.add_argument("--cols", "-c", type=int, default=100, help="Width of output ASCII art in characters (default: 100).")
    parser.add_argument("--model", "-m", help="Path to the model file (.onnx or .keras). Auto-detected if not specified.")
    parser.add_argument("--config", "-cfg", default=DEFAULT_CONFIG_PATH, help="Path to config.json (default: config.json).")
    parser.add_argument("--invert", "-i", action="store_true", help="Invert the ASCII character density mapping (useful for light-background terminals).")
    parser.add_argument("--output", "-o", help="Save the ASCII art to a text file.")
    
    args = parser.parse_args()
    
    # Auto-detect model candidate
    model_path = args.model
    if not model_path:
        onnx_candidate = os.path.join(DEFAULT_MODEL_DIR, "model.onnx")
        keras_candidate = os.path.join(DEFAULT_MODEL_DIR, "model.keras")
        if os.path.exists(onnx_candidate):
            model_path = onnx_candidate
        elif os.path.exists(keras_candidate):
            model_path = keras_candidate
        else:
            print(f"❌ No model found in default location: {DEFAULT_MODEL_DIR}")
            print("Please specify a model path using --model or run train_model.py first.")
            sys.exit(1)
            
    print(f"🔍 Loading model from: {model_path}")
    is_onnx = model_path.lower().endswith(".onnx")
    
    # Load character mapping
    ascii_chars = load_ascii_chars(args.config)
    if args.invert:
        ascii_chars = ascii_chars[::-1]
        print("🔄 Inverted ASCII character density ramp.")
        
    # Preprocess image
    print(f"🖼️  Processing image: {args.image_path}")
    start_time = time.time()
    X, rows, cols, w, h = preprocess_image(args.image_path, args.cols)
    preprocess_time = time.time() - start_time
    
    print(f"📐 Original Size: {w}x{h} | Resized Grid: {cols}x{rows} ({len(X)} patches)")
    
    # Run Inference
    inference_start = time.time()
    if is_onnx:
        predicted_classes = predict_onnx(model_path, X)
    else:
        predicted_classes = predict_keras(model_path, X)
    inference_time = time.time() - inference_start
    
    # Map predictions to characters
    ascii_chars_mapped = [ascii_chars[idx] for idx in predicted_classes]
    
    # Format lines
    ascii_rows = []
    for r in range(rows):
        start_idx = r * cols
        end_idx = start_idx + cols
        ascii_rows.append("".join(ascii_chars_mapped[start_idx:end_idx]))
    
    ascii_art = "\n".join(ascii_rows)
    
    print(f"\n📊 Benchmark Metrics:")
    print(f"   • Preprocessing: {preprocess_time * 1000:.1f}ms")
    print(f"   • Model Inference ({'ONNX' if is_onnx else 'Keras'}): {inference_time * 1000:.1f}ms")
    print(f"   • Total Time: {(time.time() - start_time) * 1000:.1f}ms\n")
    
    # Print or save
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(ascii_art)
            print(f"💾 Saved ASCII art to: {args.output}")
        except Exception as e:
            print(f"❌ Error saving output to {args.output}: {e}")
    else:
        print("🎨 Generated ASCII Art:")
        print("-" * cols)
        print(ascii_art)
        print("-" * cols)

if __name__ == "__main__":
    main()
