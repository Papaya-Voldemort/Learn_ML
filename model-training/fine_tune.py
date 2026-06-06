#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import json
import glob
import io
from collections import defaultdict

os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers, models
import numpy as np
from PIL import Image

# ── Constants ──────────────────────────────────────────────────────────
PATCH_W = 8
PATCH_H = 16
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SCRIPT_DIR, "config.json"), "r", encoding="utf-8") as f:
    ASCII_CHARS = json.load(f)["ASCII_CHARS"]

CHAR_TO_IDX = {c: i for i, c in enumerate(ASCII_CHARS)}

# ── jp2a Pipeline ──────────────────────────────────────────────────────
def convert_with_jp2a(grayscale_img, cols):
    """Pipes a pre-converted grayscale PIL image to jp2a."""
    buf = io.BytesIO()
    grayscale_img.save(buf, format="JPEG", quality=95)
    chars_ramp = "".join(ASCII_CHARS)

    result = subprocess.run(
        ["jp2a", "-", f"--width={cols}", f"--chars={chars_ramp}"],
        input=buf.getvalue(),
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode().strip())
    return result.stdout.decode("utf-8")

def process_image(image_path, cols, max_per_char=30):
    """
    Generates balanced patches. Caps the number of patches per character 
    class per image to prevent memory overload and class imbalance.
    """
    # 1. Convert to Grayscale once (so jp2a and patch extractor see the same lighting)
    img = Image.open(image_path).convert("L")
    
    # 2. Get Gold Standard labels
    ascii_text = convert_with_jp2a(img, cols)
    lines = [line for line in ascii_text.rstrip("\n").split("\n") if line]
    if not lines:
        return [], []

    rows = len(lines)
    actual_cols = max(len(line) for line in lines)
    lines = [line.ljust(actual_cols) for line in lines]

    # 3. Resize base image to match grid
    target_w = actual_cols * PATCH_W
    target_h = rows * PATCH_H
    img = img.resize((target_w, target_h), Image.LANCZOS)
    img_np = np.array(img, dtype=np.float32) / 255.0

    # 4. Group patches by character (Smart Sampling)
    grouped_patches = defaultdict(list)

    for r in range(rows):
        for c in range(actual_cols):
            char = lines[r][c]
            if char not in CHAR_TO_IDX:
                continue

            y0 = r * PATCH_H
            x0 = c * PATCH_W
            patch = img_np[y0 : y0 + PATCH_H, x0 : x0 + PATCH_W]

            if patch.shape == (PATCH_H, PATCH_W):
                grouped_patches[CHAR_TO_IDX[char]].append(patch)

    # 5. Cap and merge
    final_patches = []
    final_labels = []
    
    for char_idx, patch_list in grouped_patches.items():
        # Shuffle and cap the amount of patches for this specific character
        np.random.shuffle(patch_list)
        sampled = patch_list[:max_per_char]
        
        final_patches.extend(sampled)
        final_labels.extend([char_idx] * len(sampled))

    return final_patches, final_labels

# ── Model Architecture ─────────────────────────────────────────────────
def load_or_build_model():
    model_path = os.path.join(os.path.dirname(SCRIPT_DIR), "ascii_cam_model", "model.keras")
    if os.path.exists(model_path):
        print(f"✅ Loaded pre-trained model")
        return keras.models.load_model(model_path)

    print("⚠️ Rebuilding from scratch.")
    num_classes = len(ASCII_CHARS)
    return models.Sequential([
        layers.Input(shape=(PATCH_H, PATCH_W, 1)),
        layers.Conv2D(16, kernel_size=3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 1)),
        layers.Conv2D(32, kernel_size=3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])

# ── Main ───────────────────────────────────────────────────────────────
def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("image_folder")
    parser.add_argument("--cols", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mix-synthetic", type=float, default=0.5)
    parser.add_argument("--max-per-char", type=int, default=10)
    args = parser.parse_args(args)

    extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(args.image_folder, ext)))
        image_paths.extend(glob.glob(os.path.join(args.image_folder, ext.upper())))
    
    if not image_paths:
        print("❌ No images found.")
        sys.exit(1)

    print(f"📷 Processing {len(image_paths)} images with Smart Sampling...")
    all_patches, all_labels = [], []

    for i, path in enumerate(image_paths):
        try:
            # We cap the patches per char per image to maintain variety & low RAM
            p, l = process_image(path, args.cols, max_per_char=args.max_per_char)
            all_patches.extend(p)
            all_labels.extend(l)
            if i % 50 == 0:
                print(f"  Processed {i}/{len(image_paths)}...")
        except Exception as e:
            pass

    X_real = np.expand_dims(np.array(all_patches, dtype=np.float32), axis=-1)
    y_real = np.array(all_labels, dtype=np.int32)
    print(f"\n📊 Extracted {len(X_real):,} highly-balanced real-world patches.")

    if args.mix_synthetic > 0:
        try:
            X_synth = np.load(os.path.join(SCRIPT_DIR, "dataset", "X_data.npy"))
            y_synth = np.load(os.path.join(SCRIPT_DIR, "dataset", "y_data.npy"))
            n_synth = int(len(X_real) * args.mix_synthetic)
            replace = n_synth > len(X_synth)
            synth_idx = np.random.choice(len(X_synth), n_synth, replace=replace)
            
            X_real = np.concatenate([X_real, X_synth[synth_idx]])
            y_real = np.concatenate([y_real, y_synth[synth_idx]])
            print(f"🔀 Added {n_synth:,} structural synthetic patches.")
        except Exception as e:
            print(f"⚠️ Could not load synthetic data to mix: {e}")

    # Shuffle everything
    indices = np.arange(len(X_real))
    np.random.shuffle(indices)
    X_train, X_val = X_real[indices][:int(len(X_real)*0.9)], X_real[indices][int(len(X_real)*0.9):]
    y_train, y_val = y_real[indices][:int(len(X_real)*0.9)], y_real[indices][int(len(X_real)*0.9):]

    # Compute class weights for any remaining imbalance
    classes = np.unique(y_train)
    total_samples = len(y_train)
    n_classes = len(classes)
    class_counts = np.bincount(y_train)
    class_weight_dict = {}
    for cls in classes:
        count = class_counts[cls] if cls < len(class_counts) else 0
        if count > 0:
            class_weight_dict[int(cls)] = float(total_samples / (n_classes * count))
        else:
            class_weight_dict[int(cls)] = 1.0

    model = load_or_build_model()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00005), 
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=args.epochs, 
        batch_size=args.batch_size,
        class_weight=class_weight_dict
    )

    model_dir = os.path.join(os.path.dirname(SCRIPT_DIR), "ascii_cam_model")
    model.save(os.path.join(model_dir, "model.keras"))
    model.export(os.path.join(model_dir, "model.onnx"), format="onnx")
    print("✅ Model updated and saved!")

if __name__ == "__main__":
    main()