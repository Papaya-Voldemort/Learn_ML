#!/usr/bin/env python3
"""
Fine-tunes the ASCII cam model on real-world images.

Uses jp2a (https://github.com/cslarsen/jp2a), a classic and well-established
ASCII art converter, as the gold-standard reference. Each real photo is
converted to ASCII by jp2a, then the original image patches are paired with
jp2a's character choices to create supervised training data.

Usage:
    python fine_tune.py <image_folder>
    python fine_tune.py photos/ --cols 100 --epochs 15 --lr 0.00005

Requirements:
    - jp2a: brew install jp2a
    - A pre-trained base model: run train_model.py first
"""

import os
import sys
import argparse
import subprocess
import json
import glob
import io

os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers, models
import numpy as np
from PIL import Image

# ── Constants (must match train_model.py / app.js) ─────────────────────
PATCH_W = 8
PATCH_H = 16
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the shared character set from config
with open(os.path.join(SCRIPT_DIR, "config.json"), "r", encoding="utf-8") as f:
    ASCII_CHARS = json.load(f)["ASCII_CHARS"]

CHAR_TO_IDX = {c: i for i, c in enumerate(ASCII_CHARS)}


# ── jp2a Gold-Standard Converter ───────────────────────────────────────

def check_jp2a():
    """Verify jp2a is installed and accessible."""
    try:
        subprocess.run(["jp2a", "--version"], capture_output=True)
        return True
    except FileNotFoundError:
        return False


def convert_with_jp2a(image_path, cols):
    """
    Convert any image to ASCII using jp2a via stdin JPEG piping.
    
    This avoids writing temp files and supports all image formats
    that Pillow can read (PNG, WebP, BMP, TIFF, etc.), since we
    convert to JPEG in-memory before piping to jp2a.
    
    Returns the ASCII text output (kept in memory, never written to disk).
    """
    # Load with Pillow (supports all formats) and convert to JPEG in memory
    img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)

    # Build the character ramp string for jp2a (lightest → densest)
    chars_ramp = "".join(ASCII_CHARS)

    result = subprocess.run(
        ["jp2a", "-", f"--width={cols}", f"--chars={chars_ramp}"],
        input=buf.getvalue(),
        capture_output=True,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode().strip())

    return result.stdout.decode("utf-8")


# ── Patch Extraction ───────────────────────────────────────────────────

def process_image(image_path, cols):
    """
    Generate (patch, label) training pairs from a single real-world image:
    
    1. Run jp2a on the image → gold-standard ASCII text (in memory)
    2. Load the original image as grayscale
    3. Resize to exactly match the ASCII grid dimensions (cols×rows × 8×16 px)
    4. For each grid cell, extract the 16×8 pixel patch and pair it with
       the character jp2a chose for that cell
       
    Returns (patches, labels) lists — nothing is written to disk.
    """
    # Get gold-standard ASCII conversion (stays in memory)
    ascii_text = convert_with_jp2a(image_path, cols)
    lines = ascii_text.rstrip("\n").split("\n")
    lines = [line for line in lines if line]  # Drop empty lines

    if not lines:
        return [], []

    rows = len(lines)
    actual_cols = max(len(line) for line in lines)

    # Pad shorter lines to uniform width
    lines = [line.ljust(actual_cols) for line in lines]

    # Load original image as grayscale, resize to match the ASCII grid
    img = Image.open(image_path).convert("L")
    target_w = actual_cols * PATCH_W
    target_h = rows * PATCH_H
    img = img.resize((target_w, target_h), Image.LANCZOS)
    img_np = np.array(img, dtype=np.float32) / 255.0

    patches = []
    labels = []

    for r in range(rows):
        for c in range(actual_cols):
            char = lines[r][c]

            # Skip any character jp2a outputs that isn't in our set
            if char not in CHAR_TO_IDX:
                continue

            y0 = r * PATCH_H
            x0 = c * PATCH_W
            patch = img_np[y0 : y0 + PATCH_H, x0 : x0 + PATCH_W]

            if patch.shape == (PATCH_H, PATCH_W):
                patches.append(patch)
                labels.append(CHAR_TO_IDX[char])

    return patches, labels


# ── Model Loading ──────────────────────────────────────────────────────

def build_model(num_classes):
    """Rebuild model architecture (must match train_model.py exactly)."""
    return models.Sequential(
        [
            layers.Input(shape=(PATCH_H, PATCH_W, 1)),
            layers.Conv2D(16, kernel_size=3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 1)),
            layers.Conv2D(32, kernel_size=3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )


def load_or_build_model():
    """
    Try to load the pre-trained Keras model for fine-tuning.
    Falls back to rebuilding with random weights if no checkpoint exists.
    """
    model_path = os.path.join(SCRIPT_DIR, "ascii_cam_model", "model.keras")

    if os.path.exists(model_path):
        print(f"✅ Loaded pre-trained model from model.keras")
        return keras.models.load_model(model_path)

    # Fallback: rebuild with random weights (still works, just starts cold)
    print("⚠️  No model.keras found — rebuilding with random weights.")
    print("   For better results, re-run train_model.py first to generate model.keras.\n")
    return build_model(len(ASCII_CHARS))


# ── Main Pipeline ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune ASCII cam model on real-world images using jp2a"
    )
    parser.add_argument("image_folder", help="Path to folder of real-world images")
    parser.add_argument(
        "--cols", type=int, default=150, help="ASCII grid width in columns (default: 150)"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Fine-tuning epochs (default: 10)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="Learning rate (default: 0.0001)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--mix-synthetic",
        type=float,
        default=0.3,
        help="Fraction of synthetic data to mix in to prevent catastrophic forgetting (default: 0.3)",
    )
    args = parser.parse_args()

    # ── Preflight checks ───────────────────────────────────────────────
    if not check_jp2a():
        print("❌ jp2a is not installed. Install it with:")
        print("   brew install jp2a")
        sys.exit(1)
    print("✅ jp2a found\n")

    # ── Discover images ────────────────────────────────────────────────
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.tiff", "*.gif")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(args.image_folder, ext)))
        image_paths.extend(glob.glob(os.path.join(args.image_folder, ext.upper())))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"❌ No images found in {args.image_folder}")
        sys.exit(1)

    print(f"📷 Found {len(image_paths)} images in {args.image_folder}\n")

    # ── Process images through jp2a → extract patches ──────────────────
    all_patches = []
    all_labels = []
    preview_shown = False

    for i, path in enumerate(image_paths):
        name = os.path.basename(path)
        try:
            patches, labels = process_image(path, args.cols)
            all_patches.extend(patches)
            all_labels.extend(labels)

            # Show a small preview of jp2a's output for the first image
            if not preview_shown and patches:
                preview = convert_with_jp2a(path, 60)
                print(f"  Preview of jp2a gold-standard ({name}):")
                preview_lines = preview.strip().split("\n")
                for line in preview_lines[:12]:
                    print(f"    {line}")
                if len(preview_lines) > 12:
                    print(f"    ... ({len(preview_lines)} total rows)\n")
                preview_shown = True

            print(f"  [{i + 1}/{len(image_paths)}] {name} → {len(patches):,} patches")
        except Exception as e:
            print(f"  [{i + 1}/{len(image_paths)}] {name} → SKIPPED ({e})")

    if not all_patches:
        print("\n❌ No training patches were generated. Check your images and jp2a.")
        sys.exit(1)

    # ── Build training arrays (all in memory) ──────────────────────────
    X_real = np.expand_dims(np.array(all_patches, dtype=np.float32), axis=-1)
    y_real = np.array(all_labels, dtype=np.int32)
    print(f"\n📊 Real-world patches: {len(X_real):,}")

    # Show character distribution so user can see balance
    unique, counts = np.unique(y_real, return_counts=True)
    print("\n   Character distribution from jp2a:")
    for idx, count in zip(unique, counts):
        bar = "█" * max(1, count * 40 // counts.max())
        print(f"   '{ASCII_CHARS[idx]:>1s}' {bar} {count:,}")

    # ── Mix in synthetic data to prevent catastrophic forgetting ───────
    if args.mix_synthetic > 0:
        X_synth_path = os.path.join(SCRIPT_DIR, "dataset", "X_data.npy")
        y_synth_path = os.path.join(SCRIPT_DIR, "dataset", "y_data.npy")

        if os.path.exists(X_synth_path) and os.path.exists(y_synth_path):
            X_synth = np.load(X_synth_path)
            y_synth = np.load(y_synth_path)

            n_synth = min(int(len(X_real) * args.mix_synthetic), len(X_synth))
            if n_synth > 0:
                synth_idx = np.random.choice(len(X_synth), n_synth, replace=False)
                X_real = np.concatenate([X_real, X_synth[synth_idx]])
                y_real = np.concatenate([y_real, y_synth[synth_idx]])
                print(f"\n🔀 Mixed in {n_synth:,} synthetic samples → {len(X_real):,} total")
        else:
            print("\n⚠️  No synthetic dataset found to mix in (dataset/X_data.npy)")

    # ── Shuffle and split ──────────────────────────────────────────────
    indices = np.arange(len(X_real))
    np.random.shuffle(indices)
    X_real = X_real[indices]
    y_real = y_real[indices]

    split = int(len(X_real) * 0.9)
    X_train, X_val = X_real[:split], X_real[split:]
    y_train, y_val = y_real[:split], y_real[split:]
    print(f"\n📚 Train: {len(X_train):,}  |  Val: {len(X_val):,}")

    # ── Load model and fine-tune ───────────────────────────────────────
    model = load_or_build_model()

    # Lower learning rate than initial training — standard for fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    print(f"\n🚀 Fine-tuning for {args.epochs} epochs (lr={args.lr})...\n")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # ── Save model (the ONLY files we write) ───────────────────────────
    model_dir = os.path.join(SCRIPT_DIR, "ascii_cam_model")
    os.makedirs(model_dir, exist_ok=True)

    keras_path = os.path.join(model_dir, "model.keras")
    onnx_path = os.path.join(model_dir, "model.onnx")

    model.save(keras_path)
    model.export(onnx_path, format="onnx")

    print(f"\n✅ Fine-tuned model saved!")
    print(f"   • {keras_path}  (for further fine-tuning)")
    print(f"   • {onnx_path}   (for web inference)")


if __name__ == "__main__":
    main()
