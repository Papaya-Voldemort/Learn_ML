import os
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# 1. SETUP CONSTANTS
PATCH_WIDTH = 8
PATCH_HEIGHT = 16
SAMPLES_PER_CHAR = 500  # 13 characters * 500 variations = 6,500 total image patches

# A structurally balanced set of characters ordered from lowest to highest visual density
ASCII_CHARS = [" ", ".", "-", "=", "+", "*", "x", "%", "#", "@", "/", "\\", "|"]

DATA_DIR = "dataset"
os.makedirs(DATA_DIR, exist_ok=True)

def create_base_char_image(char, font_size=14):
    """Renders a single crisp, white ASCII character on a black canvas."""
    # Create a canvas twice as big so we can rotate it without clipping edges
    img = Image.new("L", (PATCH_WIDTH * 2, PATCH_HEIGHT * 2), color=0)
    draw = ImageDraw.Draw(img)
    
    try:
        # Standard monospace font available on macOS
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Courier New.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
        
    # Center the character perfectly on the oversized canvas
    left, top, right, bottom = draw.textbbox((0, 0), char, font=font)
    text_w = right - left
    text_h = bottom - top
    
    x = (PATCH_WIDTH * 2 - text_w) // 2
    y = (PATCH_HEIGHT * 2 - text_h) // 2
    
    draw.text((x, y), char, fill=255, font=font)
    return img

def apply_augmentations(pil_img):
    """Distorts the clean character image to teach the AI real-world visual texture."""
    img_np = np.array(pil_img)
    
    # 1. Random Rotation (Simulates crooked camera angles or slanted lines)
    angle = random.uniform(-10, 10)
    matrix = cv2.getRotationMatrix2D((PATCH_WIDTH, PATCH_HEIGHT), angle, 1.0)
    img_np = cv2.warpAffine(img_np, matrix, (PATCH_WIDTH * 2, PATCH_HEIGHT * 2))
    
    # Crop down from the double-sized center canvas to our exact target size (16x8)
    start_x = PATCH_WIDTH // 2
    start_y = PATCH_HEIGHT // 2
    img_np = img_np[start_y:start_y+PATCH_HEIGHT, start_x:start_x+PATCH_WIDTH]
    
    # 2. Random Contrast & Brightness adjustments
    alpha = random.uniform(0.5, 1.5)  # Contrast multiplier
    beta = random.randint(-30, 30)     # Brightness offset
    img_np = cv2.convertScaleAbs(img_np, alpha=alpha, beta=beta)
    
    # 3. Random Blur (Teaches the model to recognize shapes even out of focus)
    if random.random() > 0.5:
        kernel_size = random.choice([3, 5])
        img_np = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)
        
    # 4. Add Random Digital Noise
    if random.random() > 0.5:
        noise = np.random.normal(0, random.uniform(5, 15), img_np.shape).astype(np.float32)
        img_np = np.clip(img_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
    return img_np

print("🚀 Launching synthetic dataset factory...")

X_data = []
y_data = []

for char_idx, char in enumerate(ASCII_CHARS):
    print(f"Manufacturing variations for: '{char}' ({char_idx + 1}/{len(ASCII_CHARS)})")
    base_img = create_base_char_image(char)
    
    for _ in range(SAMPLES_PER_CHAR):
        augmented_patch = apply_augmentations(base_img)
        
        # Scale pixel integers (0-255) down to floats (0.0-1.0) for optimal network training
        normalized_patch = augmented_patch.astype(np.float32) / 255.0
        
        X_data.append(normalized_patch)
        y_data.append(char_idx)

# Transform raw python lists into high-performance NumPy arrays
X_data = np.array(X_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.int32)

# Reshape data to include the mandatory channel dimension: (Samples, Height, Width, Channels)
X_data = np.expand_dims(X_data, axis=-1)

# Write arrays directly to local binary files
np.save(os.path.join(DATA_DIR, "X_data.npy"), X_data)
np.save(os.path.join(DATA_DIR, "y_data.npy"), y_data)

print(f"\n✅ Pipeline complete! Processed {len(X_data)} training patches.")
print(f"Matrix Shapes ➔ Features: {X_data.shape} | Labels: {y_data.shape}")