import os
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# 1. SETUP CONSTANTS
PATCH_WIDTH = 8
PATCH_HEIGHT = 16
SAMPLES_PER_CHAR = 500  

# A structurally balanced set of characters ordered from lowest to highest visual density
try:
    import json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        ASCII_CHARS = json.load(f)["ASCII_CHARS"]
except Exception as e:
    print(f"⚠️ Could not load config.json ({e}), using default fallback.")
    ASCII_CHARS = [" ", ".", "-", "=", "+", "*", "x", "%", "#", "@", "/", "\\", "|"]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
os.makedirs(DATA_DIR, exist_ok=True)

def create_base_char_image(char, font_size=14):
    """Renders a single crisp, white ASCII character on a black canvas."""
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
    img_np = np.array(pil_img)
    
    canvas_h, canvas_w = img_np.shape
    angle = random.uniform(-10, 10)
    matrix = cv2.getRotationMatrix2D((canvas_w // 2, canvas_h // 2), angle, 1.0)
    img_np = cv2.warpAffine(img_np, matrix, (canvas_w, canvas_h))
    
    start_x = (canvas_w - PATCH_WIDTH) // 2
    start_y = (canvas_h - PATCH_HEIGHT) // 2
    img_np = img_np[start_y:start_y+PATCH_HEIGHT, start_x:start_x+PATCH_WIDTH]
        
    if random.random() > 0.3:
        grid_y, grid_x = np.mgrid[0:PATCH_HEIGHT, 0:PATCH_WIDTH]
        gradient = (grid_y * random.uniform(-2, 2) + grid_x * random.uniform(-2, 2)).astype(np.uint8)
        img_np = cv2.addWeighted(img_np, 0.8, gradient, 0.2, 0)
    
    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-20, 20)
    img_np = cv2.convertScaleAbs(img_np, alpha=alpha, beta=beta)
    
    if random.random() > 0.5:
        kernel_size = random.choice([3, 5])
        img_np = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)
        
    if random.random() > 0.5:
        noise = np.random.normal(0, random.uniform(3, 12), img_np.shape).astype(np.float32)
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