import os
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# 1. SETUP CONSTANTS
PATCH_WIDTH = 8
PATCH_HEIGHT = 16
SAMPLES_PER_CHAR = 500  

try:
    import json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        ASCII_CHARS = json.load(f)["ASCII_CHARS"]
except Exception as e:
    print(f"⚠️ Could not load config, using default.")
    ASCII_CHARS = [" ", ".", "-", "=", "+", "*", "x", "%", "#", "@", "/", "\\", "|"]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
os.makedirs(DATA_DIR, exist_ok=True)

def create_base_char_image(char, font_size=16):
    """Renders an ASCII character using standard top-left rendering (no forced centering)."""
    img = Image.new("L", (PATCH_WIDTH * 2, PATCH_HEIGHT * 2), color=0)
    draw = ImageDraw.Draw(img)
    
    try:
        # standard fallback, though PIL default works fine if ttf is missing
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Courier New.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
        
    # Draw character slightly offset, but NOT dynamically centered based on its bounding box
    draw.text((4, 4), char, fill=255, font=font)
    return img

def apply_augmentations(pil_img, is_solid=False):
    img_np = np.array(pil_img)
    canvas_h, canvas_w = img_np.shape
    
    if not is_solid:
        # Standard rotations and shifts for characters
        angle = random.uniform(-10, 10)
        matrix = cv2.getRotationMatrix2D((canvas_w // 2, canvas_h // 2), angle, 1.0)
        img_np = cv2.warpAffine(img_np, matrix, (canvas_w, canvas_h))
    
    # Crop down to our 8x16 target
    start_x = (canvas_w - PATCH_WIDTH) // 2
    start_y = (canvas_h - PATCH_HEIGHT) // 2
    img_np = img_np[start_y:start_y+PATCH_HEIGHT, start_x:start_x+PATCH_WIDTH]
        
    # Contrast & Brightness variations
    alpha = random.uniform(0.7, 1.3)
    beta = random.randint(-30, 30)
    img_np = cv2.convertScaleAbs(img_np, alpha=alpha, beta=beta)
    
    # Noise and blur
    if random.random() > 0.5:
        kernel_size = random.choice([3, 5])
        img_np = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)
        
    if random.random() > 0.5:
        noise = np.random.normal(0, random.uniform(2, 10), img_np.shape).astype(np.float32)
        img_np = np.clip(img_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
    return img_np

def main(samples_per_char=SAMPLES_PER_CHAR, ascii_chars=None, data_dir=None):
    if ascii_chars is None:
        ascii_chars = ASCII_CHARS
    if data_dir is None:
        data_dir = DATA_DIR

    print("🚀 Launching synthetic dataset factory...")
    
    X_data = []
    y_data = []
    
    for char_idx, char in enumerate(ascii_chars):
        print(f"Manufacturing variations for: '{char}' ({char_idx + 1}/{len(ascii_chars)})")
        
        # Special handling for spaces to include purely solid gray/black/white patches
        if char == " ":
            for _ in range(samples_per_char):
                base = Image.new("L", (PATCH_WIDTH * 2, PATCH_HEIGHT * 2), color=random.randint(0, 255))
                aug_patch = apply_augmentations(base, is_solid=True)
                X_data.append(aug_patch.astype(np.float32) / 255.0)
                y_data.append(char_idx)
        else:
            base_img = create_base_char_image(char)
            for _ in range(samples_per_char):
                aug_patch = apply_augmentations(base_img)
                X_data.append(aug_patch.astype(np.float32) / 255.0)
                y_data.append(char_idx)
    
    X_data = np.expand_dims(np.array(X_data, dtype=np.float32), axis=-1)
    y_data = np.array(y_data, dtype=np.int32)
    
    np.save(os.path.join(data_dir, "X_data.npy"), X_data)
    np.save(os.path.join(data_dir, "y_data.npy"), y_data)
    
    print(f"\n✅ Pipeline complete! Processed {len(X_data)} training patches.")

if __name__ == "__main__":
    main()