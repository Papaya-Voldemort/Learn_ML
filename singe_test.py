import os
import numpy as np
import cv2
import onnxruntime as ort

# The exact character list from our training data layout
ASCII_CHARS = [" ", ".", "-", "=", "+", "*", "x", "%", "#", "@", "/", "\\", "|"]

PATCH_W = 8
PATCH_HEIGHT = 16

# 1. LOAD THE UNIVERSAL ONNX BRAIN
# Look inside your exported folder for the actual model.onnx file
onnx_path = "ascii_cam_model/model.onnx" 

if not os.path.exists(onnx_path):
    raise FileNotFoundError(
        f"❌ Can't find ONNX file at '{onnx_path}'!\n"
        "Make sure you ran train_model.py and it generated the ascii_cam_model folder."
    )

print("🧠 Loading universal ONNX model session...")
session = ort.InferenceSession(onnx_path)

# Get the internal variable names your ONNX model expects for input and output layers
input_name = session.get_inputs()[0].name

# 2. LOAD YOUR PHOTO
image_path = "me.png" 
if not os.path.exists(image_path):
    raise FileNotFoundError(f"❌ Could not find '{image_path}'! Please drop a picture in this folder named me.jpg.")

# Read the image in grayscale (black and white)
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Resize the image so it fits nicely inside a standard terminal window
# We aim for an output layout roughly 60 text columns wide
target_cols = 150
target_width = target_cols * PATCH_W

# Calculate height to preserve your original photo's aspect ratio
aspect_ratio = img.shape[0] / img.shape[1]
target_rows = int(target_cols * aspect_ratio * (PATCH_W / PATCH_HEIGHT))
target_height = target_rows * PATCH_HEIGHT

img_resized = cv2.resize(img, (target_width, target_height))
img_resized = cv2.Canny(img_resized, 50, 150)

print(f"📷 Processing photo size: {img.shape} -> Resized to grid: {target_rows}x{target_cols}")

# 3. CHOP THE PHOTO AND RUN PREDICTIONS VIA ONNX
ascii_output = []

print("⏳ Running neural inference across image blocks...")
for row in range(target_rows):
    line_chars = ""
    for col in range(target_cols):
        # Calculate the exact bounding pixel coordinates for this specific patch
        y1 = row * PATCH_HEIGHT
        y2 = y1 + PATCH_HEIGHT
        x1 = col * PATCH_W
        x2 = x1 + PATCH_W
        
        # Crop out the 16x8 slice
        patch = img_resized[y1:y2, x1:x2]
        
        # Scale integers (0-255) down to floats (0.0-1.0) to match training specs
        normalized_patch = patch.astype(np.float32) / 255.0
        
        # Reshape to (1, 16, 8, 1) -> Batch size of 1, Height, Width, Grayscale Channel
        input_tensor = np.expand_dims(normalized_patch, axis=(0, -1))
        
        # Pass the matrix directly to the ONNX runtime engine session
        preds = session.run(None, {input_name: input_tensor})[0]
        
        # Locate the character index with the highest probability score
        predicted_class = np.argmax(preds[0])
        
        # Append the character symbol to our current text line array
        line_chars += ASCII_CHARS[predicted_class]
        
    ascii_output.append(line_chars)

# 4. PRINT YOUR FACE IN THE TERMINAL!
print("\n--- NEURAL ASCII OUTPUT ---")
for line in ascii_output:
    print(line)
print("----------------------------\n")