# Model Training & Export Pipeline

<!-- 
TUTOR HINT: Hey Eli! This is the documentation landing page for your model.
I've set up a human-friendly template for you to document the exact details of how you generated character patches, pre-trained the base network, and fine-tuned on real images. Fill in the placeholders below once you've polished the pipeline!
-->

This directory contains the Python pipeline used to train the deep learning model that classifies image patches into ASCII character densities.

## 🧠 Model Architecture & Concept

- **Input Dimension:** Grayscale patches of size `16` (height) × `8` (width) × `1` (channel).
- **Core Concept:** Rather than mapping simple brightness averages to characters (which lacks structure and texture detail), a Convolutional Neural Network (CNN) is trained to classify pixel configurations into ASCII characters.
- **Pipeline Stages:**
  1. **Synthetic Data Generation (`generate_data.py`):** Renders font faces onto a blank canvas, applies random augmentations (rotations, scale, noise, blur), and saves as `.npy` arrays.
  2. **Base Pre-Training (`train_model.py`):** Trains the CNN on the synthetic font character variations.
  3. **COCO Fine-Tuning (`fine_tune.py`):** Connects to `jp2a` (a standard image-to-ASCII tool) to extract "gold-standard" target characters from real COCO images, extracts corresponding image patches, balances the class distribution using smart sampling, and fine-tunes the network.
  4. **ONNX Export:** Automatically exports the trained Keras model into the standard `.onnx` format for runtime execution in browsers and desktop apps.

---

## 🛠️ How to Run the Pipeline

### 1. Set Up Environment
Use `uv` (or standard `pip` in your virtual environment) to install the dependencies:
```bash
# Add dependencies
uv pip install keras torch numpy opencv-python pillow onnxruntime
```

### 2. Generate Synthetic Dataset
Build the base character training arrays (`X_data.npy` and `y_data.npy`):
```bash
python generate_data.py
```

### 3. Pre-Train the CNN
Train the network on synthetic font layouts:
```bash
python train_model.py
```

### 4. Fine-Tune on Real Images
Fine-tune the model using COCO dataset images to align the network with real-world shading and lines:
```bash
python fine_tune.py images/ --epochs 8 --max-per-char 5
```

---

## 🚀 Pushing to Hugging Face Hub

We use the Hugging Face Hub to host and distribute the final ONNX and Keras weights.

1. Ensure the `huggingface_hub` SDK is installed.
2. Authenticate using your write token:
   ```bash
   huggingface-cli login
   ```
3. Open `push_to_hub.py`, fill in your username/repository name, and run:
   ```bash
   python push_to_hub.py
   ```

<!-- 
TUTOR HINT: You can add training metrics, accuracy logs, and loss curves here later to make it look super professional for sharing! 
-->
