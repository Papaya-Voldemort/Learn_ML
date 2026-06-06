# Model Training & Export Pipeline

This folder has the Python pipeline to make the date for and train the image to ASCII model.

> [HuggingFace](https://huggingface.co/Papaya-Voldemort/image-to-ascii)

## Model Architecture

- **Input:** The model takes small chunks of a image as input, this allows for a much smaller model! (Size: `16` (height) × `8` (width) × `1` (channel))
- **Output:** Normal programs for making ASCII art use brightness averages to characters which works but loses some dietel in more complex pieces (like humans). So a Convolutional Neural Network (CNN) is used to make the model more accurate and preserve the deteal of the image.

### Training Pipeline
- **Synthetic Data Generation** This is the first stage of training. It uses the `generate_data.py` script to create a dataset of synthetic images of characters.
- **Base Pre-Training** This is the second stage of training. It uses the `train_model.py` script to train the CNN on the synthetic font character variations.
- **COCO Fine-Tuning** This is the third stage of training. It uses the `fine_tune.py` script to fine-tune the model on real COCO images.

---

## How to Run the Pipeline

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

# Or use uv run
uv run generate_data.py
```

### 3. Pre-Train the CNN
Train the network on synthetic font layouts:
```bash
python train_model.py

uv run train_model.py
```

### 4. Fine-Tune on Real Images
Fine-tune the model using COCO dataset images to align the network with real-world shading and lines:
```bash
python fine_tune.py images/ --epochs 8 --max-per-char 5

uv run fine_tune.py images/ --epochs 8 --max-per-char 5
```

---

## Pushing to Hugging Face Hub

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