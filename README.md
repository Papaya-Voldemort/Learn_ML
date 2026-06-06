# ASCII-Camera

Welcome to **ASCII-Camera**! This project is a neural-network and pixel-intensity powered real-time ASCII video filter. It enables converting webcam streams, images, and videos into stylized ASCII art.

The project is structured into three main components:

---

## 📂 Repository Structure

### 1. 🧠 [Model & Training Pipeline](file:///Users/elinelson/Documents/Development/Learn_ML/model-training)
- **Folder:** [`model-training/`](file:///Users/elinelson/Documents/Development/Learn_ML/model-training)
- **Role:** Python scripts to generate synthetic character variations, pre-train a CNN classifier, fine-tune using smart sampling against `jp2a` outputs on COCO dataset images, export to ONNX, and upload to the Hugging Face Hub.
- **Weights output:** Saves directly to the root [`ascii_cam_model/`](file:///Users/elinelson/Documents/Development/Learn_ML/ascii_cam_model) folder.

### 2. 🌐 [Web Application](file:///Users/elinelson/Documents/Development/Learn_ML/web-app)
- **Folder:** [`web-app/`](file:///Users/elinelson/Documents/Development/Learn_ML/web-app)
- **Role:** The Svelte-based static webpage. It loads the ONNX model from the root directory and runs real-time camera-to-ASCII classification completely client-side in the browser using ONNX Runtime Web.
- **Server:** A custom python server (`server.py`) is provided to inject CORS and COOP/COEP headers for high-performance WASM execution.

### 3. 🖥️ [Desktop Application (Spec & Template)](file:///Users/elinelson/Documents/Development/Learn_ML/desktop-app)
- **Folder:** [`desktop-app/`](file:///Users/elinelson/Documents/Development/Learn_ML/desktop-app)
- **Role:** A template folder for building a native GUI app in Rust using `egui` and `eframe`. It handles webcam capture and renders a high-performance ASCII stream that can be shared in video calls (Zoom, Teams, Discord) via OBS Studio.

### 4. 📦 [Active Model Weights](file:///Users/elinelson/Documents/Development/Learn_ML/ascii_cam_model)
- **Folder:** [`ascii_cam_model/`](file:///Users/elinelson/Documents/Development/Learn_ML/ascii_cam_model)
- **Role:** Houses the current production `.onnx` and `.keras` model weights, serving as a single source of truth for the training scripts, web interface, and desktop app.

---

## 🛠️ Quick Start

### Running the Web Demo locally
To run the browser-based camera feed:
```bash
# Run the developer server (injects COOP/COEP headers for multithreading)
python web-app/server.py
```
Then visit `http://localhost:8000` in your web browser.

### Building the Desktop Tool
Once you have Rust installed on your machine, you can build the native GUI tool:
```bash
cd desktop-app
cargo run --release
```
See the [`desktop-app/README.md`](file:///Users/elinelson/Documents/Development/Learn_ML/desktop-app/README.md) for instructions on routing the GUI to Zoom/Teams using OBS Studio!