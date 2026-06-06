# Rust Desktop ASCII Camera (Filter Tool)

This folder contains a template for a Rust GUI application that captures your camera, converts the stream into real-time ASCII art, and renders it in a beautiful, high-performance window using `egui` and `eframe`.

---

## 🛠️ Requirements & Setup

To compile and run this desktop tool, you will need:
1. **Rust & Cargo:** Installed via [rustup](https://rustup.rs/):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```
2. **System Dependencies:**
   - **macOS:** Native AVFoundation is used by `nokhwa`. No extra system packages are required.
   - **Linux:** Install `libudev` and `pkg-config`:
     ```bash
     sudo apt-get install libudev-dev pkg-config
     ```
   - **Windows:** Standard MSVC toolchain.

---

## 🚀 How to Run

1. Navigate to this folder:
   ```bash
   cd desktop-app
   ```
2. Compile and run in development mode:
   ```bash
   cargo run
   ```
3. Build a highly optimized release binary:
   ```bash
   cargo build --release
   ```
   The compiled binary will be located at `target/release/ascii-cam-filter`.

---

## 🎥 Using as a Video Call Filter

Since custom virtual camera driver registration on modern OSs (especially macOS 12+ Camera Extensions) is tightly controlled and requires code-signing certificates, the industry-standard way to route your filter output into video meetings (Zoom, Microsoft Teams, Discord, Skype) is via **OBS Studio**:

1. **Download OBS Studio:** If you don't have it, download and install [OBS Studio](https://obsproject.com/).
2. **Launch the ASCII Cam App:** Compile and run `cargo run --release`. Keep this window open.
3. **Set Up OBS Capture:**
   - Open OBS Studio.
   - Under **Sources**, click the `+` button and select **Window Capture** (or **Screen Capture**).
   - Choose the `Neural ASCII Camera` window as the target.
   - Resize/crop the capture in OBS so it fills the canvas.
4. **Start Virtual Camera:**
   - Click the **Start Virtual Camera** button in the lower-right control panel of OBS.
5. **Join your Video Call:**
   - Open Zoom, Teams, or Discord.
   - Go to Video settings and change your camera input source to **OBS Virtual Camera**.
   - Your video stream is now a live ASCII art feed!

---

## 📝 Next Steps for Implementation

- **Webcam Integration:** Uncomment the `nokhwa` capture blocks in `src/main.rs`.
- **Character Mapping:** Inside the background thread, convert the raw RGB pixels to grayscale, scale the width/height to match your column size, and map pixel values to `ASCII_CHARS` indexes.
- **ONNX Model Inference (Optional):** If you'd like to run the model files inside Rust, add the `ort` crate to `Cargo.toml` and load `ascii_cam_model/model.onnx` to perform batch inferences on your image patches!
