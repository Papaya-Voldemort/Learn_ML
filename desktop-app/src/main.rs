// src/main.rs — Rust ASCII Camera GUI App Template
//
// TUTOR HINT: Hey Eli! Here is the skeleton for your Rust egui application.
// It integrates camera stream capture using `nokhwa` and sets up the user interface.
// Once you install Rust, you can fill in the core processing logic or run it directly to check the layout!

use eframe::egui;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use std::time::Instant;

// ASCII Character Density Ramp (matches config.json)
const ASCII_CHARS: &[char] = &[
    ' ', '.', ',', '-', '~', ':', 'i', 'r', 's', 't', 'l', 'C', 'O', 'Z', 'w', 'm', '#', '8', '%', '@'
];

struct AsciiCamApp {
    // Resolution control (columns)
    columns: usize,
    // Whether the camera is active
    is_active: bool,
    // Current rendered ASCII art output
    ascii_output: String,
    // Channel receiver for camera frames
    frame_rx: Option<Receiver<String>>,
    // Performance benchmarks
    fps: f64,
    last_frame_time: Instant,
}

impl Default for AsciiCamApp {
    fn default() -> Self {
        Self {
            columns: 80,
            is_active: true,
            ascii_output: "Loading camera feed...".to_string(),
            frame_rx: None,
            fps: 0.0,
            last_frame_time: Instant::now(),
        }
    }
}

impl AsciiCamApp {
    fn start_camera_thread(&mut self, columns: usize) {
        let (tx, rx) = channel();
        self.frame_rx = Some(rx);

        // TODO: Spawn a background thread to capture webcam frames.
        // Capturing in a background thread keeps the GUI rendering smooth (at 60+ FPS).
        thread::spawn(move || {
            println!("🎥 Camera thread started.");
            
            // Basic nokhwa setup example:
            // use nokhwa::pixel_format::RgbFormat;
            // use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
            // use nokhwa::Camera;
            //
            // let index = CameraIndex::Index(0);
            // let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
            // let mut camera = Camera::new(index, requested).expect("Failed to open camera");
            // camera.open_stream().expect("Failed to start stream");
            
            loop {
                // Mock frame loop: In a real app, capture frame, resize it, and map pixels to characters.
                // let frame = camera.frame().unwrap();
                // let img = frame.decode_image::<RgbFormat>().unwrap();
                
                // Let's create mock ASCII frame for visual feedback in template
                let mock_frame = generate_mock_ascii(columns);
                if tx.send(mock_frame).is_err() {
                    break; // Channel closed, app terminated
                }
                
                // Sleep to match webcam frame rate (approx. 30 FPS)
                thread::sleep(std::time::Duration::from_millis(33));
            }
        });
    }
}

// Simple placeholder mapping logic
fn generate_mock_ascii(cols: usize) -> String {
    let mut result = String::new();
    let rows = cols / 2;
    for r in 0..rows {
        for c in 0..cols {
            // Pick a moving character to simulate live video
            let char_idx = (r + c + (Instant::now().elapsed().as_millis() as usize / 100)) % ASCII_CHARS.len();
            result.push(ASCII_CHARS[char_idx]);
        }
        result.push('\n');
    }
    result
}

impl eframe::App for AsciiCamApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Start camera thread if it hasn't been initialized yet
        if self.frame_rx.is_none() && self.is_active {
            self.start_camera_thread(self.columns);
        }

        // Receive any new frames from the background thread
        if let Some(ref rx) = self.frame_rx {
            while let Ok(frame) = rx.try_recv() {
                self.ascii_output = frame;
                
                // Calculate FPS
                let now = Instant::now();
                let delta = now.duration_since(self.last_frame_time).as_secs_f64();
                if delta > 0.0 {
                    self.fps = 0.9 * self.fps + 0.1 * (1.0 / delta);
                }
                self.last_frame_time = now;
            }
        }

        // --- Layout Design ---
        egui::SidePanel::left("controls").show(ctx, |ui| {
            ui.heading("🎛️ Settings");
            ui.add_space(10.0);

            // Toggle Camera
            if ui.checkbox(&mut self.is_active, "Enable Camera").changed() {
                if !self.is_active {
                    self.frame_rx = None;
                    self.ascii_output = "Camera offline".to_string();
                }
            }

            ui.add_space(10.0);

            // Columns slider
            ui.label("Resolution (Columns):");
            if ui.add(egui::Slider::new(&mut self.columns, 40..=160).step_by(10.0)).changed() {
                if self.is_active {
                    // Restart camera thread with new columns resolution
                    self.start_camera_thread(self.columns);
                }
            }

            ui.add_space(20.0);
            ui.separator();
            ui.add_space(10.0);

            // Metrics
            ui.label("Performance metrics:");
            ui.colored_label(egui::Color32::from_rgb(0, 255, 204), format!("FPS: {:.1}", self.fps));
            ui.label(format!("Grid: {} columns", self.columns));

            ui.add_space(20.0);
            ui.separator();
            ui.add_space(10.0);

            ui.heading("🔗 Join Calls");
            ui.label("To use this as a filter in Zoom/Teams:");
            ui.small("1. Open OBS Studio.");
            ui.small("2. Add a 'Window Capture' source.");
            ui.small("3. Select this application window.");
            ui.small("4. Click 'Start Virtual Camera' in OBS.");
            ui.small("5. Choose 'OBS Virtual Camera' in your meeting app!");
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // Render the ASCII text using a monospace font style
            egui::ScrollArea::both().show(ui, |ui| {
                ui.add(
                    egui::TextEdit::multiline(&mut self.ascii_output)
                        .font(egui::TextStyle::Monospace)
                        .code_editor() // uses monospace font layout
                        .lock_focus(true)
                );
            });
        });

        // Request a repaint to keep the background camera stream updating the UI
        ctx.request_repaint();
    }
}

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Neural ASCII Camera")
            .with_inner_size([900.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "ascii_cam_app",
        options,
        Box::new(|_cc| Ok(Box::<AsciiCamApp>::default())),
    )
}
