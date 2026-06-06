// app.js — Live Webcam-to-ASCII Pipeline via ONNX Runtime Web

// Configure ONNX WebAssembly CDN paths
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

// Dom Elements
const video = document.getElementById('webcam-video');
const canvas = document.getElementById('preprocess-canvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const asciiOutput = document.getElementById('ascii-output');
const videoOverlay = document.getElementById('video-overlay');

const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');

const colsSelect = document.getElementById('cols-select');
const colsVal = document.getElementById('cols-val');
const themeSelect = document.getElementById('theme-select');
const zoomSlider = document.getElementById('zoom-slider');
const invertCheckbox = document.getElementById('invert-checkbox');
const mirrorCheckbox = document.getElementById('mirror-checkbox');
const playPauseBtn = document.getElementById('play-pause-btn');
const copyBtn = document.getElementById('copy-btn');

// Stats Elements
const fpsVal = document.getElementById('fps-value');
const inferenceTimeVal = document.getElementById('inference-time');
const preprocessTimeVal = document.getElementById('preprocess-time');
const totalTimeVal = document.getElementById('total-time');
const resolutionVal = document.getElementById('resolution-value');

// App State
let ortSession = null;
let asciiCharsDefault = [];
let asciiChars = [];
let isStreaming = false;
let isModelLoaded = false;
let animationFrameId = null;

// Stats Tracking
let lastFrameTime = performance.now();
let frameCount = 0;
let fpsIntervalStart = performance.now();

// 1. Initial configuration and setup
async function init() {
  updateStatus('loading', 'Loading configuration & model...');
  
  // Load config.json
  try {
    const response = await fetch('/config.json');
    if (!response.ok) throw new Error('Failed to fetch config');
    const config = await response.json();
    asciiCharsDefault = config.ASCII_CHARS;
    console.log('📝 Loaded character ramp from config.json:', asciiCharsDefault);
  } catch (e) {
    console.warn('⚠️ Could not load config.json, using fallback ramp:', e);
    asciiCharsDefault = [" ", ".", ",", "-", "~", ":", "i", "r", "s", "t", "l", "C", "O", "Z", "w", "m", "#", "8", "%", "@"];
  }
  
  updateCharRamp();
  
  // Initialize zoom slider font style
  updateZoom(zoomSlider.value);
  
  // Load ONNX Model
  try {
    const modelPath = '/ascii_cam_model/model.onnx';
    console.log(`🧠 Loading ONNX model from: ${modelPath}`);
    
    // Create ONNX Runtime Session using WebGL/WebGPU if available, fallback to WASM
    // We prefer CPU (wasm) because batch inference is light, and WASM has highest compatibility.
    ortSession = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['wasm']
    });
    
    isModelLoaded = true;
    console.log('🚀 ONNX Session created successfully!');
    updateStatus('ready', 'Model Ready. Requesting camera...');
    
    // Start Webcam
    await startCamera();
  } catch (err) {
    console.error('❌ Error initializing pipeline:', err);
    updateStatus('error', `Pipeline load failed: ${err.message}`);
  }
}

// 2. Camera Management
async function startCamera() {
  try {
    const constraints = {
      audio: false,
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: 'user'
      }
    };
    
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    
    // Wait for metadata to load to get dimensions
    video.onloadedmetadata = () => {
      video.play();
      isStreaming = true;
      videoOverlay.classList.add('hidden');
      playPauseBtn.disabled = false;
      playPauseBtn.querySelector('.btn-text').textContent = 'Pause Stream';
      playPauseBtn.querySelector('.btn-icon').textContent = '⏸';
      
      updateStatus('active', 'Camera Connected — Live Stream Active');
      
      // Start processing loop
      lastFrameTime = performance.now();
      frameCount = 0;
      fpsIntervalStart = performance.now();
      processFrame();
    };
  } catch (err) {
    console.error('❌ Camera access denied:', err);
    updateStatus('error', 'Camera access denied. Please allow camera permissions.');
  }
}

function stopCamera() {
  if (video.srcObject) {
    const tracks = video.srcObject.getTracks();
    tracks.forEach(track => track.stop());
    video.srcObject = null;
  }
  isStreaming = false;
  videoOverlay.classList.remove('hidden');
  playPauseBtn.querySelector('.btn-text').textContent = 'Resume Stream';
  playPauseBtn.querySelector('.btn-icon').textContent = '▶';
  updateStatus('ready', 'Camera Stream Paused');
}

// 3. Core Preprocessing & Inference Loop
async function processFrame() {
  if (!isStreaming || !isModelLoaded) return;
  
  const startTotal = performance.now();
  
  // Dimensions
  const cols = parseInt(colsSelect.value);
  const videoWidth = video.videoWidth || 640;
  const videoHeight = video.videoHeight || 480;
  const aspect = videoWidth / videoHeight;
  
  // Character cell is 8px wide by 16px high (aspect ratio 1:2)
  // Compute rows dynamically to preserve physical dimensions:
  const rows = Math.max(1, Math.round((cols * 0.5) / aspect));
  resolutionVal.textContent = `${cols} × ${rows}`;
  
  const targetW = cols * 8;
  const targetH = rows * 16;
  
  // Set canvas size for block processing
  canvas.width = targetW;
  canvas.height = targetH;
  
  // A. PREPROCESSING
  const startPreprocess = performance.now();
  
  // Draw current frame scaled to exact patch resolution
  // Mirror the drawing if mirrorCheckbox is checked to match the preview
  if (mirrorCheckbox.checked) {
    ctx.save();
    ctx.translate(targetW, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, targetW, targetH);
    ctx.restore();
  } else {
    ctx.drawImage(video, 0, 0, targetW, targetH);
  }
  
  // Fetch pixel buffer
  const imgData = ctx.getImageData(0, 0, targetW, targetH).data;
  
  // Prepare tensor data: shape (batchSize, height:16, width:8, channels:1)
  const batchSize = rows * cols;
  const inputBuffer = new Float32Array(batchSize * 16 * 8 * 1);
  
  let bufferOffset = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      // Extract a 16x8 block
      for (let py = 0; py < 16; py++) {
        const y = r * 16 + py;
        const rowOffset = y * targetW;
        for (let px = 0; px < 8; px++) {
          const x = c * 8 + px;
          const pixelIndex = (rowOffset + x) * 4;
          
          // Compute grayscale value (Standard Luma coefficients)
          const red = imgData[pixelIndex];
          const green = imgData[pixelIndex + 1];
          const blue = imgData[pixelIndex + 2];
          const gray = (0.299 * red + 0.587 * green + 0.114 * blue) / 255.0;
          
          inputBuffer[bufferOffset++] = gray;
        }
      }
    }
  }
  
  const preprocessTime = performance.now() - startPreprocess;
  
  // B. INFERENCE
  const startInference = performance.now();
  let asciiStr = '';
  
  try {
    // Construct ONNX Tensor
    const inputName = ortSession.inputNames[0];
    const outputName = ortSession.outputNames[0];
    const inputTensor = new ort.Tensor('float32', inputBuffer, [batchSize, 16, 8, 1]);
    
    // Run Inference
    const outputMap = await ortSession.run({ [inputName]: inputTensor });
    const outputTensor = outputMap[outputName];
    const outputData = outputTensor.data; // Float32Array of size batchSize * 20
    
    const inferenceTime = performance.now() - startInference;
    
    // C. ARGMAX & TEXT MAPPING
    const numClasses = 20; // from output dimension
    const asciiRows = [];
    
    for (let r = 0; r < rows; r++) {
      let rowStr = '';
      for (let c = 0; c < cols; c++) {
        const patchIdx = r * cols + c;
        const startIdx = patchIdx * numClasses;
        
        let maxVal = -Infinity;
        let maxIdx = 0;
        for (let i = 0; i < numClasses; i++) {
          const val = outputData[startIdx + i];
          if (val > maxVal) {
            maxVal = val;
            maxIdx = i;
          }
        }
        rowStr += asciiChars[maxIdx];
      }
      asciiRows.push(rowStr);
    }
    asciiStr = asciiRows.join('\n');
    
    // Render ASCII output
    asciiOutput.textContent = asciiStr;
    
    // Update Metrics
    const totalTime = performance.now() - startTotal;
    
    preprocessTimeVal.textContent = `${preprocessTime.toFixed(1)} ms`;
    inferenceTimeVal.textContent = `${inferenceTime.toFixed(1)} ms`;
    totalTimeVal.textContent = `${totalTime.toFixed(1)} ms`;
    
    // FPS counter calculation
    frameCount++;
    const now = performance.now();
    const elapsedFpsTime = now - fpsIntervalStart;
    if (elapsedFpsTime >= 1000) {
      const fps = (frameCount * 1000) / elapsedFpsTime;
      fpsVal.textContent = fps.toFixed(1);
      frameCount = 0;
      fpsIntervalStart = now;
    }
    
  } catch (err) {
    console.error('❌ Inference error:', err);
    asciiOutput.textContent = `[Inference Error: ${err.message}]`;
    stopCamera();
    updateStatus('error', `Inference failed: ${err.message}`);
    return;
  }
  
  // Queue next frame
  if (isStreaming) {
    animationFrameId = requestAnimationFrame(processFrame);
  }
}

// 4. Utility / UI Handlers
function updateStatus(state, msg) {
  statusText.textContent = msg;
  statusDot.className = 'status-dot';
  
  if (state === 'loading') {
    statusDot.classList.add('pulse');
    statusDot.style.backgroundColor = '#ffcc00'; // Amber
  } else if (state === 'ready') {
    statusDot.style.backgroundColor = '#0088ff'; // Blue
  } else if (state === 'active') {
    statusDot.classList.add('ready'); // Neon accent
  } else if (state === 'error') {
    statusDot.style.backgroundColor = '#ff3366'; // Red
  }
}

function updateCharRamp() {
  const isInverted = invertCheckbox.checked;
  // If inverted, reverse the list so light is mapped to dense and vice versa
  asciiChars = isInverted 
    ? [...asciiCharsDefault].reverse() 
    : [...asciiCharsDefault];
}

function updateZoom(size) {
  asciiOutput.style.fontSize = `${size}px`;
}

// 5. Event Listeners
colsSelect.addEventListener('input', (e) => {
  colsVal.textContent = e.target.value;
});

themeSelect.addEventListener('change', (e) => {
  // Remove existing themes
  document.body.className = '';
  // Add new theme class
  document.body.classList.add(e.target.value);
});

zoomSlider.addEventListener('input', (e) => {
  updateZoom(e.target.value);
});

invertCheckbox.addEventListener('change', () => {
  updateCharRamp();
  console.log('🔄 Character mapping inversion updated.');
});

mirrorCheckbox.addEventListener('change', (e) => {
  if (e.target.checked) {
    video.classList.add('mirror');
  } else {
    video.classList.remove('mirror');
  }
});

playPauseBtn.addEventListener('click', () => {
  if (isStreaming) {
    isStreaming = false;
    if (animationFrameId) {
      cancelAnimationFrame(animationFrameId);
      animationFrameId = null;
    }
    stopCamera();
  } else {
    isStreaming = true;
    startCamera();
  }
});

copyBtn.addEventListener('click', async () => {
  const text = asciiOutput.textContent;
  if (!text || text.startsWith('Loading') || text.startsWith('[Inference')) return;
  
  try {
    await navigator.clipboard.writeText(text);
    const originalText = copyBtn.innerHTML;
    copyBtn.innerHTML = '<span>✅</span> Copied!';
    setTimeout(() => {
      copyBtn.innerHTML = originalText;
    }, 2000);
  } catch (err) {
    console.error('Failed to copy text: ', err);
  }
});

// Setup mirroring class onload since it's checked by default
if (mirrorCheckbox.checked) {
  video.classList.add('mirror');
}

// Start app
window.addEventListener('DOMContentLoaded', init);
