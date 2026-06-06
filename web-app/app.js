// app.js — Live Webcam-to-ASCII Pipeline via ONNX Runtime Web

// Configure ONNX WebAssembly CDN paths and performance parameters for optimal fallback
ort.env.logLevel = 'error';
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.0/dist/';
ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 4);
ort.env.wasm.simd = true;

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
const backendSelect = document.getElementById('backend-select');
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
let initialLoadDone = false;

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

  // Load ONNX Model using the selected backend
  const initialBackend = backendSelect ? backendSelect.value : 'wasm';
  await loadModel(initialBackend);
}

// 1.1 Model Loader with Dynamic Backend Support
async function loadModel(backend) {
  isModelLoaded = false;
  updateStatus('loading', `Loading model on ${backend.toUpperCase()}...`);

  // Temporarily halt the active loop if streaming
  const wasStreaming = isStreaming;
  if (isStreaming) {
    isStreaming = false;
    if (animationFrameId) {
      cancelAnimationFrame(animationFrameId);
      animationFrameId = null;
    }
    stopCamera();
  }

  let activeProvider = backend;

  try {
    const modelPath = '/ascii_cam_model/model.onnx';
    console.log(`🧠 Loading ONNX model from: ${modelPath} using requested backend: ${backend}`);
    
    // We try the selected backend exclusively. If it fails, we fall back to wasm.
    try {
      ortSession = await ort.InferenceSession.create(modelPath, {
        executionProviders: [backend],
        logSeverityLevel: 3
      });
      activeProvider = backend;
    } catch (backendError) {
      if (backend !== 'wasm') {
        console.warn(`⚠️ ${backend.toUpperCase()} initialization failed. Falling back to WASM:`, backendError);
        ortSession = await ort.InferenceSession.create(modelPath, {
          executionProviders: ['wasm'],
          logSeverityLevel: 3
        });
        activeProvider = 'wasm';
      } else {
        throw backendError;
      }
    }
    
    isModelLoaded = true;
    
    // Confirm the active provider
    let confirmedProvider = activeProvider;
    if (ortSession.providers && ortSession.providers.length > 0) {
      confirmedProvider = ortSession.providers[0];
    } else if (ortSession.handler?.provider) {
      confirmedProvider = ortSession.handler.provider;
    }
    
    console.log(`🚀 ONNX Session initialized using: ${confirmedProvider}`);
    updateStatus('ready', `Model ready on ${confirmedProvider.toUpperCase()}.`);
    
    // Sync selector UI in case of fallback
    if (backendSelect) {
      backendSelect.value = confirmedProvider;
    }

    if (wasStreaming) {
      isStreaming = true;
      await startCamera();
    } else if (initialLoadDone) {
      // If we manually switched backends, resume camera automatically
      isStreaming = true;
      await startCamera();
    } else {
      initialLoadDone = true;
      await startCamera();
    }
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

  // Set canvas size only if dimensions changed to prevent browser context resets and reflows
  if (canvas.width !== targetW || canvas.height !== targetH) {
    canvas.width = targetW;
    canvas.height = targetH;
  }

  // A. PREPROCESSING
  // A. OPTIMIZED PREPROCESSING
  const startPreprocess = performance.now();

  if (mirrorCheckbox.checked) {
    ctx.save();
    ctx.translate(targetW, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, targetW, targetH);
    ctx.restore();
  } else {
    ctx.drawImage(video, 0, 0, targetW, targetH);
  }

  const imgData = ctx.getImageData(0, 0, targetW, targetH).data;
  const batchSize = rows * cols;
  const inputBuffer = new Float32Array(batchSize * 128); // 16x8 = 128 pixels per patch

  let bufferOffset = 0;
  const targetW4 = targetW * 4; // Pre-calculate row stride width

  for (let r = 0; r < rows; r++) {
    const rOffset = r * 16;
    for (let c = 0; c < cols; c++) {
      const cOffset = c * 8;

      // Flattened block extractor
      for (let py = 0; py < 16; py++) {
        const pixelRowIndex = (rOffset + py) * targetW4 + cOffset * 4;
        for (let px = 0; px < 8; px++) {
          const idx = pixelRowIndex + px * 4;

          // Fast luma conversion without excess float overhead
          inputBuffer[bufferOffset++] = (0.299 * imgData[idx] + 0.587 * imgData[idx + 1] + 0.114 * imgData[idx + 2]) / 255.0;
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

backendSelect.addEventListener('change', async (e) => {
  await loadModel(e.target.value);
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
