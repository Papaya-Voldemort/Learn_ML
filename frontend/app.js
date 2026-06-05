let ASCII_CHARS = [" ", ".", "-", "=", "+", "*", "x", "%", "#", "@", "/", "\\", "|"];
const PATCH_W = 8;
const PATCH_H = 16;
const TARGET_COLS = 150;

let video, asciiStream;
let onnxSession, inputName;
let targetRows = 56;

// Offscreen canvases for frame processing
let captureCanvas, captureCtx;
let resizeCanvas, resizeCtx;

// Pre-allocated buffers (avoid GC pressure)
let grayBuf, processedBuf, batchBuf;

async function init() {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

    video = document.querySelector('#webcam');
    asciiStream = document.querySelector('#ascii-stream');

    try {
        try {
            const response = await fetch('../config.json');
            if (response.ok) {
                const config = await response.json();
                if (config.ASCII_CHARS && Array.isArray(config.ASCII_CHARS)) {
                    ASCII_CHARS = config.ASCII_CHARS;
                }
            }
        } catch (fetchErr) { }

        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        video.srcObject = stream;
        await video.play().catch(() => {});

        onnxSession = await ort.InferenceSession.create('../ascii_cam_model/model.onnx', {
            executionProviders: ['wasm']
        });
        inputName = onnxSession.inputNames[0];
        console.log("ONNX session loaded. Input:", inputName);

        waitForDimensions();
    } catch (err) {
        console.error("Init error:", err);
        if (asciiStream) asciiStream.textContent = "Error: " + err.message;
    }
}

function waitForDimensions() {
    if (video.videoWidth > 0 && video.videoHeight > 0) {
        setupAndRun();
    } else {
        requestAnimationFrame(waitForDimensions);
    }
}

function setupAndRun() {
    const width = video.videoWidth;
    const height = video.videoHeight;
    const aspect = height / width;
    targetRows = Math.floor(TARGET_COLS * aspect * (PATCH_W / PATCH_H));

    captureCanvas = document.createElement('canvas');
    captureCanvas.width = width;
    captureCanvas.height = height;
    captureCtx = captureCanvas.getContext('2d', { willReadFrequently: true });

    const targetWidth = TARGET_COLS * PATCH_W;
    const targetHeight = targetRows * PATCH_H;
    resizeCanvas = document.createElement('canvas');
    resizeCanvas.width = targetWidth;
    resizeCanvas.height = targetHeight;
    resizeCtx = resizeCanvas.getContext('2d', { willReadFrequently: true });

    // Pre-allocate all typed arrays once
    const totalPixels = targetWidth * targetHeight;
    const totalPatches = targetRows * TARGET_COLS;
    grayBuf = new Uint8Array(totalPixels);
    processedBuf = new Uint8Array(totalPixels);
    batchBuf = new Float32Array(totalPatches * PATCH_H * PATCH_W);

    adjustFontSize();
    window.addEventListener('resize', adjustFontSize);

    console.log(`Pipeline ready: ${TARGET_COLS}x${targetRows} patches, ${targetWidth}x${targetHeight}px`);
    requestAnimationFrame(processFrame);
}

function adjustFontSize() {
    if (!asciiStream) return;
    const charAspect = 0.6;
    const S = Math.min(window.innerWidth / (TARGET_COLS * charAspect), window.innerHeight / targetRows);
    asciiStream.style.fontSize = S + 'px';
    asciiStream.style.lineHeight = S + 'px';
}

// ── Fast Image Processing ──────────────────────────────────────────────

function captureAndGrayscale() {
    const w = captureCanvas.width;
    const h = captureCanvas.height;

    // Draw video frame mirrored
    captureCtx.save();
    captureCtx.translate(w, 0);
    captureCtx.scale(-1, 1);
    captureCtx.drawImage(video, 0, 0, w, h);
    captureCtx.restore();

    // Downscale via GPU-accelerated canvas
    resizeCtx.drawImage(captureCanvas, 0, 0, resizeCanvas.width, resizeCanvas.height);

    // RGBA → grayscale into pre-allocated buffer
    const imgData = resizeCtx.getImageData(0, 0, resizeCanvas.width, resizeCanvas.height);
    const rgba = imgData.data;
    const len = grayBuf.length;

    for (let i = 0; i < len; i++) {
        const off = i << 2;
        grayBuf[i] = (rgba[off] * 77 + rgba[off | 1] * 150 + rgba[off | 2] * 29) >> 8;
    }
}

/**
 * Fast CLAHE — no bilinear interpolation (direct tile LUT application).
 * Uses smaller tiles for aggressive local contrast on faces/features.
 * ~10x faster than interpolated CLAHE since it's just one LUT lookup per pixel.
 */
function fastCLAHE(src, dst, width, height, gridX, gridY, clipLimit) {
    const tileW = (width / gridX) | 0;
    const tileH = (height / gridY) | 0;

    for (let ty = 0; ty < gridY; ty++) {
        const y0 = ty * tileH;
        const y1 = (ty === gridY - 1) ? height : y0 + tileH;

        for (let tx = 0; tx < gridX; tx++) {
            const x0 = tx * tileW;
            const x1 = (tx === gridX - 1) ? width : x0 + tileW;
            const tilePixels = (x1 - x0) * (y1 - y0);

            // Build histogram for this tile
            const hist = new Uint32Array(256);
            for (let y = y0; y < y1; y++) {
                const row = y * width;
                for (let x = x0; x < x1; x++) {
                    hist[src[row + x]]++;
                }
            }

            // Clip and redistribute
            const threshold = (clipLimit * tilePixels / 256) | 0;
            let excess = 0;
            for (let i = 0; i < 256; i++) {
                if (hist[i] > threshold) {
                    excess += hist[i] - threshold;
                    hist[i] = threshold;
                }
            }
            const perBin = (excess / 256) | 0;
            const leftover = excess - (perBin << 8);
            for (let i = 0; i < 256; i++) {
                hist[i] += perBin + (i < leftover ? 1 : 0);
            }

            // CDF → LUT
            const lut = new Uint8Array(256);
            let cdf = 0;
            const scale = 255 / tilePixels;
            for (let i = 0; i < 256; i++) {
                cdf += hist[i];
                lut[i] = (cdf * scale + 0.5) | 0;
            }

            // Apply LUT directly to this tile's pixels (no interpolation)
            for (let y = y0; y < y1; y++) {
                const row = y * width;
                for (let x = x0; x < x1; x++) {
                    dst[row + x] = lut[src[row + x]];
                }
            }
        }
    }
}

/**
 * Fast contrast S-curve via pre-built LUT.
 */
const contrastLUT = new Uint8Array(256);
(function buildContrastLUT() {
    const strength = 1.5;
    for (let i = 0; i < 256; i++) {
        const x = i / 255;
        const s = 1 / (1 + Math.exp(-strength * (x - 0.5) * 12));
        contrastLUT[i] = (s * 255 + 0.5) | 0;
    }
})();

function applyContrast(src, dst, len) {
    for (let i = 0; i < len; i++) {
        dst[i] = contrastLUT[src[i]];
    }
}

// ── Main Frame Loop ────────────────────────────────────────────────────

async function processFrame() {
    try {
        const W = resizeCanvas.width;
        const H = resizeCanvas.height;
        const totalPixels = W * H;

        // 1. Capture, flip, resize, grayscale (GPU-accelerated canvas)
        captureAndGrayscale();

        // 2. CLAHE — 10×10 tiles for tight local contrast around face features
        fastCLAHE(grayBuf, processedBuf, W, H, 10, 10, 4.0);

        // 3. Contrast S-curve (single LUT pass)
        applyContrast(processedBuf, processedBuf, totalPixels);

        // 4. Extract patches and normalize to [0, 1]
        const totalPatches = targetRows * TARGET_COLS;
        let batchIdx = 0;
        for (let r = 0; r < targetRows; r++) {
            const rowOffset = r * PATCH_H;
            for (let c = 0; c < TARGET_COLS; c++) {
                const colOffset = c * PATCH_W;
                for (let py = 0; py < PATCH_H; py++) {
                    const srcRow = (rowOffset + py) * W + colOffset;
                    for (let px = 0; px < PATCH_W; px++) {
                        batchBuf[batchIdx++] = processedBuf[srcRow + px] * 0.00392156863;  // 1/255
                    }
                }
            }
        }

        // 5. Run ONNX inference
        const inputTensor = new ort.Tensor('float32', batchBuf, [totalPatches, PATCH_H, PATCH_W, 1]);
        const outputMap = await onnxSession.run({ [inputName]: inputTensor });
        const scores = outputMap[onnxSession.outputNames[0]].data;
        const numClasses = scores.length / totalPatches;

        // 6. Decode argmax → ASCII
        let asciiOutput = "";
        let scoreOffset = 0;

        for (let r = 0; r < targetRows; r++) {
            let line = "";
            for (let c = 0; c < TARGET_COLS; c++) {
                let maxIdx = 0;
                let maxScore = scores[scoreOffset];
                for (let i = 1; i < numClasses; i++) {
                    const s = scores[scoreOffset + i];
                    if (s > maxScore) { maxScore = s; maxIdx = i; }
                }
                line += (maxIdx < ASCII_CHARS.length) ? ASCII_CHARS[maxIdx] : " ";
                scoreOffset += numClasses;
            }
            asciiOutput += line + "\n";
        }

        asciiStream.textContent = asciiOutput;
    } catch (err) {
        console.error("Frame error:", err);
    }

    requestAnimationFrame(processFrame);
}

// ── Boot ───────────────────────────────────────────────────────────────

if (document.readyState === 'complete' || document.readyState === 'interactive') {
    init();
} else {
    window.addEventListener('load', init);
}