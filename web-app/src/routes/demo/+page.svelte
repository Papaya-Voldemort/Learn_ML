<script lang="ts">
	import { browser } from '$app/environment';
	import { resolve } from '$app/paths';

	// Constants
	const asciiRamp = [
		' ', '.', ',', '-', '~', ':', 'i', 'r', 's', 't',
		'l', 'C', 'O', 'Z', 'w', 'm', '#', '8', '%', '@'
	];

	const characterRamps = {
		standard: asciiRamp,
		minimal: [
			' ', ' ', '.', '.', '-', '-', '~', '~',
			':', ':', 'o', 'o', 'x', 'x', '%', '%',
			'#', '#', '@', '@'
		],
		blocks: [
			' ', '░', '░', '▒', '▒', '▓', '▓', '█',
			'█', '█', '█', '█', '█', '█', '█', '█',
			'█', '█', '█', '█'
		],
		binary: [
			'0', '1', '0', '1', '0', '1', '0', '1',
			'0', '1', '0', '1', '0', '1', '0', '1',
			'0', '1', '0', '1'
		]
	};

	const colorThemes = [
		{ id: 'color', name: 'Average Patch Color' },
		{ id: 'ectoplasm', name: 'Cyber Teal (Ectoplasm)' },
		{ id: 'green', name: 'Matrix Green' },
		{ id: 'amber', name: 'Retro Amber' },
		{ id: 'mono', name: 'Monochromatic' }
	];

	// Svelte 5 Runes for App State
	let cols = $state(90);
	let brightness = $state(0.05);
	let contrast = $state(1.1);
	const activeModelName = 'model.onnx';
	let selectedProvider = $state('wasm');
	let colorMode = $state('color');
	let renderStyle = $state('canvas');
	let fontSize = $state(10);
	let activeRampName = $state('standard');
	let targetFps = $state(30);
	let showPreview = $state(false);
	let mirrorCamera = $state(true);
	let invertChars = $state(false);
	let glowEffect = $state(true);
	let scanlinesEffect = $state(true);

	// Hardware capabilities (detected on mount)
	let isWebGPUSupported = $state(false);
	let isWebGLSupported = $state(false);

	// Running state
	let isCameraActive = $state(false);
	let isPaused = $state(false);
	let isModelLoading = $state(false);
	let modelLoadError = $state('');
	let cameraError = $state('');

	// Benchmark Metrics
	let preTime = $state(0);
	let infTime = $state(0);
	let postTime = $state(0);
	let totalTime = $state(0);
	let measuredFps = $state(0);

	// Web API elements & stream references
	let videoElement: HTMLVideoElement | null = $state(null);
	let renderCanvasElement: HTMLCanvasElement | null = $state(null);
	let mediaStream: MediaStream | null = null;
	let cameraDevices = $state<MediaDeviceInfo[]>([]);
	let selectedDeviceId = $state('');

	// ONNX Runtime session & instance references
	let ort: any = null;
	let session: any = $state.raw(null);
	let activeProvider = $state('wasm');

	// Loop management
	let animationFrameId: number | null = null;
	let lastFrameTime = 0;
	let frameCount = 0;
	let lastFpsUpdate = 0;

	// Pre-allocated buffers for performance
	let offscreenCanvas: HTMLCanvasElement | null = null;
	let offscreenCtx: CanvasRenderingContext2D | null = null;

	// Text output state (for pre tag rendering & text copy/export)
	let rawTextOutput = $state('');
	let lastAsciiText = $state.raw('');

	// Detect device support
	function detectHardwareSupport() {
		if (!browser) return;
		isWebGPUSupported = !!navigator.gpu;
		try {
			const canvas = document.createElement('canvas');
			isWebGLSupported = !!(
				window.WebGLRenderingContext &&
				(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'))
			);
		} catch (e) {
			isWebGLSupported = false;
		}
	}

	// List camera devices
	async function getCameraDevices() {
		if (!browser) return;
		try {
			// Request permission first to get labels
			await navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
				stream.getTracks().forEach(t => t.stop());
			}).catch(() => {});

			const devices = await navigator.mediaDevices.enumerateDevices();
			cameraDevices = devices.filter(d => d.kind === 'videoinput');
			if (cameraDevices.length > 0 && !selectedDeviceId) {
				selectedDeviceId = cameraDevices[0].deviceId;
			}
		} catch (err) {
			console.error('Failed to list camera devices:', err);
		}
	}

	// Load ONNX Model
	async function loadModel(modelName: string, provider: string) {
		if (!browser || !ort) return;
		isModelLoading = true;
		modelLoadError = '';

		try {
			const modelUrl = `/model/${modelName}`;
			
			// Quantized model operators (INT8) are not supported by GPU (WebGPU/WebGL) backends in ONNX Runtime Web.
			// Force WASM provider for quantized models to prevent runtime execution crashes.
			let targetProvider = provider;
			if (modelName.includes('quant') && provider !== 'wasm') {
				console.warn(`Quantized models are not supported on GPU providers in ORT Web. Forcing WASM CPU backend.`);
				targetProvider = 'wasm';
			}

			// Try creating session with ONLY the target execution provider first
			try {
				const options = {
					executionProviders: [targetProvider]
				};
				session = await ort.InferenceSession.create(modelUrl, options);
				activeProvider = targetProvider;
				console.log(`Loaded model ${modelName} on execution provider: ${targetProvider}`);
			} catch (err) {
				console.warn(`Failed to initialize session with execution provider "${targetProvider}", falling back to wasm:`, err);
				if (targetProvider !== 'wasm') {
					const options = {
						executionProviders: ['wasm']
					};
					session = await ort.InferenceSession.create(modelUrl, options);
					activeProvider = 'wasm';
					console.log(`Loaded model ${modelName} on fallback execution provider: wasm`);
				} else {
					throw err;
				}
			}
		} catch (err: any) {
			console.error('Failed to load ONNX model:', err);
			modelLoadError = err.message || 'Failed to initialize session on selected backend.';
		} finally {
			isModelLoading = false;
		}
	}

	// Start webcam stream
	async function startCamera() {
		if (!browser) return;
		stopCamera();
		cameraError = '';
		if (!selectedDeviceId) return;

		try {
			const constraints = {
				video: {
					deviceId: { exact: selectedDeviceId },
					width: { ideal: 640 },
					height: { ideal: 480 }
				}
			};
			mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
			isCameraActive = true;
			isPaused = false;

			if (videoElement) {
				videoElement.srcObject = mediaStream;
				videoElement.onloadedmetadata = () => {
					videoElement?.play();
					startLoop();
				};
			}
		} catch (err: any) {
			console.error('Failed to open video source:', err);
			cameraError = err.message || 'Could not access webcam. Please verify permissions.';
			isCameraActive = false;
		}
	}

	// Stop webcam stream
	function stopCamera() {
		stopLoop();
		if (mediaStream) {
			mediaStream.getTracks().forEach(track => track.stop());
			mediaStream = null;
		}
		isCameraActive = false;
	}

	// Loop management
	function startLoop() {
		if (animationFrameId) return;
		lastFrameTime = performance.now();
		lastFpsUpdate = performance.now();
		frameCount = 0;
		animationFrameId = requestAnimationFrame(frameLoop);
	}

	function stopLoop() {
		if (animationFrameId !== null) {
			cancelAnimationFrame(animationFrameId);
			animationFrameId = null;
		}
	}

	async function frameLoop(timestamp: number) {
		if (!isCameraActive || isPaused) {
			animationFrameId = null;
			return;
		}

		const elapsed = timestamp - lastFrameTime;
		const targetInterval = targetFps === 0 ? 0 : 1000 / targetFps;

		if (elapsed >= targetInterval) {
			lastFrameTime = timestamp - (targetFps === 0 ? 0 : elapsed % targetInterval);
			try {
				await processFrame();
			} catch (err) {
				console.error('Error processing camera frame:', err);
			}
		}

		animationFrameId = requestAnimationFrame(frameLoop);
	}

	// Process Frame: Downsampling, Preprocessing, Neural Inference, Canvas/Text rendering
	async function processFrame() {
		if (!videoElement || videoElement.paused || videoElement.ended) return;
		if (!ort || !session) return;

		const startFrameTime = performance.now();

		const vWidth = videoElement.videoWidth;
		const vHeight = videoElement.videoHeight;
		if (!vWidth || !vHeight) return;

		const aspect = vWidth / vHeight;
		const rows = Math.max(1, Math.round((cols * 0.5) / aspect));
		const targetW = cols * 8;
		const targetH = rows * 16;

		// Initialize/resize offscreen canvas
		if (!offscreenCanvas) {
			offscreenCanvas = document.createElement('canvas');
		}
		if (offscreenCanvas.width !== targetW || offscreenCanvas.height !== targetH) {
			offscreenCanvas.width = targetW;
			offscreenCanvas.height = targetH;
			offscreenCtx = offscreenCanvas.getContext('2d', { willReadFrequently: true });
		}

		if (!offscreenCtx) return;

		// 1. Draw video frame to offscreen canvas
		offscreenCtx.save();
		if (mirrorCamera) {
			offscreenCtx.translate(targetW, 0);
			offscreenCtx.scale(-1, 1);
		}
		offscreenCtx.drawImage(videoElement, 0, 0, targetW, targetH);
		offscreenCtx.restore();

		const imgData = offscreenCtx.getImageData(0, 0, targetW, targetH).data;

		// 2. Extract 16x8 patches and compute colors
		const floatData = new Float32Array(rows * cols * 128);
		const averageColors = [];
		let offset = 0;

		const preStart = performance.now();

		for (let r = 0; r < rows; r++) {
			for (let c = 0; c < cols; c++) {
				const startX = c * 8;
				const startY = r * 16;

				let sumR = 0, sumG = 0, sumB = 0;

				for (let py = 0; py < 16; py++) {
					const y = startY + py;
					const rowOffset = y * targetW;
					for (let px = 0; px < 8; px++) {
						const x = startX + px;
						const pixelIdx = (rowOffset + x) * 4;

						const rVal = imgData[pixelIdx];
						const gVal = imgData[pixelIdx + 1];
						const bVal = imgData[pixelIdx + 2];

						sumR += rVal;
						sumG += gVal;
						sumB += bVal;

						// Grayscale conversion
						let gray = (0.299 * rVal + 0.587 * gVal + 0.114 * bVal) / 255.0;

						// Apply Brightness/Contrast preprocessing
						gray = (gray - 0.5) * contrast + 0.5 + brightness;
						if (gray < 0) gray = 0;
						if (gray > 1) gray = 1;

						floatData[offset++] = gray;
					}
				}

				// Cache patch average color for colorized renders
				averageColors.push({
					r: Math.round(sumR / 128),
					g: Math.round(sumG / 128),
					b: Math.round(sumB / 128)
				});
			}
		}

		preTime = performance.now() - preStart;

		// 3. Tensor construction & Model Inference
		const N = rows * cols;
		const inputTensor = new ort.Tensor('float32', floatData, [N, 16, 8, 1]);
		const feeds = { [session.inputNames[0]]: inputTensor };

		const infStart = performance.now();
		const results = await session.run(feeds);
		const outputTensor = results[session.outputNames[0]];
		const outputData = outputTensor.data;
		infTime = performance.now() - infStart;

		// 4. Argmax
		const postStart = performance.now();
		const predictedClasses = new Int32Array(N);
		for (let i = 0; i < N; i++) {
			let maxVal = -Infinity;
			let maxIdx = 0;
			const startIdx = i * 20;
			for (let j = 0; j < 20; j++) {
				const val = outputData[startIdx + j];
				if (val > maxVal) {
					maxVal = val;
					maxIdx = j;
				}
			}
			predictedClasses[i] = maxIdx;
		}

		// Map to characters
		const currentRamp = characterRamps[activeRampName as keyof typeof characterRamps] || asciiRamp;
		const mappingChars = invertChars ? [...currentRamp].reverse() : currentRamp;

		const asciiCharsMapped = [];
		for (let i = 0; i < N; i++) {
			asciiCharsMapped.push(mappingChars[predictedClasses[i]]);
		}

		// Rebuild full string for copy/text export
		let textResult = '';
		for (let r = 0; r < rows; r++) {
			const rowStart = r * cols;
			for (let c = 0; c < cols; c++) {
				textResult += asciiCharsMapped[rowStart + c];
			}
			textResult += '\n';
		}
		lastAsciiText = textResult;

		// 5. Render output
		if (renderStyle === 'canvas') {
			renderCanvas(asciiCharsMapped, averageColors, cols, rows);
		} else {
			rawTextOutput = textResult;
		}

		postTime = performance.now() - postStart;
		totalTime = performance.now() - startFrameTime;

		// Frame Rate calculation
		frameCount++;
		const now = performance.now();
		if (now - lastFpsUpdate >= 1000) {
			measuredFps = Math.round((frameCount * 1000) / (now - lastFpsUpdate));
			frameCount = 0;
			lastFpsUpdate = now;
		}
	}

	// Cache for Canvas character dimensions to avoid calling measureText every frame
	let cachedFontSize = 0;
	let cachedCharWidth = 0;
	let cachedCharHeight = 0;

	function getCharMetrics(ctx: CanvasRenderingContext2D, size: number) {
		if (cachedFontSize === size && cachedCharWidth > 0) {
			return { width: cachedCharWidth, height: cachedCharHeight };
		}
		ctx.save();
		ctx.font = `bold ${size}px "Courier New", Courier, monospace`;
		const metrics = ctx.measureText('@');
		cachedCharWidth = metrics.width;
		cachedCharHeight = size * 1.25;
		cachedFontSize = size;
		ctx.restore();
		return { width: cachedCharWidth, height: cachedCharHeight };
	}

	// Canvas Render implementation
	function renderCanvas(
		asciiChars: string[],
		colors: { r: number; g: number; b: number }[],
		gridCols: number,
		gridRows: number
	) {
		if (!renderCanvasElement) return;
		const ctx = renderCanvasElement.getContext('2d');
		if (!ctx) return;

		// Set font size and determine layout metric from cache
		const { width: charWidth, height: charHeight } = getCharMetrics(ctx, fontSize);

		const canvasW = gridCols * charWidth;
		const canvasH = gridRows * charHeight;

		// Update canvas dimensions if changed
		if (renderCanvasElement.width !== canvasW || renderCanvasElement.height !== canvasH) {
			renderCanvasElement.width = canvasW;
			renderCanvasElement.height = canvasH;
		}

		// Reset settings context after resize
		ctx.fillStyle = '#0d0e0f';
		ctx.fillRect(0, 0, canvasW, canvasH);
		ctx.font = `bold ${fontSize}px "Courier New", Courier, monospace`;
		ctx.textBaseline = 'top';

		// GPU-accelerated CSS drop-shadow filters are used for glows, which allows us to keep canvas shadowBlur disabled (0) to hit 60 FPS
		ctx.shadowBlur = 0;

		// Set fillStyle once outside the loop if the color mode is static, preventing 5000 state mutations
		if (colorMode !== 'color') {
			if (colorMode === 'ectoplasm') {
				ctx.fillStyle = '#a3d8d4'; // --primary
			} else if (colorMode === 'green') {
				ctx.fillStyle = '#39ff14'; // matrix green
			} else if (colorMode === 'amber') {
				ctx.fillStyle = '#ffb000'; // CRT amber
			} else {
				ctx.fillStyle = '#e1e3e2'; // monochrome white
			}
		}

		for (let r = 0; r < gridRows; r++) {
			for (let c = 0; c < gridCols; c++) {
				const idx = r * gridCols + c;
				const char = asciiChars[idx];

				if (colorMode === 'color') {
					const col = colors[idx];
					ctx.fillStyle = `rgb(${col.r}, ${col.g}, ${col.b})`;
				}

				const x = c * charWidth;
				const y = r * charHeight;
				ctx.fillText(char, x, y);
			}
		}
	}

	// Utility: Pause/Play
	function togglePlayPause() {
		if (isPaused) {
			isPaused = false;
			startLoop();
		} else {
			isPaused = true;
			stopLoop();
		}
	}

	// Utility: Capture Canvas PNG
	function captureScreenshot() {
		if (renderStyle !== 'canvas' || !renderCanvasElement) return;
		const url = renderCanvasElement.toDataURL('image/png');
		const a = document.createElement('a');
		a.download = `ghostchar-ascii-cam-${Date.now()}.png`;
		a.href = url;
		a.click();
	}

	// Utility: Copy raw text to clipboard
	function copyToClipboard() {
		if (!lastAsciiText) return;
		navigator.clipboard.writeText(lastAsciiText).then(() => {
			alert('ASCII art copied to clipboard!');
		}).catch(err => {
			console.error('Failed to copy text:', err);
		});
	}

	// Utility: Save ASCII text file
	function downloadTextFile() {
		if (!lastAsciiText) return;
		const blob = new Blob([lastAsciiText], { type: 'text/plain;charset=utf-8' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.download = `ghostchar-ascii-cam-${Date.now()}.txt`;
		a.href = url;
		a.click();
		URL.revokeObjectURL(url);
	}

	// Lifecycle hooks
	$effect(() => {
		detectHardwareSupport();
		getCameraDevices();

		// Dynamically import ONNX runtime web only in client-side execution
		const initOrt = async () => {
			try {
				ort = await import('onnxruntime-web');
				// Point ORT to local WASM binaries cached in /model/
				ort.env.wasm.wasmPaths = '/model/';
				await loadModel(activeModelName, selectedProvider);
				await startCamera();
			} catch (err) {
				console.error('Failed to initialize ONNX Runtime Web:', err);
			}
		};

		if (browser) {
			initOrt();
		}

		return () => {
			stopCamera();
		};
	});

	// Reactivity: Reload model on source settings changes
	$effect(() => {
		const model = activeModelName;
		const prov = selectedProvider;
		if (browser && ort) {
			loadModel(model, prov);
		}
	});

	// Reactivity: Restart webcam stream if active device changes
	$effect(() => {
		if (browser && selectedDeviceId) {
			startCamera();
		}
	});
</script>

<svelte:head>
	<title>GhostChar | Live ASCII Translation</title>
	<meta name="description" content="Neural-powered live webcam translating into high-fidelity ASCII artwork in real time." />
</svelte:head>

<div class="demo-wrapper">
	<div class="header-row">
		<div class="header-title">
			<h1>Live ASCII Translator</h1>
			<p class="subtitle">Neural Net powered real-time video translation pipeline</p>
		</div>
		<div class="status-indicators">
			<span class="badge" class:badge-active={isCameraActive && !isPaused}>
				{isCameraActive ? (isPaused ? 'PAUSED' : 'LIVE FEED') : 'CAMERA INACTIVE'}
			</span>
			<span class="badge" class:badge-active={!isModelLoading && session}>
				{isModelLoading ? 'LOADING MODEL...' : 'MODEL READY'}
			</span>
		</div>
	</div>

	<div class="demo-container">
		<!-- Sidebar for Settings -->
		<aside class="settings-panel">
			<div class="section-title">Video Source</div>
			<div class="control-group">
				<label for="camera-select">Webcam Device</label>
				<select id="camera-select" bind:value={selectedDeviceId} disabled={cameraDevices.length === 0}>
					{#if cameraDevices.length === 0}
						<option value="">No Camera Found</option>
					{:else}
						{#each cameraDevices as device}
							<option value={device.deviceId}>{device.label || `Camera (${device.deviceId.slice(0, 5)}...)`}</option>
						{/each}
					{/if}
				</select>
				{#if cameraError}
					<p class="error-text">{cameraError}</p>
				{/if}
			</div>

			<div class="section-title">Model Settings</div>

			<div class="control-group">
				<label for="provider-select">Hardware Acceleration</label>
				<select id="provider-select" bind:value={selectedProvider}>
					<option value="wasm">CPU (WebAssembly)</option>
					{#if isWebGPUSupported}
						<option value="webgpu">GPU (WebGPU - Recommended)</option>
					{/if}
					{#if isWebGLSupported}
						<option value="webgl">GPU (WebGL)</option>
					{/if}
				</select>
				<div class="hardware-info">
					<span>WebGPU: <span class={isWebGPUSupported ? 'supported' : 'unsupported'}>{isWebGPUSupported ? 'Yes' : 'No'}</span></span>
					<span style="margin-left: 1rem;">WebGL: <span class={isWebGLSupported ? 'supported' : 'unsupported'}>{isWebGLSupported ? 'Yes' : 'No'}</span></span>
				</div>
			</div>

			<div class="control-group">
				<div class="slider-header">
					<label for="columns-range">Resolution (Columns)</label>
					<span class="slider-value">{cols} chars</span>
				</div>
				<!-- use onchange to update resolution to avoid shader compilation lag on sliding -->
				<input
					type="range"
					id="columns-range"
					min="40"
					max="160"
					step="5"
					value={cols}
					onchange={(e) => cols = parseInt(e.currentTarget.value)}
				/>
				<p class="slider-help">Resolution updates when you release the slider.</p>
			</div>

			<div class="section-title">Preprocessing</div>
			<div class="control-group">
				<div class="slider-header">
					<label for="brightness-range">Brightness</label>
					<span class="slider-value">{(brightness >= 0 ? '+' : '') + brightness.toFixed(2)}</span>
				</div>
				<input
					type="range"
					id="brightness-range"
					min="-0.4"
					max="0.4"
					step="0.02"
					bind:value={brightness}
				/>
			</div>

			<div class="control-group">
				<div class="slider-header">
					<label for="contrast-range">Contrast</label>
					<span class="slider-value">{contrast.toFixed(2)}x</span>
				</div>
				<input
					type="range"
					id="contrast-range"
					min="0.5"
					max="2.0"
					step="0.05"
					bind:value={contrast}
				/>
			</div>

			<div class="section-title">Rendering & Styling</div>
			<div class="control-group">
				<label for="render-style">Render Interface</label>
				<select id="render-style" bind:value={renderStyle}>
					<option value="canvas">Interactive Canvas (Supports Color & Glows)</option>
					<option value="text">Raw Text (Standard Monospace Pre)</option>
				</select>
			</div>

			<div class="control-group">
				<label for="color-mode">Color Palette</label>
				<select id="color-mode" bind:value={colorMode} disabled={renderStyle === 'text'}>
					{#each colorThemes as theme}
						<option value={theme.id}>{theme.name}</option>
					{/each}
				</select>
				{#if renderStyle === 'text'}
					<p class="warning-text">Full colors require canvas render interface.</p>
				{/if}
			</div>

			<div class="control-group">
				<label for="ramp-select">ASCII Character Set</label>
				<select id="ramp-select" bind:value={activeRampName}>
					<option value="standard">Standard Ramp (20 classes)</option>
					<option value="minimal">Minimalist Ramp (8 classes scaled)</option>
					<option value="blocks">Block Shades (Gradient blocks)</option>
					<option value="binary">Binary Matrix (0 / 1)</option>
				</select>
			</div>

			{#if renderStyle === 'canvas'}
				<div class="control-group">
					<div class="slider-header">
						<label for="font-size-range">ASCII Font Size</label>
						<span class="slider-value">{fontSize}px</span>
					</div>
					<input
						type="range"
						id="font-size-range"
						min="6"
						max="18"
						step="1"
						bind:value={fontSize}
					/>
				</div>
			{/if}

			<div class="toggles-grid">
				<label class="checkbox-label">
					<input type="checkbox" bind:checked={mirrorCamera} />
					<span>Mirror Video</span>
				</label>
				<label class="checkbox-label">
					<input type="checkbox" bind:checked={invertChars} />
					<span>Invert Density</span>
				</label>
				{#if renderStyle === 'canvas'}
					<label class="checkbox-label">
						<input type="checkbox" bind:checked={glowEffect} />
						<span>CRT Glow</span>
					</label>
					<label class="checkbox-label">
						<input type="checkbox" bind:checked={scanlinesEffect} />
						<span>Scanlines</span>
					</label>
				{/if}
			</div>

			<div class="control-group" style="margin-top: 1rem;">
				<label for="fps-select">Target Frame Rate</label>
				<select id="fps-select" bind:value={targetFps}>
					<option value={10}>10 FPS (Low resources)</option>
					<option value={15}>15 FPS (Cinematic retro)</option>
					<option value={24}>24 FPS (Film standard)</option>
					<option value={30}>30 FPS (Smooth - Default)</option>
					<option value={0}>Uncapped (Max hardware potential)</option>
				</select>
			</div>
		</aside>

		<!-- Main Workspace (ASCII Render Screen) -->
		<main class="ascii-workspace">
			{#if isModelLoading}
				<div class="workspace-overlay">
					<div class="spinner"></div>
					<p>Initializing Neural ASCII Model...</p>
					<p class="loading-sub">Configuring execution provider on {selectedProvider}...</p>
				</div>
			{/if}

			{#if modelLoadError}
				<div class="workspace-overlay error-overlay">
					<div class="error-icon">⚠️</div>
					<p>Model Loading Failed</p>
					<p class="error-details">{modelLoadError}</p>
					<button class="retry-btn" onclick={() => loadModel(activeModelName, selectedProvider)}>Retry Loading</button>
				</div>
			{/if}

			<div class="viewport-card">
				<div class="viewport-header">
					<span class="crt-tag">CRT-MONITOR-01 // ASCII RENDER</span>
					<div class="window-controls">
						<div class="dot red"></div>
						<div class="dot yellow"></div>
						<div class="dot green"></div>
					</div>
				</div>

				<div class="viewport-display">
					<div class="canvas-container" class:with-scanlines={scanlinesEffect && renderStyle === 'canvas'}>
						{#if renderStyle === 'canvas'}
							<canvas
								bind:this={renderCanvasElement}
								class="rendered-canvas"
								style="--glow-color: {colorMode === 'ectoplasm'
									? 'rgba(163, 216, 212, 0.6)'
									: colorMode === 'green'
										? 'rgba(57, 255, 20, 0.6)'
										: colorMode === 'amber'
											? 'rgba(255, 176, 0, 0.6)'
											: colorMode === 'mono'
												? 'rgba(225, 227, 226, 0.6)'
												: 'rgba(163, 216, 212, 0.6)'};"
							></canvas>
						{:else}
							<pre
								class="ascii-pre-output"
								class:glow-pre={glowEffect}
								style="font-size: {fontSize}px; color: {colorMode === 'ectoplasm'
									? 'var(--primary)'
									: colorMode === 'green'
										? '#39ff14'
										: colorMode === 'amber'
											? '#ffb000'
											: colorMode === 'mono'
												? 'var(--on-surface)'
												: '#a3d8d4'};"
							>{rawTextOutput || 'Awaiting webcam stream...'}</pre>
						{/if}

						<!-- Webcam Preview Box (Overlay - default toggle off) -->
						<div class="camera-preview-card" class:visible={showPreview} class:mirrored={mirrorCamera}>
							<video bind:this={videoElement} autoplay playsinline muted></video>
							<div class="preview-badge">CAMERA FEED</div>
						</div>
					</div>
				</div>

				<!-- Quick controls footer -->
				<div class="viewport-controls">
					<div class="action-buttons">
						<button class="action-btn" onclick={togglePlayPause} disabled={!isCameraActive}>
							{isPaused ? '▶ Resume' : '⏸ Pause'}
						</button>
						<button class="action-btn" onclick={() => showPreview = !showPreview}>
							{showPreview ? 'Hide Camera Feed' : 'Show Camera Feed'}
						</button>
						<button class="action-btn border-accent" onclick={captureScreenshot} disabled={renderStyle !== 'canvas'}>
							📷 Snapshot
						</button>
						<button class="action-btn" onclick={copyToClipboard} disabled={!lastAsciiText}>
							📋 Copy Raw Text
						</button>
						<button class="action-btn" onclick={downloadTextFile} disabled={!lastAsciiText}>
							💾 Download .TXT
						</button>
					</div>

					<!-- Real-time performance statistics -->
					<div class="performance-metrics">
						<div class="metric-item">
							<span class="metric-label">FPS</span>
							<span class="metric-value">{measuredFps}</span>
						</div>
						<div class="metric-item">
							<span class="metric-label">BACKEND</span>
							<span class="metric-value" style="text-transform: uppercase; color: var(--primary);">{activeProvider}</span>
						</div>
						<div class="metric-item">
							<span class="metric-label">PRE</span>
							<span class="metric-value">{preTime.toFixed(1)}ms</span>
						</div>
						<div class="metric-item">
							<span class="metric-label">INF</span>
							<span class="metric-value">{infTime.toFixed(1)}ms</span>
						</div>
						<div class="metric-item">
							<span class="metric-label">POST</span>
							<span class="metric-value">{postTime.toFixed(1)}ms</span>
						</div>
						<div class="metric-item highlight">
							<span class="metric-label">LATENCY</span>
							<span class="metric-value">{totalTime.toFixed(1)}ms</span>
						</div>
					</div>
				</div>
			</div>
		</main>
	</div>
</div>

<style>
	/* Layout Structure */
	.demo-wrapper {
		display: flex;
		flex-direction: column;
		height: calc(100vh - 64px - 4rem);
		max-width: 1300px;
		margin: 0 auto;
		font-family: var(--font-family);
		color: var(--on-surface);
	}

	.header-row {
		display: flex;
		justify-content: space-between;
		align-items: flex-end;
		margin-bottom: 1.5rem;
		border-bottom: 1px solid var(--outline-variant);
		padding-bottom: 1rem;
	}

	.header-title h1 {
		font-size: 1.8rem;
		font-weight: 700;
		letter-spacing: -0.01em;
	}

	.header-title .subtitle {
		font-size: 0.9rem;
		color: var(--on-surface-variant);
	}

	.status-indicators {
		display: flex;
		gap: 0.75rem;
	}

	.badge {
		font-family: 'Courier New', Courier, monospace;
		font-size: 0.75rem;
		font-weight: 700;
		padding: 0.25rem 0.5rem;
		border-radius: 2px;
		background: var(--surface-container-high);
		border: 1px solid var(--outline-variant);
		color: var(--on-surface-variant);
		letter-spacing: 0.05em;
		transition: all 0.2s;
	}

	.badge-active {
		background: var(--primary-container);
		border-color: var(--primary);
		color: var(--on-primary-container);
	}

	.demo-container {
		display: grid;
		grid-template-columns: 310px 1fr;
		gap: 2rem;
		flex: 1;
		min-height: 0; /* Ensures overflow scrolls correctly */
	}

	/* Settings Panel Styling */
	.settings-panel {
		background: var(--surface-container-lowest);
		border: 1px solid var(--outline-variant);
		border-radius: var(--roundness);
		padding: 1.25rem;
		overflow-y: auto;
		font-size: 0.9rem;
		display: flex;
		flex-direction: column;
		gap: 1.25rem;
	}

	.section-title {
		font-family: 'Courier New', Courier, monospace;
		font-weight: 700;
		text-transform: uppercase;
		font-size: 0.8rem;
		color: var(--primary);
		border-bottom: 1px dashed var(--outline-variant);
		padding-bottom: 0.25rem;
		letter-spacing: 0.08em;
	}

	.control-group {
		display: flex;
		flex-direction: column;
		gap: 0.35rem;
	}

	.control-group label {
		font-size: 0.8rem;
		color: var(--on-surface-variant);
		font-weight: 500;
	}

	.control-group select {
		background: var(--surface-container-low);
		border: 1px solid var(--outline-variant);
		color: var(--on-surface);
		padding: 0.45rem 0.6rem;
		border-radius: var(--roundness);
		font-size: 0.85rem;
		outline: none;
		transition: border-color 0.2s;
	}

	.control-group select:focus {
		border-color: var(--primary);
	}

	.hardware-info {
		display: flex;
		font-size: 0.75rem;
		color: var(--on-surface-variant);
		margin-top: 0.25rem;
	}

	.supported {
		color: #55ee55;
		font-weight: bold;
	}

	.unsupported {
		color: #ee5555;
		opacity: 0.8;
	}

	.slider-header {
		display: flex;
		justify-content: space-between;
		font-size: 0.8rem;
	}

	.slider-value {
		color: var(--primary);
		font-family: 'Courier New', Courier, monospace;
		font-weight: 700;
	}

	.control-group input[type='range'] {
		-webkit-appearance: none;
		appearance: none;
		background: var(--surface-container-highest);
		height: 4px;
		border-radius: 2px;
		outline: none;
		margin: 0.5rem 0;
	}

	.control-group input[type='range']::-webkit-slider-thumb {
		-webkit-appearance: none;
		appearance: none;
		width: 14px;
		height: 14px;
		border-radius: 50%;
		background: var(--primary);
		cursor: pointer;
		box-shadow: 0 0 5px var(--primary);
		transition: transform 0.1s;
	}

	.control-group input[type='range']::-webkit-slider-thumb:hover {
		transform: scale(1.25);
	}

	.slider-help {
		font-size: 0.7rem;
		color: var(--on-surface-variant);
		opacity: 0.7;
		font-style: italic;
	}

	.toggles-grid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 0.75rem;
		margin-top: 0.25rem;
	}

	.checkbox-label {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		font-size: 0.8rem;
		cursor: pointer;
		color: var(--on-surface-variant);
	}

	.checkbox-label input[type='checkbox'] {
		accent-color: var(--primary);
		width: 14px;
		height: 14px;
	}

	.error-text {
		color: #ff5555;
		font-size: 0.75rem;
		margin-top: 0.25rem;
	}

	.warning-text {
		color: var(--outline);
		font-size: 0.75rem;
		font-style: italic;
	}

	/* Main Workspace styling */
	.ascii-workspace {
		position: relative;
		display: flex;
		flex-direction: column;
		min-height: 0;
	}

	.viewport-card {
		background: var(--surface-container-lowest);
		border: none;
		border-radius: var(--roundness);
		display: flex;
		flex-direction: column;
		flex: 1;
		min-height: 0;
		overflow: hidden;
	}

	.viewport-header {
		background: var(--surface-container-low);
		border-bottom: 1px solid var(--outline-variant);
		padding: 0.5rem 1rem;
		display: flex;
		justify-content: space-between;
		align-items: center;
	}

	.crt-tag {
		font-family: 'Courier New', Courier, monospace;
		font-weight: 700;
		font-size: 0.75rem;
		color: var(--outline);
		letter-spacing: 0.05em;
	}

	.window-controls {
		display: flex;
		gap: 0.35rem;
	}

	.dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		opacity: 0.6;
	}

	.dot.red { background-color: #ff5555; }
	.dot.yellow { background-color: #ffbb00; }
	.dot.green { background-color: #00cc55; }

	.viewport-display {
		flex: 1;
		position: relative;
		min-height: 0;
		background: #060708;
		display: flex;
		justify-content: center;
		align-items: center;
		overflow: hidden;
	}

	.canvas-container {
		position: relative;
		width: 100%;
		height: 100%;
		display: flex;
		justify-content: center;
		align-items: center;
		overflow: hidden;
	}

	.rendered-canvas {
		max-width: 100%;
		max-height: 100%;
		object-fit: contain;
		image-rendering: pixelated;
		transition: filter 0.2s ease;
	}

	:global(.rendered-canvas.canvas-glow) {
		filter: drop-shadow(0 0 3px var(--glow-color, rgba(163, 216, 212, 0.6)));
	}

	.ascii-pre-output {
		font-family: 'Courier New', Courier, monospace;
		font-weight: bold;
		line-height: 1.25;
		margin: 0;
		white-space: pre;
		overflow: auto;
		width: 100%;
		height: 100%;
		padding: 1.5rem;
		box-sizing: border-box;
		text-align: left;
		background: #0d0e0f;
	}

	.glow-pre {
		text-shadow: 0 0 3px currentColor;
	}

	/* CRT scanline simulation overlay */
	.with-scanlines::after {
		content: ' ';
		display: block;
		position: absolute;
		top: 0;
		left: 0;
		bottom: 0;
		right: 0;
		background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.2) 50%);
		background-size: 100% 4px;
		pointer-events: none;
		z-index: 5;
		opacity: 0.3;
	}

	/* Camera preview PIP overlay (default invisible / off) */
	.camera-preview-card {
		position: absolute;
		bottom: 1rem;
		right: 1rem;
		width: 150px;
		aspect-ratio: 4/3;
		border-radius: var(--roundness);
		border: 1px solid var(--primary);
		background: var(--surface-container-lowest);
		box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
		overflow: hidden;
		z-index: 10;
		pointer-events: none;
		opacity: 0;
		transition: opacity 0.3s cubic-bezier(0.16, 1, 0.3, 1), transform 0.3s;
		transform: scale(0.95);
	}

	.camera-preview-card.visible {
		opacity: 1;
		transform: scale(1);
		pointer-events: auto;
	}

	.camera-preview-card video {
		width: 100%;
		height: 100%;
		object-fit: cover;
	}

	.camera-preview-card.mirrored video {
		transform: scaleX(-1);
	}

	.preview-badge {
		position: absolute;
		top: 0.25rem;
		left: 0.25rem;
		background: rgba(0, 0, 0, 0.7);
		font-family: 'Courier New', Courier, monospace;
		font-size: 0.6rem;
		font-weight: bold;
		color: var(--primary);
		padding: 0.1rem 0.3rem;
		border-radius: 2px;
		letter-spacing: 0.05em;
	}

	/* Quick controls footer */
	.viewport-controls {
		background: var(--surface-container-low);
		border-top: 1px solid var(--outline-variant);
		padding: 0.75rem 1rem;
		display: flex;
		justify-content: space-between;
		align-items: center;
		flex-wrap: wrap;
		gap: 1rem;
	}

	.action-buttons {
		display: flex;
		gap: 0.5rem;
		flex-wrap: wrap;
	}

	.action-btn {
		background: var(--surface-container-high);
		border: 1px solid var(--outline-variant);
		color: var(--on-surface);
		padding: 0.4rem 0.8rem;
		font-size: 0.8rem;
		border-radius: var(--roundness);
		font-weight: 500;
		transition: all 0.2s;
	}

	.action-btn:hover:not(:disabled) {
		background: var(--surface-container-highest);
		border-color: var(--primary);
		color: var(--primary);
	}

	.action-btn:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}

	.action-btn.border-accent {
		border-color: var(--primary-container);
		color: var(--primary);
	}

	.action-btn.border-accent:hover:not(:disabled) {
		background: var(--primary-container);
		color: var(--on-primary-container);
	}

	/* Metrics dashboard */
	.performance-metrics {
		display: flex;
		gap: 1.25rem;
	}

	.metric-item {
		display: flex;
		flex-direction: column;
		align-items: flex-end;
	}

	.metric-label {
		font-size: 0.65rem;
		color: var(--on-surface-variant);
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}

	.metric-value {
		font-family: 'Courier New', Courier, monospace;
		font-weight: 700;
		font-size: 0.95rem;
		color: var(--on-surface);
	}

	.metric-item.highlight .metric-value {
		color: var(--primary);
	}

	/* Loading & error screens */
	.workspace-overlay {
		position: absolute;
		top: 0;
		left: 0;
		right: 0;
		bottom: 0;
		background: rgba(13, 14, 15, 0.9);
		backdrop-filter: blur(4px);
		z-index: 100;
		display: flex;
		flex-direction: column;
		justify-content: center;
		align-items: center;
		gap: 1rem;
		color: var(--on-surface);
	}

	.loading-sub {
		font-size: 0.8rem;
		color: var(--on-surface-variant);
		font-style: italic;
	}

	.error-overlay {
		background: rgba(26, 12, 12, 0.95);
		border: 1px solid #ff5555;
	}

	.error-icon {
		font-size: 2.5rem;
	}

	.error-details {
		font-family: 'Courier New', Courier, monospace;
		font-size: 0.8rem;
		color: #ff7777;
		max-width: 80%;
		text-align: center;
		background: rgba(0, 0, 0, 0.4);
		padding: 0.5rem 1rem;
		border-radius: var(--roundness);
		border: 1px solid rgba(255, 85, 85, 0.2);
	}

	.retry-btn {
		background: rgba(255, 85, 85, 0.2);
		border: 1px solid #ff5555;
		color: #ffdddd;
		padding: 0.5rem 1.25rem;
		border-radius: var(--roundness);
		font-size: 0.85rem;
		cursor: pointer;
		font-weight: 600;
		transition: background 0.2s;
	}

	.retry-btn:hover {
		background: rgba(255, 85, 85, 0.4);
	}

	.spinner {
		width: 36px;
		height: 36px;
		border: 3px solid var(--outline-variant);
		border-top: 3px solid var(--primary);
		border-radius: 50%;
		animation: spin 1s linear infinite;
	}

	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}

	/* Responsive tweaks */
	@media (max-width: 900px) {
		.demo-wrapper {
			height: auto;
		}

		.demo-container {
			grid-template-columns: 1fr;
			grid-template-rows: auto auto;
		}

		.settings-panel {
			max-height: 400px;
		}

		.viewport-card {
			height: 550px;
		}
	}
</style>
