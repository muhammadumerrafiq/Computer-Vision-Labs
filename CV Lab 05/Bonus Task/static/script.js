/**
 * script.js
 * =========
 * Frontend Controller for HOG Industrial Defect Detection.
 * Computer Vision Lab 05 Bonus Challenge.
 */

// Application State
const state = {
    currentMode: 'image', // 'image' | 'video' | 'webcam'
    selectedImagePayload: null, // { type: 'file'|'sample', data: File|string, is_unseen: bool }
    selectedVideoPayload: null,
    webcamStream: null,
    webcamRunning: false,
    webcamProcessing: false,
    webcamAnimationId: null,
    lastFrameTime: performance.now(),
    frameCount: 0,
    fps: 0,
};

// DOM Elements Initialization on DOMContentLoaded
document.addEventListener('DOMContentLoaded', () => {
    initDropzones();
    initTabNavigation();
    loadModelMetadata();
});

/* =========================================================
   Mode / Tab Switching
   ========================================================= */

function switchMode(mode) {
    if (state.currentMode === 'webcam' && mode !== 'webcam') {
        stopWebcam();
    }
    state.currentMode = mode;

    document.querySelectorAll('.mode-section').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));

    if (mode === 'image') {
        document.getElementById('modeImage').classList.add('active');
        document.getElementById('tabBtnImage').classList.add('active');
    } else if (mode === 'video') {
        document.getElementById('modeVideo').classList.add('active');
        document.getElementById('tabBtnVideo').classList.add('active');
    } else if (mode === 'webcam') {
        document.getElementById('modeWebcam').classList.add('active');
        document.getElementById('tabBtnWebcam').classList.add('active');
        setupWebcamCanvas();
    }
}

function initTabNavigation() {
    // Mode tabs bound in HTML onclick
}

async function loadModelMetadata() {
    try {
        const resp = await fetch('/api/model_info');
        const data = await resp.json();
        console.log('[Model Info]', data);
    } catch (e) {
        console.warn('Could not fetch model info:', e);
    }
}

/* =========================================================
   Dropzone & File Input Handling
   ========================================================= */

function initDropzones() {
    // Image Dropzone
    const imgDrop = document.getElementById('imageDropzone');
    const imgInput = document.getElementById('imageFileInput');

    imgDrop.addEventListener('dragover', (e) => {
        e.preventDefault();
        imgDrop.classList.add('dragover');
    });
    imgDrop.addEventListener('dragleave', () => imgDrop.classList.remove('dragover'));
    imgDrop.addEventListener('drop', (e) => {
        e.preventDefault();
        imgDrop.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleImageFileSelected(e.dataTransfer.files[0], true);
        }
    });
    imgInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleImageFileSelected(e.target.files[0], true);
        }
    });

    // Video Dropzone
    const vidDrop = document.getElementById('videoDropzone');
    const vidInput = document.getElementById('videoFileInput');

    vidDrop.addEventListener('dragover', (e) => {
        e.preventDefault();
        vidDrop.classList.add('dragover');
    });
    vidDrop.addEventListener('dragleave', () => vidDrop.classList.remove('dragover'));
    vidDrop.addEventListener('drop', (e) => {
        e.preventDefault();
        vidDrop.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleVideoFileSelected(e.dataTransfer.files[0]);
        }
    });
    vidInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleVideoFileSelected(e.target.files[0]);
        }
    });
}

/* =========================================================
   MODE 1: Image Inspection Logic
   ========================================================= */

function handleImageFileSelected(file, isUnseen = false) {
    state.selectedImagePayload = {
        type: 'file',
        file: file,
        is_unseen: isUnseen
    };

    const reader = new FileReader();
    reader.onload = (e) => {
        const origImg = document.getElementById('origImage');
        origImg.src = e.target.result;
        origImg.style.display = 'block';
        document.querySelector('#origImageWrapper .placeholder-msg').style.display = 'none';

        // Reset annotated view
        document.getElementById('annotatedImage').style.display = 'none';
        document.querySelector('#annotatedImageWrapper .placeholder-msg').style.display = 'block';

        // Reset decision banner
        resetImageDecisionBanner();

        // Show unseen badge if user uploaded
        document.getElementById('unseenBadge').style.display = isUnseen ? 'inline-block' : 'none';

        // Enable Run Button
        document.getElementById('btnRunImageInspection').disabled = false;
    };
    reader.readAsDataURL(file);
}

function loadSampleImage(filename, isUnseen = false) {
    state.selectedImagePayload = {
        type: 'sample',
        sample_filename: filename,
        is_unseen: isUnseen
    };

    const sampleUrl = `/sample_images/${filename}?t=${Date.now()}`;
    const origImg = document.getElementById('origImage');
    origImg.src = sampleUrl;
    origImg.style.display = 'block';
    document.querySelector('#origImageWrapper .placeholder-msg').style.display = 'none';

    // Reset annotated view
    document.getElementById('annotatedImage').style.display = 'none';
    document.querySelector('#annotatedImageWrapper .placeholder-msg').style.display = 'block';

    resetImageDecisionBanner();
    document.getElementById('unseenBadge').style.display = isUnseen ? 'inline-block' : 'none';
    document.getElementById('btnRunImageInspection').disabled = false;

    // Automatically trigger inspection for instant feedback
    executeImageInspection();
}

function resetImageDecisionBanner() {
    const banner = document.getElementById('imageDecisionBanner');
    banner.className = 'decision-banner idle';
    document.getElementById('imagePredictionText').textContent = 'AWAITING INSPECTION';
    document.getElementById('imageActionText').textContent = 'CLICK "RUN QUALITY INSPECTION"';
    document.getElementById('imageConfidenceVal').textContent = '--%';
    document.getElementById('imageDefectiveWindowsVal').textContent = '-- / --';
    document.getElementById('imageLatencyVal').textContent = '-- ms';
}

async function executeImageInspection() {
    if (!state.selectedImagePayload) return;

    const btn = document.getElementById('btnRunImageInspection');
    const spinner = document.getElementById('imageProcessingSpinner');
    btn.disabled = true;
    spinner.style.display = 'inline-block';

    try {
        let resp;
        if (state.selectedImagePayload.type === 'file') {
            const formData = new FormData();
            formData.append('file', state.selectedImagePayload.file);
            resp = await fetch('/api/predict_image', {
                method: 'POST',
                body: formData
            });
        } else {
            resp = await fetch('/api/predict_image', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    sample_filename: state.selectedImagePayload.sample_filename,
                    is_unseen: state.selectedImagePayload.is_unseen
                })
            });
        }

        const data = await resp.json();
        if (data.status !== 'success') {
            alert('Inspection failed: ' + (data.message || 'Unknown error'));
            return;
        }

        // Display results
        updateImageResultsUI(data);

    } catch (err) {
        console.error('Inspection error:', err);
        alert('Inspection request failed: ' + err.message);
    } finally {
        btn.disabled = false;
        spinner.style.display = 'none';
    }
}

function updateImageResultsUI(data) {
    const banner = document.getElementById('imageDecisionBanner');
    const predText = document.getElementById('imagePredictionText');
    const actionText = document.getElementById('imageActionText');
    const confVal = document.getElementById('imageConfidenceVal');
    const winVal = document.getElementById('imageDefectiveWindowsVal');
    const latVal = document.getElementById('imageLatencyVal');

    if (data.is_defective) {
        banner.className = 'decision-banner defect';
        predText.textContent = 'DEFECTIVE';
        actionText.textContent = 'ACTION: REJECT PRODUCT';
    } else {
        banner.className = 'decision-banner pass';
        predText.textContent = 'PASS';
        actionText.textContent = 'ACTION: ACCEPT PRODUCT';
    }

    confVal.textContent = `${data.confidence_pct}%`;
    winVal.textContent = `${data.defective_windows} / ${data.total_windows}`;
    latVal.textContent = `${data.latency_ms} ms`;

    // Show annotated image
    const annImg = document.getElementById('annotatedImage');
    annImg.src = data.annotated_image;
    annImg.style.display = 'block';
    document.querySelector('#annotatedImageWrapper .placeholder-msg').style.display = 'none';

    if (data.is_unseen) {
        document.getElementById('unseenBadge').style.display = 'inline-block';
    }
}

/* =========================================================
   MODE 2: Video Inspection Logic
   ========================================================= */

function handleVideoFileSelected(file) {
    state.selectedVideoPayload = {
        type: 'file',
        file: file
    };
    document.getElementById('btnRunVideoInspection').disabled = false;
    document.getElementById('videoStatusMsg').textContent = `Loaded: ${file.name}`;
    document.getElementById('videoProgressArea').style.display = 'block';
    document.getElementById('videoProgressBar').style.width = '0%';
    document.getElementById('videoProgressPct').textContent = 'Ready';
}

function loadSampleVideo(filename) {
    state.selectedVideoPayload = {
        type: 'sample',
        sample_filename: filename
    };
    document.getElementById('btnRunVideoInspection').disabled = false;
    document.getElementById('videoStatusMsg').textContent = `Loaded Sample: ${filename}`;
    document.getElementById('videoProgressArea').style.display = 'block';
    document.getElementById('videoProgressBar').style.width = '0%';
    document.getElementById('videoProgressPct').textContent = 'Ready';

    executeVideoInspection();
}

async function executeVideoInspection() {
    if (!state.selectedVideoPayload) return;

    const btn = document.getElementById('btnRunVideoInspection');
    btn.disabled = true;

    const progressArea = document.getElementById('videoProgressArea');
    const progressBar = document.getElementById('videoProgressBar');
    const progressPct = document.getElementById('videoProgressPct');
    const statusMsg = document.getElementById('videoStatusMsg');

    progressArea.style.display = 'block';
    progressBar.style.width = '30%';
    progressPct.textContent = 'Processing...';
    statusMsg.textContent = 'Extracting HOG features & running SVM classifier...';

    try {
        let resp;
        if (state.selectedVideoPayload.type === 'file') {
            const formData = new FormData();
            formData.append('video', state.selectedVideoPayload.file);
            resp = await fetch('/api/inspect_video', {
                method: 'POST',
                body: formData
            });
        } else {
            const formData = new FormData();
            formData.append('sample_filename', state.selectedVideoPayload.sample_filename);
            resp = await fetch('/api/inspect_video', {
                method: 'POST',
                body: formData
            });
        }

        progressBar.style.width = '90%';
        const data = await resp.json();

        if (data.status !== 'success') {
            alert('Video inspection failed: ' + (data.message || 'Unknown error'));
            progressBar.style.width = '0%';
            return;
        }

        progressBar.style.width = '100%';
        progressPct.textContent = '100%';
        statusMsg.textContent = 'Inspection complete.';

        updateVideoResultsUI(data);

    } catch (err) {
        console.error('Video inspection error:', err);
        alert('Video inspection failed: ' + err.message);
    } finally {
        btn.disabled = false;
    }
}

function updateVideoResultsUI(data) {
    document.getElementById('videoLiveBadge').style.display = 'inline-block';

    const banner = document.getElementById('videoDecisionBanner');
    const overallText = document.getElementById('videoOverallResultText');
    const overallAction = document.getElementById('videoOverallActionText');

    if (data.overall_is_defective) {
        banner.className = 'decision-banner defect';
        overallText.textContent = `OVERALL: ${data.overall_prediction}`;
        overallAction.textContent = `ACTION: ${data.overall_action} (${data.defect_rate_pct}% Defect Rate)`;
    } else {
        banner.className = 'decision-banner pass';
        overallText.textContent = `OVERALL: ${data.overall_prediction}`;
        overallAction.textContent = `ACTION: ${data.overall_action} (100% Pass Rate)`;
    }

    document.getElementById('videoProcessedFramesVal').textContent = data.processed_frames;
    document.getElementById('videoDefectiveFramesVal').textContent = data.defective_frames;
    document.getElementById('videoPassFramesVal').textContent = data.pass_frames;

    // Render keyframes gallery
    const grid = document.getElementById('videoKeyframesGrid');
    grid.innerHTML = '';

    if (!data.keyframes || data.keyframes.length === 0) {
        grid.innerHTML = '<div class="placeholder-msg">No keyframes extracted</div>';
        return;
    }

    data.keyframes.forEach(kf => {
        const card = document.createElement('div');
        const isDefect = kf.prediction === 'DEFECTIVE';
        card.className = `keyframe-card ${isDefect ? 'defect' : 'pass'}`;
        card.innerHTML = `
            <img src="${kf.image}" alt="Frame ${kf.frame_idx}">
            <div class="keyframe-meta">
                <span>Frame #${kf.frame_idx}</span>
                <strong style="color: ${isDefect ? '#ff3b56' : '#00e676'}">${kf.prediction}</strong>
            </div>
        `;
        grid.appendChild(card);
    });
}

/* =========================================================
   MODE 3: Live Webcam Inspection Logic
   ========================================================= */

const ROI_SIZE = 200; // Expected 200x200 inspection region matching the trained model

function setupWebcamCanvas() {
    const canvas = document.getElementById('webcamCanvas');
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#0a0d14';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

async function startWebcam() {
    const video = document.getElementById('webcamVideo');
    const offOverlay = document.getElementById('cameraOffOverlay');
    const btnStart = document.getElementById('btnStartWebcam');
    const btnStop = document.getElementById('btnStopWebcam');
    const recDot = document.getElementById('webcamRecDot');
    const liveStatus = document.getElementById('liveStreamStatus');

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'environment'
            },
            audio: false
        });

        state.webcamStream = stream;
        video.srcObject = stream;
        await video.play();

        state.webcamRunning = true;
        offOverlay.style.display = 'none';
        btnStart.disabled = true;
        btnStop.disabled = false;
        recDot.classList.add('active');
        liveStatus.textContent = 'STREAMING';

        // Start processing loop
        state.lastFrameTime = performance.now();
        state.frameCount = 0;
        requestAnimationFrame(processWebcamLoop);

    } catch (err) {
        console.error('Camera access error:', err);
        alert('Could not access webcam: ' + err.message + '\nPlease allow camera permissions in your browser.');
    }
}

function stopWebcam() {
    state.webcamRunning = false;

    if (state.webcamStream) {
        state.webcamStream.getTracks().forEach(track => track.stop());
        state.webcamStream = null;
    }

    if (state.webcamAnimationId) {
        cancelAnimationFrame(state.webcamAnimationId);
        state.webcamAnimationId = null;
    }

    document.getElementById('cameraOffOverlay').style.display = 'flex';
    document.getElementById('btnStartWebcam').disabled = false;
    document.getElementById('btnStopWebcam').disabled = true;
    document.getElementById('webcamRecDot').classList.remove('active');
    document.getElementById('liveStreamStatus').textContent = 'OFFLINE';

    // Reset status banner to idle
    const card = document.getElementById('liveStatusCard');
    card.className = 'live-status-card idle';
    document.getElementById('liveStatusText').textContent = 'STANDBY';
    document.getElementById('liveActionText').textContent = 'START WEBCAM TO INSPECT';
    document.getElementById('webcamFpsVal').textContent = '--';
    document.getElementById('webcamLatencyVal').textContent = '-- ms';
}

// Cached latest prediction results for overlay rendering
let latestWebcamResult = null;

async function processWebcamLoop(timestamp) {
    if (!state.webcamRunning) return;

    const video = document.getElementById('webcamVideo');
    const canvas = document.getElementById('webcamCanvas');
    const ctx = canvas.getContext('2d');

    // FPS calculation
    state.frameCount++;
    const elapsed = timestamp - state.lastFrameTime;
    if (elapsed >= 1000) {
        state.fps = Math.round((state.frameCount * 1000) / elapsed);
        document.getElementById('webcamFpsVal').textContent = state.fps;
        state.frameCount = 0;
        state.lastFrameTime = timestamp;
    }

    if (video.readyState === video.HAVE_ENOUGH_DATA) {
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
        }

        // Draw camera frame
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Center Inspection Area (200x200)
        const roiX = Math.round((canvas.width - ROI_SIZE) / 2);
        const roiY = Math.round((canvas.height - ROI_SIZE) / 2);

        // Draw Inspection Region Reticle & Overlay on Canvas
        drawInspectionReticle(ctx, roiX, roiY, ROI_SIZE, ROI_SIZE, latestWebcamResult);

        // Update mini ROI zoom canvas
        updateRoiZoomCanvas(video, roiX, roiY, ROI_SIZE, ROI_SIZE);

        // Trigger Async Backend Prediction if not currently waiting on a response
        if (!state.webcamProcessing) {
            state.webcamProcessing = true;
            sendRoiForPrediction(canvas, roiX, roiY, ROI_SIZE, ROI_SIZE);
        }
    }

    state.webcamAnimationId = requestAnimationFrame(processWebcamLoop);
}

function updateRoiZoomCanvas(video, rx, ry, rw, rh) {
    const roiCanvas = document.getElementById('roiCanvas');
    const roiCtx = roiCanvas.getContext('2d');
    roiCtx.drawImage(video, rx, ry, rw, rh, 0, 0, roiCanvas.width, roiCanvas.height);
}

async function sendRoiForPrediction(fullCanvas, rx, ry, rw, rh) {
    try {
        // Extract crop to an in-memory canvas
        const cropCanvas = document.createElement('canvas');
        cropCanvas.width = rw;
        cropCanvas.height = rh;
        const cropCtx = cropCanvas.getContext('2d');
        cropCtx.drawImage(fullCanvas, rx, ry, rw, rh, 0, 0, rw, rh);

        const cropB64 = cropCanvas.toDataURL('image/jpeg', 0.85);

        const t0 = performance.now();
        const resp = await fetch('/api/predict_frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                frame: cropB64,
                roi: { x: 0, y: 0, w: rw, h: rh }
            })
        });

        const data = await resp.json();
        const clientRoundtrip = Math.round(performance.now() - t0);

        if (data.status === 'success') {
            latestWebcamResult = data;
            document.getElementById('webcamLatencyVal').textContent = `${data.latency_ms} ms`;
            updateLiveWebcamUI(data);
        }
    } catch (e) {
        console.warn('Frame predict error:', e);
    } finally {
        state.webcamProcessing = false;
    }
}

function updateLiveWebcamUI(data) {
    const card = document.getElementById('liveStatusCard');
    const statusText = document.getElementById('liveStatusText');
    const actionText = document.getElementById('liveActionText');
    const overlayBox = document.getElementById('inspectionOverlay');

    if (data.is_defective) {
        card.className = 'live-status-card defect';
        statusText.textContent = 'DEFECTIVE';
        actionText.textContent = 'ACTION: REJECT PRODUCT';
        overlayBox.className = 'inspection-overlay-box defect';
    } else {
        card.className = 'live-status-card pass';
        statusText.textContent = 'PASS';
        actionText.textContent = 'ACTION: ACCEPT PRODUCT';
        overlayBox.className = 'inspection-overlay-box pass';
    }

    document.getElementById('liveConfidenceVal').textContent = `${data.confidence_pct}%`;
    document.getElementById('liveDefectCountVal').textContent = data.defective_windows;
}

function drawInspectionReticle(ctx, rx, ry, rw, rh, res) {
    // Darken outside area slightly
    ctx.save();
    ctx.fillStyle = 'rgba(0, 0, 0, 0.35)';
    ctx.fillRect(0, 0, ctx.canvas.width, ry); // top
    ctx.fillRect(0, ry + rh, ctx.canvas.width, ctx.canvas.height - (ry + rh)); // bottom
    ctx.fillRect(0, ry, rx, rh); // left
    ctx.fillRect(rx + rw, ry, ctx.canvas.width - (rx + rw), rh); // right

    // Reticle border
    const isDefect = res && res.is_defective;
    ctx.lineWidth = 2;
    ctx.strokeStyle = isDefect ? '#ff3b56' : '#00e676';
    ctx.strokeRect(rx, ry, rw, rh);

    // Draw detected defect boxes inside the ROI
    if (res && res.boxes && res.boxes.length > 0) {
        ctx.strokeStyle = '#ff2244';
        ctx.lineWidth = 2;
        ctx.fillStyle = 'rgba(255, 34, 68, 0.25)';

        res.boxes.forEach(b => {
            const bx = rx + b.x;
            const by = ry + b.y;
            ctx.strokeRect(bx, by, b.w, b.h);
            ctx.fillRect(bx, by, b.w, b.h);

            // Defect badge
            ctx.fillStyle = '#ff2244';
            ctx.fillRect(bx, Math.max(by - 14, 0), 62, 14);
            ctx.fillStyle = '#ffffff';
            ctx.font = '10px JetBrains Mono, monospace';
            ctx.fillText(`DEFECT ${Math.round(b.prob * 100)}%`, bx + 2, Math.max(by - 3, 11));
        });
    }

    ctx.restore();
}
