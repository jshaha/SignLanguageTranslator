// Sign Language Translator - Frontend JavaScript

let video = null;
let canvas = null;
let stream = null;
let isRunning = false;
let predictionInterval = null;
let predictionsHistory = [];
const HISTORY_SIZE = 5;
const PREDICTION_INTERVAL = 200; // ms

// DOM elements
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const videoElement = document.getElementById('video');
const canvasElement = document.getElementById('canvas');
const statusElement = document.getElementById('status');
const predictionElement = document.getElementById('prediction');
const confidenceText = document.getElementById('confidenceText');
const confidenceFill = document.getElementById('confidenceFill');
const topPredictionsElement = document.getElementById('topPredictions');
let videoPredictionElement = null;
let videoPredictionLetter = null;
let videoPredictionConfidence = null;
let videoTopPredictionsElement = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Initialize overlay elements
    videoPredictionElement = document.getElementById('videoPrediction');
    if (videoPredictionElement) {
        videoPredictionLetter = videoPredictionElement.querySelector('.video-prediction-letter');
        videoPredictionConfidence = videoPredictionElement.querySelector('.video-prediction-confidence');
    }
    videoTopPredictionsElement = document.getElementById('videoTopPredictions');
    
    loadModelInfo();
    setupEventListeners();
});

function setupEventListeners() {
    startBtn.addEventListener('click', startCamera);
    stopBtn.addEventListener('click', stopCamera);
}

async function loadModelInfo() {
    try {
        const response = await fetch('/api/info');
        const data = await response.json();
        
        if (data.accuracy) {
            document.getElementById('modelAccuracy').textContent = 
                `${(data.accuracy * 100).toFixed(1)}%`;
        }
        
        if (data.classes) {
            document.getElementById('modelClasses').textContent = 
                `${data.classes.length} letters (${data.classes.join(', ')})`;
        }
    } catch (error) {
        console.error('Error loading model info:', error);
    }
}

async function startCamera() {
    try {
        statusElement.textContent = 'Requesting camera access...';
        
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            }
        });
        
        videoElement.srcObject = stream;
        video = videoElement;
        canvas = canvasElement;
        
        await video.play();
        
        statusElement.textContent = 'Camera active - Show your hand!';
        startBtn.disabled = true;
        stopBtn.disabled = false;
        isRunning = true;
        
        // Start prediction loop
        startPredictionLoop();
        
    } catch (error) {
        console.error('Error accessing camera:', error);
        statusElement.textContent = 'Error: Could not access camera';
        alert('Could not access camera. Please allow camera permissions and try again.');
    }
}

function stopCamera() {
    isRunning = false;
    
    if (predictionInterval) {
        clearInterval(predictionInterval);
        predictionInterval = null;
    }
    
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    
    videoElement.srcObject = null;
    statusElement.textContent = 'Camera stopped';
    startBtn.disabled = false;
    stopBtn.disabled = true;
    
    // Reset UI
    predictionElement.textContent = '-';
    confidenceText.textContent = '0%';
    confidenceFill.style.width = '0%';
    topPredictionsElement.innerHTML = '<div class="prediction-item">-</div><div class="prediction-item">-</div><div class="prediction-item">-</div>';
    predictionsHistory = [];
    
    // Reset video overlays
    if (videoPredictionElement) {
        videoPredictionElement.classList.remove('active');
        if (videoPredictionLetter) videoPredictionLetter.textContent = '-';
        if (videoPredictionConfidence) videoPredictionConfidence.textContent = '0%';
    }
    if (videoTopPredictionsElement) {
        videoTopPredictionsElement.classList.remove('active');
        videoTopPredictionsElement.innerHTML = '';
    }
}

function startPredictionLoop() {
    if (predictionInterval) {
        clearInterval(predictionInterval);
    }
    
    predictionInterval = setInterval(async () => {
        if (!isRunning || !video || video.readyState !== video.HAVE_ENOUGH_DATA) {
            return;
        }
        
        await predictSign();
    }, PREDICTION_INTERVAL);
}

async function predictSign() {
    try {
        // Capture frame from video
        const width = video.videoWidth;
        const height = video.videoHeight;
        
        canvas.width = width;
        canvas.height = height;
        
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, width, height);
        
        // Convert to base64
        const imageData = canvas.toDataURL('image/jpeg', 0.8);
        
        // Send to API
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        });
        
        const data = await response.json();
        
        if (data.success && data.prediction) {
            updatePrediction(data);
            updateVideoOverlay(data);
        } else if (data.message === 'No hand detected') {
            statusElement.textContent = 'No hand detected - Show your hand in the blue box';
            predictionElement.textContent = '-';
            confidenceText.textContent = '0%';
            confidenceFill.style.width = '0%';
            predictionsHistory = [];
            
            // Hide video overlays
            if (videoPredictionElement) {
                videoPredictionElement.classList.remove('active');
            }
            if (videoTopPredictionsElement) {
                videoTopPredictionsElement.classList.remove('active');
            }
        }
        
    } catch (error) {
        console.error('Error making prediction:', error);
        statusElement.textContent = 'Error making prediction';
    }
}

function updatePrediction(data) {
    const { prediction, confidence, top_predictions } = data;
    
    // Add to history for smoothing
    predictionsHistory.push(prediction);
    if (predictionsHistory.length > HISTORY_SIZE) {
        predictionsHistory.shift();
    }
    
    // Use most common prediction in history (smoothing)
    let finalPrediction = prediction;
    if (predictionsHistory.length >= 3) {
        const counts = {};
        predictionsHistory.forEach(p => {
            counts[p] = (counts[p] || 0) + 1;
        });
        finalPrediction = Object.keys(counts).reduce((a, b) => 
            counts[a] > counts[b] ? a : b
        );
    }
    
    // Update prediction display
    if (predictionElement.textContent !== finalPrediction) {
        predictionElement.textContent = finalPrediction;
        predictionElement.classList.add('active');
        setTimeout(() => {
            predictionElement.classList.remove('active');
        }, 500);
    }
    
    // Update confidence
    const confidencePercent = Math.round(confidence * 100);
    confidenceText.textContent = `${confidencePercent}%`;
    confidenceFill.style.width = `${confidencePercent}%`;
    
    // Update confidence bar color
    confidenceFill.className = 'confidence-fill';
    if (confidence > 0.8) {
        confidenceFill.classList.add('high');
    } else if (confidence > 0.5) {
        confidenceFill.classList.add('medium');
    } else {
        confidenceFill.classList.add('low');
    }
    
    // Update top predictions
    if (top_predictions && top_predictions.length > 0) {
        topPredictionsElement.innerHTML = top_predictions.map((pred, index) => `
            <div class="prediction-item">
                <span class="letter">${pred.letter}</span>
                <span class="confidence-value">${Math.round(pred.confidence * 100)}%</span>
            </div>
        `).join('');
    }
    
    statusElement.textContent = 'Camera active - Show your hand!';
}

function updateVideoOverlay(data) {
    const { prediction, confidence, top_predictions } = data;
    
    // Use smoothed prediction from history
    let finalPrediction = prediction;
    if (predictionsHistory.length >= 3) {
        const counts = {};
        predictionsHistory.forEach(p => {
            counts[p] = (counts[p] || 0) + 1;
        });
        finalPrediction = Object.keys(counts).reduce((a, b) => 
            counts[a] > counts[b] ? a : b
        );
    }
    
    // Update video prediction overlay
    if (videoPredictionElement && videoPredictionLetter && videoPredictionConfidence) {
        // Update letter
        if (videoPredictionLetter.textContent !== finalPrediction) {
            videoPredictionLetter.textContent = finalPrediction;
            videoPredictionLetter.classList.add('pulse');
            setTimeout(() => {
                videoPredictionLetter.classList.remove('pulse');
            }, 500);
        }
        
        // Update confidence
        const confidencePercent = Math.round(confidence * 100);
        videoPredictionConfidence.textContent = `${confidencePercent}%`;
        
        // Update confidence color
        videoPredictionConfidence.className = 'video-prediction-confidence';
        if (confidence > 0.8) {
            videoPredictionConfidence.classList.add('high');
        } else if (confidence > 0.5) {
            videoPredictionConfidence.classList.add('medium');
        } else {
            videoPredictionConfidence.classList.add('low');
        }
        
        // Show overlay
        videoPredictionElement.classList.add('active');
    }
    
    // Update top predictions overlay
    if (videoTopPredictionsElement && top_predictions && top_predictions.length > 0) {
        videoTopPredictionsElement.innerHTML = `
            <div class="video-top-predictions-title">Top Predictions</div>
            ${top_predictions.map((pred, index) => `
                <div class="video-top-prediction-item">
                    <span class="letter">${pred.letter}</span>
                    <span class="confidence">${Math.round(pred.confidence * 100)}%</span>
                </div>
            `).join('')}
        `;
        videoTopPredictionsElement.classList.add('active');
    }
}

// Health check on load
fetch('/api/health')
    .then(res => res.json())
    .then(data => {
        if (!data.model_loaded) {
            statusElement.textContent = 'Error: Model not loaded. Please check server.';
            startBtn.disabled = true;
        }
    })
    .catch(error => {
        console.error('Health check failed:', error);
        statusElement.textContent = 'Error: Cannot connect to server';
        startBtn.disabled = true;
    });

