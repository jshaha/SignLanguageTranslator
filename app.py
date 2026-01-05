#!/usr/bin/env python3
"""
Flask web application for sign language translator.
Provides API endpoints for real-time sign language classification.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import pickle
import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    print("MediaPipe not installed. Please install it: pip install mediapipe")
    mp = None
    python = None
    vision = None

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "sign_language_landmark_model.keras"
METADATA_PATH = MODEL_DIR / "landmark_model_metadata.json"
SCALER_PATH = MODEL_DIR / "landmark_scaler.pkl"
HAND_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

# Global variables for model and detector
model = None
metadata = None
scaler = None
detector = None
classes = None

def initialize_mediapipe():
    """Initialize MediaPipe Hand Landmarker"""
    global detector
    
    if mp is None or python is None or vision is None:
        raise ImportError("MediaPipe is not installed. Please install it: pip install mediapipe")
    
    if not HAND_MODEL_PATH.exists():
        print("Downloading MediaPipe hand landmark model...")
        import urllib.request
        HAND_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(model_url, HAND_MODEL_PATH)
        print("Model downloaded!")

    base_options = python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )
    detector = vision.HandLandmarker.create_from_options(options)
    print("MediaPipe initialized!")

def load_model():
    """Load the trained model and metadata"""
    global model, metadata, scaler, classes
    
    print("Loading sign language model...")
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}\n"
            "Please train the model first by running: python scripts/train_landmark_model.py"
        )
    
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found at {METADATA_PATH}\n"
            "Please train the model first by running: python scripts/train_landmark_model.py"
        )
    
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found at {SCALER_PATH}\n"
            "Please train the model first by running: python scripts/train_landmark_model.py"
        )
    
    model = keras.models.load_model(MODEL_PATH)
    
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    classes = metadata['classes']
    
    print(f"Model loaded! Classes: {len(classes)} letters (A-Z)")
    print(f"Training accuracy: {metadata['final_val_accuracy']*100:.2f}%")

def extract_landmarks_from_image(image_data):
    """
    Extract hand landmarks from image data.
    Returns features array if hand detected, None otherwise.
    """
    global detector, mp
    
    if detector is None or mp is None:
        raise RuntimeError("MediaPipe detector not initialized")
    
    # Decode base64 image
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert('RGB')
        img_array = np.array(image)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None, False
    
    # Convert to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)
    
    # Detect hands
    results = detector.detect(mp_image)
    
    if results.hand_landmarks and len(results.hand_landmarks) > 0:
        landmarks = results.hand_landmarks[0]
        
        # Extract x, y, z coordinates for all 21 landmarks
        features = []
        for landmark in landmarks:
            features.extend([landmark.x, landmark.y, landmark.z])
        
        return np.array(features, dtype=np.float32), True
    
    return None, False

@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for sign language prediction"""
    global model, scaler, classes
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Extract landmarks
        features, hand_detected = extract_landmarks_from_image(image_data)
        
        if not hand_detected:
            return jsonify({
                'success': False,
                'message': 'No hand detected',
                'prediction': None,
                'confidence': 0.0,
                'top_predictions': []
            })
        
        # Normalize features
        features = features.reshape(1, -1)
        features_normalized = scaler.transform(features)
        
        # Make prediction
        predictions = model.predict(features_normalized, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_letter = classes[predicted_class_idx]
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_predictions = [
            {
                'letter': classes[idx],
                'confidence': float(predictions[0][idx])
            }
            for idx in top_3_idx
        ]
        
        return jsonify({
            'success': True,
            'prediction': predicted_letter,
            'confidence': confidence,
            'top_predictions': top_predictions
        })
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes': classes if classes else []
    })

@app.route('/api/info', methods=['GET'])
def info():
    """Get model information"""
    if metadata is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_type': metadata.get('model_type', 'unknown'),
        'num_classes': metadata.get('num_classes', 0),
        'classes': metadata.get('classes', []),
        'accuracy': metadata.get('final_val_accuracy', 0),
        'feature_description': metadata.get('feature_description', '')
    })

if __name__ == '__main__':
    # Load model and initialize MediaPipe
    try:
        load_model()
        initialize_mediapipe()
    except Exception as e:
        print(f"Error initializing: {e}")
        print("Make sure you have trained the model first!")
        exit(1)
    
    print("\n" + "="*60)
    print("SIGN LANGUAGE TRANSLATOR WEB APP")
    print("="*60)
    
    # Get port from environment variable (for Render, Heroku, etc.) or use default
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    print(f"Starting Flask server on port {port}...")
    print(f"Open your browser and navigate to: http://localhost:{port}")
    print("="*60 + "\n")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=port, debug=debug)

