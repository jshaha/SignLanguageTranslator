#!/usr/bin/env python3
"""
Real-time sign language classifier using MediaPipe landmark coordinates.
Uses hand geometry for robust, lighting-independent recognition.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
from pathlib import Path
import pickle

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    print("Installing MediaPipe...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'mediapipe'])
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

# Configuration
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "sign_language_landmark_model.keras"
METADATA_PATH = MODEL_DIR / "landmark_model_metadata.json"
SCALER_PATH = MODEL_DIR / "landmark_scaler.pkl"
HAND_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

def initialize_mediapipe():
    """Initialize MediaPipe Hand Landmarker"""
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
    return detector

def load_model_and_metadata():
    """Load the trained landmark model and metadata"""
    print("Loading landmark-based model...")

    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please run train_landmark_model.py first!")
        return None, None, None

    model = keras.models.load_model(MODEL_PATH)

    # Load metadata
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)

    # Load scaler
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    classes = metadata['classes']

    print(f"Model loaded successfully!")
    print(f"Classes: {classes}")
    print(f"Features: {metadata['num_features']} (21 landmarks × 3 coordinates)")
    print(f"Training accuracy: {metadata['final_train_accuracy']*100:.2f}%")
    print(f"Validation accuracy: {metadata['final_val_accuracy']*100:.2f}%")

    return model, metadata, scaler

def extract_landmarks(frame, detector):
    """
    Extract hand landmarks from frame.
    Returns 63 features (21 landmarks × 3 coordinates) if hand detected.
    """
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect hands
    results = detector.detect(mp_image)

    if results.hand_landmarks and len(results.hand_landmarks) > 0:
        landmarks = results.hand_landmarks[0]

        # Extract x, y, z coordinates for all 21 landmarks
        features = []
        for landmark in landmarks:
            features.extend([landmark.x, landmark.y, landmark.z])

        return np.array(features, dtype=np.float32), landmarks, True

    return None, None, False

def draw_landmarks(frame, landmarks):
    """Draw hand landmarks and connections on frame"""
    if landmarks is None:
        return frame

    h, w = frame.shape[:2]

    # Draw connections
    connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
    for connection in connections:
        start_idx = connection.start
        end_idx = connection.end
        start_point = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
        end_point = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
        cv2.line(frame, start_point, end_point, (0, 255, 255), 2)

    # Draw landmarks
    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    return frame

def draw_prediction_box(frame, prediction, confidence, fps):
    """Draw prediction info on frame"""
    height, width = frame.shape[:2]

    # Create semi-transparent overlay
    overlay = frame.copy()

    # Draw background box
    cv2.rectangle(overlay, (10, 10), (320, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Draw prediction text
    cv2.putText(frame, f"Sign: {prediction}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

    # Draw confidence
    cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw confidence bar
    bar_width = int(280 * confidence)
    cv2.rectangle(frame, (20, 105), (300, 125), (100, 100, 100), -1)

    # Color based on confidence
    if confidence > 0.8:
        color = (0, 255, 0)  # Green
    elif confidence > 0.5:
        color = (0, 255, 255)  # Yellow
    else:
        color = (0, 0, 255)  # Red

    cv2.rectangle(frame, (20, 105), (20 + bar_width, 125), color, -1)

    # Draw FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw method indicator
    cv2.putText(frame, "Landmark-Based AI", (width - 250, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)

    return frame

def main():
    """Main real-time classification loop"""
    # Load model
    model, metadata, scaler = load_model_and_metadata()

    if model is None:
        return

    classes = metadata['classes']

    # Initialize MediaPipe
    print("\nInitializing MediaPipe hand detector...")
    detector = initialize_mediapipe()
    print("MediaPipe initialized!")

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n" + "="*60)
    print("REAL-TIME SIGN LANGUAGE CLASSIFIER")
    print("Landmark-Based AI - Robust & Lighting-Independent")
    print("="*60)
    print("Instructions:")
    print("- Place your hand in the blue rectangle")
    print("- Make a sign language letter")
    print("- Hand landmarks will be extracted automatically")
    print("- The AI uses hand geometry, not pixels!")
    print("- Press 'w' to quit")
    print("="*60 + "\n")

    # For FPS calculation
    import time
    prev_time = time.time()
    fps = 0

    # Smoothing
    predictions_history = []
    HISTORY_SIZE = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Draw ROI rectangle
        roi_x1, roi_y1 = 100, 100
        roi_x2, roi_y2 = 540, 380
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)

        # Extract ROI
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        # Extract landmarks from ROI
        features, landmarks, hand_detected = extract_landmarks(roi, detector)

        if hand_detected:
            # Draw landmarks on ROI
            roi = draw_landmarks(roi, landmarks)
            frame[roi_y1:roi_y2, roi_x1:roi_x2] = roi

            # Normalize features using scaler
            features = features.reshape(1, -1)
            features_normalized = scaler.transform(features)

            # Make prediction
            predictions = model.predict(features_normalized, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            predicted_letter = classes[predicted_class_idx]

            # Add to history for smoothing
            predictions_history.append(predicted_letter)
            if len(predictions_history) > HISTORY_SIZE:
                predictions_history.pop(0)

            # Use most common prediction in history
            if len(predictions_history) >= 3:
                from collections import Counter
                predicted_letter = Counter(predictions_history).most_common(1)[0][0]

        else:
            predicted_letter = "No hand"
            confidence = 0.0
            predictions = [np.zeros(len(classes))]

        # Draw prediction info
        frame = draw_prediction_box(frame, predicted_letter, confidence, fps)

        # Show top 3 predictions (for debugging)
        if hand_detected:
            top_3_idx = np.argsort(predictions[0])[-3:][::-1]
            y_offset = 200
            cv2.putText(frame, "Top 3 Predictions:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            for i, idx in enumerate(top_3_idx):
                text = f"{i+1}. {classes[idx]}: {predictions[0][idx]*100:.1f}%"
                cv2.putText(frame, text, (10, y_offset + 25 + i*20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Display frame
        cv2.imshow('Sign Language Classifier - Landmark AI', frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('w'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("\nClassifier closed.")

if __name__ == "__main__":
    main()
