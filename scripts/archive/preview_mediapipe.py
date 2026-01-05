#!/usr/bin/env python3
"""
Preview MediaPipe hand detection before processing all images.
"""

import cv2
import numpy as np
from pathlib import Path
import random

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

DATA_DIR = Path(__file__).parent.parent / "data" / "training"

# Download hand landmark model if not exists
MODEL_PATH = Path(__file__).parent.parent / "models" / "hand_landmarker.task"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    print("Downloading MediaPipe hand landmark model...")
    import urllib.request
    model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(model_url, MODEL_PATH)
    print("Model downloaded!")

# Initialize MediaPipe HandLandmarker
base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)
detector = vision.HandLandmarker.create_from_options(options)

def detect_hand_mediapipe(img):
    """Detect hand and create mask"""
    # Convert numpy array to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

    # Detect hands
    results = detector.detect(mp_image)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    if results.hand_landmarks:
        hand_landmarks = results.hand_landmarks[0]
        h, w = img.shape[:2]
        points = []

        for landmark in hand_landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append([x, y])

        points = np.array(points, dtype=np.int32)
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (11, 11), 0)

        return mask, True, hand_landmarks

    return mask, False, None

def add_random_background(img, mask):
    """Add random background"""
    bg_colors = [(255, 255, 255), (0, 0, 0), (128, 128, 128), (200, 200, 200)]
    bg_color = random.choice(bg_colors)
    background = np.full_like(img, bg_color, dtype=np.uint8)

    mask_normalized = mask.astype(np.float32) / 255.0
    mask_3channel = np.stack([mask_normalized] * 3, axis=-1)

    result = (img * mask_3channel + background * (1 - mask_3channel)).astype(np.uint8)
    return result

def preview_images(letter='A', num_samples=10):
    """Preview MediaPipe detection"""
    print("\n" + "="*60)
    print(f"MEDIAPIPE HAND DETECTION PREVIEW - Letter '{letter}'")
    print("="*60)

    letter_dir = DATA_DIR / letter
    if not letter_dir.exists():
        print(f"Error: No folder for '{letter}'")
        return

    image_files = list(letter_dir.glob("*.jpg"))[:num_samples]

    if len(image_files) == 0:
        print(f"No images found")
        return

    print(f"\nPreviewing {len(image_files)} images")
    print("\nControls:")
    print("  SPACE: Next image")
    print("  W: Quit")
    print()

    detected_count = 0

    for idx, img_path in enumerate(image_files):
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect hand
        mask, detected, landmarks = detect_hand_mediapipe(img)

        if detected:
            detected_count += 1
            status = "✓ DETECTED"
        else:
            status = "✗ NOT DETECTED"

        print(f"Image {idx+1}/{len(image_files)}: {status}")

        # Create result with background
        if detected:
            result = add_random_background(img, mask)
            mask_coverage = (np.sum(mask > 128) / mask.size) * 100
            print(f"  Hand coverage: {mask_coverage:.1f}%")
        else:
            result = img.copy()

        # Visualize
        h, w = img.shape[:2]
        display_h = 400
        display_w = int(w * display_h / h)

        # Draw landmarks on original
        img_with_landmarks = img.copy()
        if landmarks:
            # Draw hand skeleton manually
            h, w = img.shape[:2]
            # Draw landmarks
            for landmark in landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(img_with_landmarks, (x, y), 3, (0, 255, 0), -1)

            # Draw connections
            connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
            for connection in connections:
                start_idx = connection.start
                end_idx = connection.end
                start_point = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
                end_point = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
                cv2.line(img_with_landmarks, start_point, end_point, (0, 255, 255), 2)

        img_display = cv2.resize(img_with_landmarks, (display_w, display_h))
        mask_display = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), (display_w, display_h))
        result_display = cv2.resize(result, (display_w, display_h))

        # Stack horizontally
        combined = np.hstack([img_display, mask_display, result_display])

        # Add labels
        cv2.putText(combined, "Original + Landmarks", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined, "Hand Mask", (display_w + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined, "Result", (2*display_w + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Convert to BGR for display
        combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

        cv2.imshow('MediaPipe Hand Detection Preview', combined)

        # Wait for key
        key = cv2.waitKey(0) & 0xFF
        if key == ord('w'):
            break

    cv2.destroyAllWindows()

    print("\n" + "="*60)
    print(f"Detection success: {detected_count}/{len(image_files)} images")
    print("="*60)

    if detected_count < len(image_files) * 0.7:
        print("\n⚠ Low detection rate. Possible issues:")
        print("  - Images too dark or blurry")
        print("  - Hand partially cut off in frame")
        print("  - Unusual hand positions")
    else:
        print("\n✓ Good detection rate!")
        print("  Run preprocess_mediapipe.py to process all images")

def main():
    import sys
    letter = 'A'
    if len(sys.argv) > 1:
        letter = sys.argv[1].upper()

    preview_images(letter, num_samples=10)
    detector.close()

if __name__ == "__main__":
    main()
