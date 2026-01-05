#!/usr/bin/env python3
"""
Process training images with MediaPipe hand landmarks overlay.
Draws hand skeleton and keypoints on original images for training.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    print("MediaPipe not installed. Installing now...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'mediapipe'])
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data" / "training"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed_landmarks"

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
    """
    Detect hand using MediaPipe and return landmarks.
    """
    # Convert numpy array to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

    # Detect hands
    results = detector.detect(mp_image)

    if results.hand_landmarks:
        return results.hand_landmarks[0], True  # Successfully detected

    return None, False  # No hand detected

def draw_landmarks_on_image(img, landmarks):
    """
    Draw hand landmarks and connections on the image.
    Returns a copy of the image with landmarks drawn.
    """
    result = img.copy()
    h, w = result.shape[:2]

    # Draw connections first (so they appear under the landmarks)
    connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
    for connection in connections:
        start_idx = connection.start
        end_idx = connection.end
        start_point = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
        end_point = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
        cv2.line(result, start_point, end_point, (0, 255, 255), 2)

    # Draw landmarks
    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(result, (x, y), 3, (0, 255, 0), -1)

    return result

def process_image(img_path, letter, output_folder):
    """Process a single image with MediaPipe hand detection and landmark overlay"""
    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        return 0

    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect hand with MediaPipe
    landmarks, detected = detect_hand_mediapipe(img)

    if not detected:
        # No hand detected - skip this image
        print(f"  ⚠ No hand detected in {img_path.name}, skipping")
        return 0

    # Draw landmarks on image
    result = draw_landmarks_on_image(img, landmarks)

    # Save processed image
    output_path = output_folder / f"landmarks_{img_path.name}"
    cv2.imwrite(str(output_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    return 1

def main():
    """Process all images using MediaPipe"""
    print("\n" + "="*60)
    print("MEDIAPIPE HAND LANDMARK OVERLAY")
    print("Drawing hand skeleton on training images")
    print("="*60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_processed = 0
    total_created = 0
    total_failed = 0

    # Process each letter
    for letter_folder in sorted(DATA_DIR.iterdir()):
        if not letter_folder.is_dir():
            continue

        letter = letter_folder.name
        print(f"\nProcessing letter '{letter}'...")

        # Create output folder
        output_folder = OUTPUT_DIR / letter
        output_folder.mkdir(parents=True, exist_ok=True)

        # Get images
        image_files = list(letter_folder.glob("*.jpg"))

        if len(image_files) == 0:
            print(f"  No images found")
            continue

        # Process each image
        detected_count = 0
        for img_path in tqdm(image_files, desc=f"  {letter}"):
            count = process_image(img_path, letter, output_folder)

            if count > 0:
                detected_count += 1
                total_processed += 1
                total_created += count
            else:
                total_failed += 1

        print(f"  ✓ {detected_count}/{len(image_files)} images had hands detected")

    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"\nImages with landmarks drawn: {total_created}")
    print(f"Images successfully processed: {total_processed}")
    print(f"Failed detections (skipped): {total_failed}")

    if total_failed > total_processed * 0.3:
        print("\n⚠ Warning: Many images had no hand detected")
        print("  This might happen if:")
        print("  - Images are too dark/blurry")
        print("  - Hand is partially cut off")
        print("  - Camera angle is unusual")

    print(f"\nProcessed images saved to: {OUTPUT_DIR}")
    print("\nNext: Train your model using these landmark-annotated images!")
    print("="*60)

if __name__ == "__main__":
    try:
        from tqdm import tqdm
    except ImportError:
        print("Installing tqdm...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'tqdm'])
        from tqdm import tqdm

    main()
    detector.close()
