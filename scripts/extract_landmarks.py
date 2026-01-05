#!/usr/bin/env python3
"""
Extract MediaPipe hand landmark coordinates from training images.
Saves landmark data as features for direct ML training.
"""

import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

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
DATA_DIR = Path(__file__).parent.parent / "data" / "training"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "landmark_features"

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

def extract_landmarks(img):
    """
    Extract hand landmarks from image.
    Returns a flattened array of 63 features (21 landmarks × 3 coordinates: x, y, z).
    """
    # Convert numpy array to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

    # Detect hands
    results = detector.detect(mp_image)

    if results.hand_landmarks:
        landmarks = results.hand_landmarks[0]

        # Extract x, y, z coordinates for all 21 landmarks
        features = []
        for landmark in landmarks:
            features.extend([landmark.x, landmark.y, landmark.z])

        return np.array(features, dtype=np.float32), True

    return None, False

def flip_landmarks(features):
    """
    Horizontally flip hand landmarks to create mirror image.
    This makes the model work with both left and right hands!

    For each landmark, we flip the x-coordinate: new_x = 1 - x
    y and z coordinates stay the same.
    """
    flipped = features.copy()

    # Features are structured as: [x0, y0, z0, x1, y1, z1, ...]
    # We need to flip every x coordinate (indices 0, 3, 6, 9, ...)
    for i in range(0, len(flipped), 3):
        flipped[i] = 1.0 - flipped[i]  # Flip x coordinate

    return flipped

def process_dataset():
    """Process all training images and extract landmark features"""
    print("\n" + "="*60)
    print("EXTRACTING LANDMARK FEATURES FROM TRAINING DATA")
    print("With Hand Flipping Augmentation (Left & Right Hands)")
    print("="*60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_features = []
    all_labels = []
    stats = {'total': 0, 'detected': 0, 'failed': 0}

    # Process each letter folder
    for letter_folder in sorted(DATA_DIR.iterdir()):
        if not letter_folder.is_dir():
            continue

        letter = letter_folder.name
        print(f"\nProcessing letter '{letter}'...")

        image_files = list(letter_folder.glob("*.jpg"))

        if len(image_files) == 0:
            print(f"  No images found")
            continue

        detected_count = 0

        for img_path in tqdm(image_files, desc=f"  {letter}"):
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                stats['failed'] += 1
                continue

            # Convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Extract landmarks
            features, detected = extract_landmarks(img)

            if detected:
                # Add original features
                all_features.append(features)
                all_labels.append(letter)

                # Add horizontally flipped version (for hand-agnostic training)
                flipped_features = flip_landmarks(features)
                all_features.append(flipped_features)
                all_labels.append(letter)

                detected_count += 1
                stats['detected'] += 1
            else:
                stats['failed'] += 1

            stats['total'] += 1

        print(f"  ✓ {detected_count}/{len(image_files)} images processed")

    # Convert to numpy arrays
    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels)

    # Save the dataset
    print("\n" + "="*60)
    print("SAVING LANDMARK DATASET")
    print("="*60)

    np.save(OUTPUT_DIR / "features.npy", X)
    np.save(OUTPUT_DIR / "labels.npy", y)

    # Save metadata
    metadata = {
        'num_samples': len(X),
        'num_features': X.shape[1],
        'num_classes': len(set(y)),
        'classes': sorted(list(set(y))),
        'feature_description': '21 landmarks × 3 coordinates (x, y, z) = 63 features',
        'augmentation': 'Horizontal flipping (2x data: original + mirrored)',
        'hand_agnostic': True,
        'detection_stats': stats
    }

    with open(OUTPUT_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nFeatures saved: {OUTPUT_DIR / 'features.npy'}")
    print(f"Labels saved: {OUTPUT_DIR / 'labels.npy'}")
    print(f"Metadata saved: {OUTPUT_DIR / 'metadata.json'}")

    print("\n" + "="*60)
    print("EXTRACTION COMPLETE!")
    print("="*60)
    print(f"\nOriginal images processed: {stats['detected']}")
    print(f"Total samples (with augmentation): {len(X)} (2x with flipping)")
    print(f"Feature shape: {X.shape}")
    print(f"Classes: {len(set(y))}")
    print(f"Successfully detected: {stats['detected']}/{stats['total']} ({stats['detected']/stats['total']*100:.1f}%)")
    print(f"Failed detections: {stats['failed']}")
    print(f"\n✓ Hand Flipping Augmentation: ENABLED")
    print(f"  Each hand sample has been mirrored to work with both hands!")

    if stats['detected'] > stats['total'] * 0.9:
        print("\n✓ Excellent detection rate!")
        print("  Ready to train landmark-based model!")
    elif stats['detected'] > stats['total'] * 0.7:
        print("\n✓ Good detection rate!")
    else:
        print("\n⚠ Detection rate could be better")
        print("  Consider checking image quality")

    print("="*60)

if __name__ == "__main__":
    try:
        from tqdm import tqdm
    except ImportError:
        print("Installing tqdm...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'tqdm'])
        from tqdm import tqdm

    process_dataset()
    detector.close()
