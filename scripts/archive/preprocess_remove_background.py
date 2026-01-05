#!/usr/bin/env python3
"""
Preprocess images by removing backgrounds and adding random backgrounds.
This prevents the model from memorizing your room/background.
"""

import cv2
import numpy as np
from pathlib import Path
import random
import json
from tqdm import tqdm

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data" / "training"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"

# Background colors to use (diverse set)
SOLID_BACKGROUNDS = [
    (255, 255, 255),  # White
    (0, 0, 0),        # Black
    (128, 128, 128),  # Gray
    (200, 200, 200),  # Light gray
    (50, 50, 50),     # Dark gray
    (240, 240, 220),  # Beige
    (220, 240, 240),  # Light blue
    (240, 220, 240),  # Light purple
    (240, 240, 220),  # Cream
    (180, 200, 180),  # Light green
]

def load_calibration():
    """Load calibrated skin tone ranges"""
    calibration_file = Path(__file__).parent.parent / "models" / "skin_calibration.json"

    if not calibration_file.exists():
        print("\n⚠ WARNING: No skin calibration found!")
        print("Using default skin detection ranges.")
        print("For better results, run: python3 calibrate_skin_tone.py")
        print()
        return None

    with open(calibration_file, 'r') as f:
        data = json.load(f)

    print(f"✓ Loaded calibration with {data['num_samples']} samples")
    return data['ranges']

def remove_background_skin_detection(img, calibration=None):
    """
    Remove background using skin color detection.
    Uses calibrated ranges if available, otherwise defaults.
    """
    if calibration:
        # Use calibrated ranges (better!)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv_lower = np.array(calibration['hsv_lower'], dtype=np.uint8)
        hsv_upper = np.array(calibration['hsv_upper'], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, hsv_lower, hsv_upper)

        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        ycrcb_lower = np.array(calibration['ycrcb_lower'], dtype=np.uint8)
        ycrcb_upper = np.array(calibration['ycrcb_upper'], dtype=np.uint8)
        mask_ycrcb = cv2.inRange(ycrcb, ycrcb_lower, ycrcb_upper)

        # Combine both masks
        mask = cv2.bitwise_or(mask_hsv, mask_ycrcb)
    else:
        # Default skin detection (less accurate)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

    # Clean up mask with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Dilate slightly to include hand edges
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Gaussian blur to smooth edges
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    return mask

def add_random_background(img, mask):
    """Add a random solid background to the image"""
    # Choose random background color
    bg_color = random.choice(SOLID_BACKGROUNDS)

    # Create background
    background = np.full_like(img, bg_color, dtype=np.uint8)

    # Normalize mask to 0-1 range
    mask_normalized = mask.astype(np.float32) / 255.0
    mask_3channel = np.stack([mask_normalized] * 3, axis=-1)

    # Blend: foreground where mask is white, background where mask is black
    result = (img * mask_3channel + background * (1 - mask_3channel)).astype(np.uint8)

    return result

def add_gradient_background(img, mask):
    """Add a gradient background for more variety"""
    h, w = img.shape[:2]

    # Random gradient direction
    if random.random() < 0.5:
        # Vertical gradient (top to bottom)
        gradient = np.linspace(random.randint(180, 255), random.randint(50, 120), h)
        # Create 2D gradient: each row has the same value
        gradient_2d = np.tile(gradient[:, np.newaxis], (1, w))
    else:
        # Horizontal gradient (left to right)
        gradient = np.linspace(random.randint(180, 255), random.randint(50, 120), w)
        # Create 2D gradient: each column has the same value
        gradient_2d = np.tile(gradient[np.newaxis, :], (h, 1))

    # Convert to 3-channel (RGB)
    background = np.stack([gradient_2d, gradient_2d, gradient_2d], axis=-1).astype(np.uint8)

    # Blend
    mask_normalized = mask.astype(np.float32) / 255.0
    mask_3channel = np.stack([mask_normalized] * 3, axis=-1)

    result = (img * mask_3channel + background * (1 - mask_3channel)).astype(np.uint8)

    return result

def process_image(img_path, letter, output_folder, num_augmented=3, calibration=None):
    """Process a single image: remove background and create augmented versions"""
    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        return 0

    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Remove background
    mask = remove_background_skin_detection(img, calibration)

    # Check if mask is reasonable (found some skin)
    if np.sum(mask) < 1000:  # Too little skin detected
        print(f"Warning: Poor skin detection for {img_path.name}, using original")
        # Save original without processing
        output_path = output_folder / f"orig_{img_path.name}"
        cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return 1

    saved_count = 0

    # Save version with different backgrounds
    for i in range(num_augmented):
        if i % 2 == 0:
            # Solid background
            result = add_random_background(img, mask)
        else:
            # Gradient background
            result = add_gradient_background(img, mask)

        # Save
        output_path = output_folder / f"bg{i}_{img_path.name}"
        cv2.imwrite(str(output_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        saved_count += 1

    return saved_count

def main():
    """Process all images"""
    print("\n" + "="*60)
    print("BACKGROUND REMOVAL & REPLACEMENT")
    print("="*60)

    # Load calibration
    calibration = load_calibration()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_processed = 0
    total_created = 0

    # Process each letter folder
    for letter_folder in sorted(DATA_DIR.iterdir()):
        if not letter_folder.is_dir():
            continue

        letter = letter_folder.name
        print(f"\nProcessing letter '{letter}'...")

        # Create output folder for this letter
        output_folder = OUTPUT_DIR / letter
        output_folder.mkdir(parents=True, exist_ok=True)

        # Get all images
        image_files = list(letter_folder.glob("*.jpg"))

        if len(image_files) == 0:
            print(f"  No images found for '{letter}'")
            continue

        # Process each image
        for img_path in tqdm(image_files, desc=f"  {letter}"):
            count = process_image(img_path, letter, output_folder, num_augmented=3, calibration=calibration)
            total_created += count
            if count > 0:
                total_processed += 1

    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"\nOriginal images processed: {total_processed}")
    print(f"Augmented images created: {total_created}")
    print(f"\nProcessed images saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("1. Check the processed images to make sure they look good")
    print("2. Update train_model.py to use the 'processed' folder instead of 'training'")
    print("3. Train your model!")

if __name__ == "__main__":
    try:
        from tqdm import tqdm
    except ImportError:
        print("Installing tqdm for progress bars...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'tqdm'])
        from tqdm import tqdm

    main()
