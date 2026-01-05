#!/usr/bin/env python3
"""
Preview how background removal works on your images.
Shows original, mask, and result side-by-side to debug issues.
"""

import cv2
import numpy as np
from pathlib import Path
import json
import random

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data" / "training"
CONFIG_DIR = Path(__file__).parent.parent / "models"
CALIBRATION_FILE = CONFIG_DIR / "skin_calibration.json"

def load_calibration():
    """Load calibrated skin tone ranges"""
    if not CALIBRATION_FILE.exists():
        print("No calibration found!")
        return None

    with open(CALIBRATION_FILE, 'r') as f:
        data = json.load(f)

    print(f"✓ Loaded calibration with {data['num_samples']} samples")
    print(f"  HSV range: {data['ranges']['hsv_lower']} to {data['ranges']['hsv_upper']}")
    print(f"  YCrCb range: {data['ranges']['ycrcb_lower']} to {data['ranges']['ycrcb_upper']}")
    return data['ranges']

def remove_background_skin_detection(img, calibration=None):
    """Remove background using skin color detection"""
    if calibration:
        # Use calibrated ranges
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
        # Default skin detection
        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    return mask

def add_random_background(img, mask):
    """Add random background"""
    bg_colors = [
        (255, 255, 255), (0, 0, 0), (128, 128, 128),
        (200, 200, 200), (240, 240, 220)
    ]
    bg_color = random.choice(bg_colors)
    background = np.full_like(img, bg_color, dtype=np.uint8)

    mask_normalized = mask.astype(np.float32) / 255.0
    mask_3channel = np.stack([mask_normalized] * 3, axis=-1)

    result = (img * mask_3channel + background * (1 - mask_3channel)).astype(np.uint8)
    return result

def preview_images(letter='A', num_samples=5):
    """Preview background removal on sample images"""
    print("\n" + "="*60)
    print(f"PREVIEWING BACKGROUND REMOVAL FOR LETTER '{letter}'")
    print("="*60)

    # Load calibration
    calibration = load_calibration()

    # Get sample images
    letter_dir = DATA_DIR / letter
    if not letter_dir.exists():
        print(f"Error: No folder found for letter '{letter}'")
        return

    image_files = list(letter_dir.glob("*.jpg"))[:num_samples]

    if len(image_files) == 0:
        print(f"No images found for '{letter}'")
        return

    print(f"\nShowing {len(image_files)} sample images")
    print("\nControls:")
    print("  SPACE: Next image")
    print("  R: Recalibrate (if detection is bad)")
    print("  W: Quit")
    print()

    for idx, img_path in enumerate(image_files):
        print(f"\nImage {idx+1}/{len(image_files)}: {img_path.name}")

        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Remove background
        mask = remove_background_skin_detection(img, calibration)

        # Add random background
        result = add_random_background(img, mask)

        # Calculate detection quality
        mask_coverage = (np.sum(mask > 128) / mask.size) * 100
        print(f"  Detected skin: {mask_coverage:.1f}% of image")

        if mask_coverage < 5:
            print("  ⚠ WARNING: Very little skin detected!")
        elif mask_coverage > 60:
            print("  ⚠ WARNING: Too much detected - might include background!")
        else:
            print("  ✓ Detection looks reasonable")

        # Create visualization
        h, w = img.shape[:2]

        # Create mask visualization (colorized)
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        # Resize for display
        display_h = 400
        display_w = int(w * display_h / h)

        img_display = cv2.resize(img, (display_w, display_h))
        mask_display = cv2.resize(mask_colored, (display_w, display_h))
        result_display = cv2.resize(result, (display_w, display_h))

        # Stack horizontally
        combined = np.hstack([img_display, mask_display, result_display])

        # Add labels
        cv2.putText(combined, "Original", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(combined, "Mask (white=detected)", (display_w + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(combined, "Result", (2*display_w + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Convert back to BGR for display
        combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

        cv2.imshow('Background Removal Preview', combined)

        # Wait for key
        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord(' '):
                # Next image
                break
            elif key == ord('w'):
                # Quit
                cv2.destroyAllWindows()
                return
            elif key == ord('r'):
                print("\n→ Please run: python3 calibrate_skin_tone.py")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("\n" + "="*60)
    print("Preview complete!")
    print("\nIf detection was poor:")
    print("  1. Run: python3 calibrate_skin_tone.py")
    print("  2. Get 10+ samples from different hand angles")
    print("  3. Check preview mode (press P) to verify")
    print("  4. Save with S")
    print("="*60)

def main():
    """Main preview"""
    import sys

    # Allow letter selection
    letter = 'A'
    if len(sys.argv) > 1:
        letter = sys.argv[1].upper()

    preview_images(letter, num_samples=10)

if __name__ == "__main__":
    main()
