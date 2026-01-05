#!/usr/bin/env python3
"""
Calibrate skin tone detection for accurate hand segmentation.
This will sample your skin color and calculate optimal detection ranges.
"""

import cv2
import numpy as np
from pathlib import Path
import json

# Configuration
CONFIG_DIR = Path(__file__).parent.parent / "models"
CALIBRATION_FILE = CONFIG_DIR / "skin_calibration.json"

def sample_skin_color(frame, x, y, size=20):
    """Sample skin color from a small region"""
    # Extract region
    region = frame[y-size:y+size, x-size:x+size]

    if region.size == 0:
        return None

    # Calculate mean color in different color spaces
    mean_rgb = np.mean(region, axis=(0, 1))

    # Convert to HSV
    hsv_region = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
    mean_hsv = np.mean(hsv_region, axis=(0, 1))
    std_hsv = np.std(hsv_region, axis=(0, 1))

    # Convert to YCrCb
    ycrcb_region = cv2.cvtColor(region, cv2.COLOR_RGB2YCrCb)
    mean_ycrcb = np.mean(ycrcb_region, axis=(0, 1))
    std_ycrcb = np.std(ycrcb_region, axis=(0, 1))

    return {
        'rgb': mean_rgb.tolist(),
        'hsv': {'mean': mean_hsv.tolist(), 'std': std_hsv.tolist()},
        'ycrcb': {'mean': mean_ycrcb.tolist(), 'std': std_ycrcb.tolist()}
    }

def calculate_detection_ranges(samples):
    """Calculate optimal detection ranges from multiple samples"""
    # Collect all HSV and YCrCb values
    hsv_means = np.array([s['hsv']['mean'] for s in samples])
    ycrcb_means = np.array([s['ycrcb']['mean'] for s in samples])

    # Calculate overall mean and std
    hsv_mean = np.mean(hsv_means, axis=0)
    hsv_std = np.std(hsv_means, axis=0)

    ycrcb_mean = np.mean(ycrcb_means, axis=0)
    ycrcb_std = np.std(ycrcb_means, axis=0)

    # Create ranges (mean ± 2*std for good coverage)
    # For HSV
    hsv_lower = np.maximum([0, 30, 30], hsv_mean - 2.5 * hsv_std)
    hsv_upper = np.minimum([180, 255, 255], hsv_mean + 2.5 * hsv_std)

    # Special handling for Hue (circular)
    if hsv_lower[0] < 0:
        hsv_lower[0] = 0
    if hsv_upper[0] > 180:
        hsv_upper[0] = 180

    # For YCrCb
    ycrcb_lower = np.maximum([0, 0, 0], ycrcb_mean - 2.5 * ycrcb_std)
    ycrcb_upper = np.minimum([255, 255, 255], ycrcb_mean + 2.5 * ycrcb_std)

    return {
        'hsv_lower': hsv_lower.astype(int).tolist(),
        'hsv_upper': hsv_upper.astype(int).tolist(),
        'ycrcb_lower': ycrcb_lower.astype(int).tolist(),
        'ycrcb_upper': ycrcb_upper.astype(int).tolist()
    }

def test_mask_quality(frame, ranges):
    """Test how well the calibration detects the hand"""
    # HSV mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    hsv_lower = np.array(ranges['hsv_lower'], dtype=np.uint8)
    hsv_upper = np.array(ranges['hsv_upper'], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, hsv_lower, hsv_upper)

    # YCrCb mask
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
    ycrcb_lower = np.array(ranges['ycrcb_lower'], dtype=np.uint8)
    ycrcb_upper = np.array(ranges['ycrcb_upper'], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, ycrcb_lower, ycrcb_upper)

    # Combine masks
    mask = cv2.bitwise_or(mask_hsv, mask_ycrcb)

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask

def main():
    """Main calibration process"""
    print("\n" + "="*60)
    print("SKIN TONE CALIBRATION")
    print("="*60)
    print("\nInstructions:")
    print("1. Place your hand in the GREEN square")
    print("2. Try different hand positions (palm, fist, fingers)")
    print("3. Press SPACE to capture a sample (need 5-10 samples)")
    print("4. Press 'p' to preview the mask")
    print("5. Press 's' to save calibration")
    print("6. Press 'w' to quit without saving")
    print("\nMake sure to sample different parts of your hand!")
    print("="*60 + "\n")

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    samples = []
    calibration_ranges = None
    preview_mode = False

    # Sample region (center of frame)
    sample_x, sample_y = 320, 240
    sample_size = 30

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        display = frame.copy()

        if not preview_mode:
            # Draw sampling square
            cv2.rectangle(display,
                         (sample_x - sample_size, sample_y - sample_size),
                         (sample_x + sample_size, sample_y + sample_size),
                         (0, 255, 0), 2)

            # Draw instructions
            cv2.putText(display, "Place hand in GREEN square", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"Samples: {len(samples)}/10", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, "SPACE: capture  P: preview  S: save  W: quit", (10, 460),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        else:
            # Preview mode - show mask
            if calibration_ranges:
                mask = test_mask_quality(frame_rgb, calibration_ranges)

                # Create colored overlay
                mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                mask_colored[:, :, 0] = 0  # Remove blue
                mask_colored[:, :, 1] = mask  # Green channel

                # Blend with original
                display = cv2.addWeighted(display, 0.6, mask_colored, 0.4, 0)

                cv2.putText(display, "PREVIEW MODE - Green = detected hand", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display, "P: exit preview  S: save  W: quit", (10, 460),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Skin Tone Calibration', display)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('w'):
            print("\nCalibration cancelled")
            break

        elif key == ord(' ') and not preview_mode:
            # Capture sample
            sample = sample_skin_color(frame_rgb, sample_x, sample_y, sample_size)
            if sample:
                samples.append(sample)
                print(f"✓ Sample {len(samples)} captured")

                # Recalculate ranges
                if len(samples) >= 3:
                    calibration_ranges = calculate_detection_ranges(samples)
                    print(f"  Updated detection ranges")

                if len(samples) >= 5:
                    print("  → You have enough samples! Press 'p' to preview, 's' to save")

        elif key == ord('p'):
            # Toggle preview mode
            if len(samples) >= 3:
                preview_mode = not preview_mode
                if preview_mode:
                    print("\n→ Preview mode: Green overlay shows detected skin")
                else:
                    print("\n→ Back to sampling mode")
            else:
                print("Need at least 3 samples before preview")

        elif key == ord('s'):
            # Save calibration
            if len(samples) >= 3:
                CONFIG_DIR.mkdir(parents=True, exist_ok=True)

                calibration_data = {
                    'num_samples': len(samples),
                    'ranges': calibration_ranges,
                    'samples': samples
                }

                with open(CALIBRATION_FILE, 'w') as f:
                    json.dump(calibration_data, f, indent=2)

                print("\n" + "="*60)
                print("CALIBRATION SAVED!")
                print("="*60)
                print(f"Saved to: {CALIBRATION_FILE}")
                print(f"Samples captured: {len(samples)}")
                print("\nYour skin tone ranges:")
                print(f"  HSV Lower: {calibration_ranges['hsv_lower']}")
                print(f"  HSV Upper: {calibration_ranges['hsv_upper']}")
                print(f"  YCrCb Lower: {calibration_ranges['ycrcb_lower']}")
                print(f"  YCrCb Upper: {calibration_ranges['ycrcb_upper']}")
                print("\nNow run preprocess_remove_background.py")
                print("="*60)
                break
            else:
                print("Need at least 3 samples before saving!")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
