#!/usr/bin/env python3
"""
Simple script to capture and label sign language photos for training data.
Press the letter key (A-Z) to capture images for that letter.
Press 'q' to quit.
"""

import cv2
import os
import time
from pathlib import Path

# Configuration
IMAGES_PER_LETTER = 100  # Number of images to capture per letter
CAPTURE_DELAY = 0.1  # Delay between captures in seconds
DATA_DIR = Path(__file__).parent.parent / "data" / "training"

def setup_data_folders():
    """Create folders for each letter A-Z"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        (DATA_DIR / letter).mkdir(exist_ok=True)
    print(f"Data folders created at: {DATA_DIR}")

def capture_images():
    """Capture images from webcam and save them to labeled folders"""
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    current_letter = None
    capture_count = 0

    print("\n=== Sign Language Data Capture ===")
    print("Instructions:")
    print("- Press A-Z to start capturing images for that letter")
    print("- Press the same letter again to stop capturing")
    print("- Press 'q' to quit")
    print("\nReady to capture!")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Display info on frame
        display_frame = frame.copy()
        if current_letter:
            cv2.putText(display_frame, f"Capturing: {current_letter}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Count: {capture_count}/{IMAGES_PER_LETTER}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Press A-Z to start capturing", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw ROI rectangle where hand should be placed
        cv2.rectangle(display_frame, (100, 100), (540, 380), (255, 0, 0), 2)
        cv2.putText(display_frame, "Place hand here", (200, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow('Sign Language Data Capture', display_frame)

        # Capture images if a letter is selected
        if current_letter and capture_count < IMAGES_PER_LETTER:
            # Crop to ROI
            roi = frame[100:380, 100:540]

            # Save image
            timestamp = int(time.time() * 1000)
            filename = DATA_DIR / current_letter / f"{current_letter}_{timestamp}_{capture_count}.jpg"
            cv2.imwrite(str(filename), roi)

            capture_count += 1

            if capture_count >= IMAGES_PER_LETTER:
                print(f"\nCompleted capturing {IMAGES_PER_LETTER} images for letter '{current_letter}'")
                current_letter = None
                capture_count = 0

            time.sleep(CAPTURE_DELAY)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('w'):
            break
        elif key >= ord('a') and key <= ord('z'):
            letter = chr(key).upper()
            if current_letter == letter:
                # Stop capturing if same letter pressed again
                print(f"\nStopped capturing for letter '{letter}'")
                current_letter = None
                capture_count = 0
            else:
                # Start capturing new letter
                current_letter = letter
                capture_count = 0
                print(f"\nStarting capture for letter '{letter}'...")
        elif key >= ord('A') and key <= ord('Z'):
            letter = chr(key)
            if current_letter == letter:
                print(f"\nStopped capturing for letter '{letter}'")
                current_letter = None
                capture_count = 0
            else:
                current_letter = letter
                capture_count = 0
                print(f"\nStarting capture for letter '{letter}'...")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Display summary
    print("\n=== Capture Summary ===")
    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        letter_dir = DATA_DIR / letter
        if letter_dir.exists():
            count = len(list(letter_dir.glob("*.jpg")))
            if count > 0:
                print(f"{letter}: {count} images")

if __name__ == "__main__":
    setup_data_folders()
    capture_images()
