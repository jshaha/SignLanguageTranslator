#!/usr/bin/env python3
"""
Analyze training data to identify potential issues.
Shows sample images, class distribution, and similarity between classes.
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data" / "training"

def analyze_class_distribution():
    """Count images per class"""
    print("="*60)
    print("CLASS DISTRIBUTION")
    print("="*60)

    class_counts = {}

    for letter_folder in sorted(DATA_DIR.iterdir()):
        if not letter_folder.is_dir():
            continue

        letter = letter_folder.name
        count = len(list(letter_folder.glob("*.jpg")))
        class_counts[letter] = count

        status = "✓" if count >= 80 else "⚠"
        print(f"{status} {letter}: {count} images")

    print(f"\nTotal classes: {len(class_counts)}")
    print(f"Total images: {sum(class_counts.values())}")

    # Check for imbalance
    if class_counts:
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else 0

        if imbalance_ratio > 1.5:
            print(f"\n⚠ WARNING: Class imbalance detected (ratio: {imbalance_ratio:.2f})")
            print(f"   Min: {min_count}, Max: {max_count}")
        else:
            print(f"\n✓ Class distribution is balanced")

    return class_counts

def visualize_samples(letters=['T', 'U', 'V'], num_samples=5):
    """Visualize sample images from specific letters"""
    print("\n" + "="*60)
    print(f"VISUALIZING SAMPLES FOR: {', '.join(letters)}")
    print("="*60)

    fig, axes = plt.subplots(len(letters), num_samples, figsize=(15, len(letters)*3))

    if len(letters) == 1:
        axes = [axes]

    for i, letter in enumerate(letters):
        letter_dir = DATA_DIR / letter

        if not letter_dir.exists():
            print(f"Warning: No data found for letter '{letter}'")
            continue

        image_files = list(letter_dir.glob("*.jpg"))[:num_samples]

        for j, img_path in enumerate(image_files):
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if len(letters) > 1:
                axes[i][j].imshow(img)
                axes[i][j].axis('off')
                if j == 0:
                    axes[i][j].set_title(f"Letter: {letter}", fontsize=14, fontweight='bold')
            else:
                axes[j].imshow(img)
                axes[j].axis('off')
                if j == 0:
                    axes[j].set_title(f"Letter: {letter}", fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_path = Path(__file__).parent.parent / "models" / "data_samples.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSample visualization saved to: {save_path}")
    plt.close()

def analyze_image_variance(letters=['T', 'U', 'V']):
    """Analyze variance within each class to see if images are too similar"""
    print("\n" + "="*60)
    print("IMAGE VARIANCE ANALYSIS")
    print("="*60)

    for letter in letters:
        letter_dir = DATA_DIR / letter

        if not letter_dir.exists():
            continue

        image_files = list(letter_dir.glob("*.jpg"))

        if len(image_files) == 0:
            continue

        # Load and resize images
        images = []
        for img_path in image_files[:50]:  # Sample first 50 images
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            images.append(img.flatten())

        images = np.array(images, dtype=np.float32)

        # Calculate variance
        mean_img = np.mean(images, axis=0)
        variance = np.var(images, axis=0)
        avg_variance = np.mean(variance)

        # Calculate standard deviation across all images
        overall_std = np.std(images)

        status = "✓" if avg_variance > 500 else "⚠"
        print(f"{status} Letter {letter}:")
        print(f"   Average pixel variance: {avg_variance:.2f}")
        print(f"   Overall std dev: {overall_std:.2f}")

        if avg_variance < 500:
            print(f"   ⚠ Low variance - images might be too similar!")

    print("\nNote: Higher variance = more diverse images (better for training)")

def compute_similarity_matrix(letters=['T', 'U', 'V']):
    """Compute average similarity between different letter classes"""
    print("\n" + "="*60)
    print("INTER-CLASS SIMILARITY MATRIX")
    print("="*60)
    print("(Lower values = more distinct classes)")
    print()

    # Load average images for each letter
    avg_images = {}

    for letter in letters:
        letter_dir = DATA_DIR / letter

        if not letter_dir.exists():
            continue

        image_files = list(letter_dir.glob("*.jpg"))[:30]

        images = []
        for img_path in image_files:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            images.append(img.astype(np.float32) / 255.0)

        if images:
            avg_images[letter] = np.mean(images, axis=0)

    # Compute similarity matrix
    print("     ", end="")
    for letter in letters:
        print(f"{letter:>8}", end="")
    print()

    for letter1 in letters:
        print(f"{letter1:>4} ", end="")

        if letter1 not in avg_images:
            print()
            continue

        for letter2 in letters:
            if letter2 not in avg_images:
                print(f"{'N/A':>8}", end="")
                continue

            # Compute mean squared error (MSE) as similarity metric
            mse = np.mean((avg_images[letter1] - avg_images[letter2])**2)

            # Also compute structural similarity
            similarity = np.corrcoef(avg_images[letter1].flatten(),
                                    avg_images[letter2].flatten())[0, 1]

            if letter1 == letter2:
                print(f"{'--':>8}", end="")
            else:
                # Show correlation coefficient (higher = more similar, bad!)
                print(f"{similarity:>8.3f}", end="")
        print()

    print("\nNote: Values close to 1.0 indicate very similar classes (problematic!)")
    print("      Values close to 0.0 indicate distinct classes (good!)")

def provide_recommendations(class_counts):
    """Provide specific recommendations based on analysis"""
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    print("\n1. DATA COLLECTION:")
    print("   - Collect more images with varied conditions:")
    print("     • Different lighting (bright, dim, side-lit)")
    print("     • Different backgrounds")
    print("     • Different hand angles (rotate wrist slightly)")
    print("     • Different distances from camera")
    print("     • Multiple sessions on different days")

    print("\n2. FOR T, U, V specifically:")
    print("   - T: Ensure thumb is clearly visible between fingers")
    print("   - U: Keep fingers TOGETHER and straight")
    print("   - V: Spread fingers WIDE apart (exaggerate the gap)")
    print("   - Try rotating hand slightly left/right during capture")

    print("\n3. TRAINING IMPROVEMENTS:")
    print("   - Add data augmentation (rotation, brightness, etc.)")
    print("   - Collect 200+ images per letter instead of 100")
    print("   - Consider using a larger model architecture")
    print("   - Train for more epochs with early stopping")

    print("\n4. TESTING:")
    print("   - Test with different lighting than training")
    print("   - Test with hand at different angles")
    print("   - If still confused, collect 50+ more varied images for problem letters")

def main():
    """Run all analyses"""
    print("\n" + "="*60)
    print("SIGN LANGUAGE TRAINING DATA ANALYSIS")
    print("="*60)

    # 1. Class distribution
    class_counts = analyze_class_distribution()

    # 2. Variance analysis
    analyze_image_variance(['T', 'U', 'V'])

    # 3. Similarity matrix
    compute_similarity_matrix(['T', 'U', 'V'])

    # 4. Visualize samples
    available_letters = [l for l in ['T', 'U', 'V'] if (DATA_DIR / l).exists()]
    if available_letters:
        visualize_samples(available_letters, num_samples=5)

    # 5. Recommendations
    provide_recommendations(class_counts)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
