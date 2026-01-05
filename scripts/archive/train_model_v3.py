#!/usr/bin/env python3
"""
BALANCED Training script - middle ground between overfitting and underfitting.
Moderate augmentation, reasonable regularization.
"""

import os
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import json

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data" / "training"
MODEL_DIR = Path(__file__).parent.parent / "models"
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.2

def load_dataset():
    """Load all images and labels from the training data folder"""
    print("Loading dataset...")

    images = []
    labels = []

    for letter_folder in sorted(DATA_DIR.iterdir()):
        if not letter_folder.is_dir():
            continue

        letter = letter_folder.name
        image_files = list(letter_folder.glob("*.jpg"))

        print(f"Loading {len(image_files)} images for letter '{letter}'...")

        for img_path in image_files:
            img = cv2.imread(str(img_path))

            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            images.append(img)
            labels.append(letter)

    print(f"\nTotal images loaded: {len(images)}")
    print(f"Total classes: {len(set(labels))}")

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)

    # Normalize pixel values to [0, 1]
    images = images / 255.0

    return images, labels

def prepare_labels(labels):
    """Encode string labels to integers"""
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    num_classes = len(label_encoder.classes_)
    categorical_labels = keras.utils.to_categorical(encoded_labels, num_classes)

    print(f"\nClasses found: {list(label_encoder.classes_)}")
    print(f"Number of classes: {num_classes}")

    return categorical_labels, label_encoder

def create_data_augmentation():
    """Create MODERATE data augmentation - not too aggressive"""
    return keras.Sequential([
        layers.RandomRotation(0.05),  # Only ±5% rotation (was 15%)
        layers.RandomZoom(0.1),       # Only 10% zoom (was 15%)
        layers.RandomTranslation(0.05, 0.05),  # Only 5% shift (was 10%)
        layers.RandomBrightness(0.1), # Only 10% brightness (was 20%)
        layers.RandomContrast(0.1),   # Only 10% contrast (was 20%)
    ])

def create_model(num_classes):
    """Create balanced CNN model"""

    data_augmentation = create_data_augmentation()

    model = keras.Sequential([
        # Moderate data augmentation
        data_augmentation,

        # First conv block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),  # Reduced from 0.3

        # Second conv block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),  # Reduced from 0.3

        # Third conv block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),  # Reduced from 0.4

        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),  # Reduced from 512
        layers.BatchNormalization(),
        layers.Dropout(0.4),  # Reduced from 0.5
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),  # Reduced from 0.5

        # Output
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

def plot_training_history(history, save_path):
    """Plot and save training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nTraining plots saved to: {save_path}")
    plt.close()

def main():
    """Main training pipeline"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("BALANCED SIGN LANGUAGE MODEL TRAINING v3")
    print("Moderate augmentation + regularization")
    print("="*60)

    # Load dataset
    images, labels = load_dataset()

    if len(images) == 0:
        print("Error: No images found!")
        return

    # Prepare labels
    categorical_labels, label_encoder = prepare_labels(labels)
    num_classes = len(label_encoder.classes_)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        images, categorical_labels,
        test_size=VALIDATION_SPLIT,
        random_state=42,
        stratify=labels
    )

    print(f"\nTraining set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")

    # Create model
    print("\nBuilding model...")
    model = create_model(num_classes)
    model.summary()

    # Compile with normal settings (no label smoothing)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Higher LR
        loss='categorical_crossentropy',  # No label smoothing
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / 'best_model_v3.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # Train
    print(f"\nStarting training for up to {EPOCHS} epochs...\n")

    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)

    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

    print(f"\nFinal Training Accuracy: {train_acc*100:.2f}%")
    print(f"Final Validation Accuracy: {val_acc*100:.2f}%")

    gap = train_acc - val_acc
    print(f"Accuracy Gap: {gap*100:.1f}%")

    if val_acc < 0.5:
        print("\n⚠ Warning: Validation accuracy is low (<50%)")
        print("   Your training data might be too similar.")
        print("   Try collecting images with:")
        print("   - Different lighting conditions")
        print("   - Different backgrounds")
        print("   - Different hand angles")
    elif gap > 0.15:
        print(f"\n⚠ Some overfitting detected (gap: {gap*100:.1f}%)")
        print("   Model should still work, but could be better with more diverse data")
    else:
        print(f"\n✓ Good! Model is learning well")

    # Save model
    model.save(MODEL_DIR / 'sign_language_model_v3.keras')
    print(f"\nModel saved to: {MODEL_DIR / 'sign_language_model_v3.keras'}")

    # Save metadata
    metadata = {
        'classes': label_encoder.classes_.tolist(),
        'img_size': IMG_SIZE,
        'num_classes': num_classes,
        'final_train_accuracy': float(train_acc),
        'final_val_accuracy': float(val_acc),
        'version': 3,
        'data_augmentation': 'moderate'
    }

    with open(MODEL_DIR / 'model_metadata_v3.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {MODEL_DIR / 'model_metadata_v3.json'}")

    # Plot history
    plot_training_history(history, MODEL_DIR / 'training_history_v3.png')

    print("\n" + "="*60)
    print("Next: Update real_time_classifier.py to use v3 model")
    print("="*60)

if __name__ == "__main__":
    main()
