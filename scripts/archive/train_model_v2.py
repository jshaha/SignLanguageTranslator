#!/usr/bin/env python3
"""
IMPROVED Training script with data augmentation to prevent overfitting.
This version adds random transformations to make the model more robust.
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
EPOCHS = 100  # More epochs with early stopping
VALIDATION_SPLIT = 0.2

def load_dataset():
    """Load all images and labels from the training data folder"""
    print("Loading dataset...")

    images = []
    labels = []

    # Iterate through each letter folder
    for letter_folder in sorted(DATA_DIR.iterdir()):
        if not letter_folder.is_dir():
            continue

        letter = letter_folder.name
        image_files = list(letter_folder.glob("*.jpg"))

        print(f"Loading {len(image_files)} images for letter '{letter}'...")

        for img_path in image_files:
            img = cv2.imread(str(img_path))

            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize to standard size
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            images.append(img)
            labels.append(letter)

    print(f"\nTotal images loaded: {len(images)}")
    print(f"Total classes: {len(set(labels))}")

    # Convert to numpy arrays
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)

    # Normalize pixel values to [0, 1]
    images = images / 255.0

    return images, labels

def prepare_labels(labels):
    """Encode string labels to integers"""
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Convert to categorical (one-hot encoding)
    num_classes = len(label_encoder.classes_)
    categorical_labels = keras.utils.to_categorical(encoded_labels, num_classes)

    print(f"\nClasses found: {list(label_encoder.classes_)}")
    print(f"Number of classes: {num_classes}")

    return categorical_labels, label_encoder

def create_data_augmentation():
    """Create aggressive data augmentation to prevent overfitting"""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),  # Rotate up to 15%
        layers.RandomZoom(0.15),      # Zoom in/out up to 15%
        layers.RandomTranslation(0.1, 0.1),  # Shift up to 10%
        layers.RandomBrightness(0.2), # Change brightness
        layers.RandomContrast(0.2),   # Change contrast
    ])

def create_model(num_classes):
    """Create improved CNN model with data augmentation"""

    # Data augmentation (only applied during training)
    data_augmentation = create_data_augmentation()

    model = keras.Sequential([
        # Data augmentation layers (only active during training)
        data_augmentation,

        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Fourth convolutional block (deeper network)
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),

        # Output layer
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
    print("IMPROVED SIGN LANGUAGE MODEL TRAINING v2")
    print("With Data Augmentation to Prevent Overfitting")
    print("="*60)

    # Step 1: Load dataset
    images, labels = load_dataset()

    if len(images) == 0:
        print("Error: No images found! Please run capture_training_data.py first.")
        return

    # Step 2: Prepare labels
    categorical_labels, label_encoder = prepare_labels(labels)
    num_classes = len(label_encoder.classes_)

    # Step 3: Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        images, categorical_labels,
        test_size=VALIDATION_SPLIT,
        random_state=42,
        stratify=labels
    )

    print(f"\nTraining set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")

    # Step 4: Create model
    print("\nBuilding improved model with data augmentation...")
    model = create_model(num_classes)

    # Display model architecture
    model.summary()

    # Step 5: Compile model with label smoothing to prevent overconfidence
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),  # Lower learning rate
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),  # Label smoothing
        metrics=['accuracy']
    )

    # Step 6: Set up callbacks
    callbacks = [
        # Save best model
        keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / 'best_model_v2.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Reduce learning rate when validation loss plateaus
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        # Early stopping to prevent overfitting
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # Step 7: Train model
    print(f"\nStarting training for up to {EPOCHS} epochs...")
    print("(Will stop early if validation loss stops improving)\n")

    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # Step 8: Evaluate model
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)

    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

    print(f"\nFinal Training Accuracy: {train_acc*100:.2f}%")
    print(f"Final Validation Accuracy: {val_acc*100:.2f}%")

    gap = train_acc - val_acc
    if gap > 0.1:
        print(f"\n⚠ Warning: Large accuracy gap ({gap*100:.1f}%) suggests some overfitting")
        print("   Consider collecting more diverse training data")
    else:
        print(f"\n✓ Good! Accuracy gap is small ({gap*100:.1f}%)")

    # Step 9: Save final model and metadata
    model.save(MODEL_DIR / 'sign_language_model_v2.keras')
    print(f"\nModel saved to: {MODEL_DIR / 'sign_language_model_v2.keras'}")

    # Save label encoder classes
    metadata = {
        'classes': label_encoder.classes_.tolist(),
        'img_size': IMG_SIZE,
        'num_classes': num_classes,
        'final_train_accuracy': float(train_acc),
        'final_val_accuracy': float(val_acc),
        'version': 2,
        'data_augmentation': True
    }

    with open(MODEL_DIR / 'model_metadata_v2.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {MODEL_DIR / 'model_metadata_v2.json'}")

    # Step 10: Plot training history
    plot_training_history(history, MODEL_DIR / 'training_history_v2.png')

    print("\n" + "="*60)
    print("MODEL READY FOR TESTING!")
    print("Run real_time_classifier_v2.py to test the new model")
    print("="*60)

if __name__ == "__main__":
    main()
