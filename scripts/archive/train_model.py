#!/usr/bin/env python3
"""
Train a CNN model to classify sign language letters.
This script loads the captured images, trains a model, and saves it for inference.
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
IMG_SIZE = 64  # Resize images to 64x64
BATCH_SIZE = 32
EPOCHS = 50
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
        print(f"Loading images for letter '{letter}'...")

        # Load all images in this letter's folder
        image_files = list(letter_folder.glob("*.jpg"))

        for img_path in image_files:
            # Read image
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
    print(f"Total labels: {len(set(labels))}")

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

    print(f"\nClasses found: {label_encoder.classes_}")
    print(f"Number of classes: {num_classes}")

    return categorical_labels, label_encoder

def create_model(num_classes):
    """Create a CNN model for image classification"""
    model = keras.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
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

def main():
    """Main training pipeline"""
    # Create models directory if it doesn't exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

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
    print("\nBuilding model...")
    model = create_model(num_classes)

    # Display model architecture
    model.summary()

    # Step 5: Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Step 6: Set up callbacks
    callbacks = [
        # Save best model
        keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Reduce learning rate when validation loss plateaus
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        # Early stopping to prevent overfitting
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # Step 7: Train model
    print(f"\nStarting training for {EPOCHS} epochs...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # Step 8: Evaluate model
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)

    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

    print(f"\nFinal Training Accuracy: {train_acc*100:.2f}%")
    print(f"Final Validation Accuracy: {val_acc*100:.2f}%")

    # Step 9: Save final model and metadata
    model.save(MODEL_DIR / 'sign_language_model.keras')
    print(f"\nModel saved to: {MODEL_DIR / 'sign_language_model.keras'}")

    # Save label encoder classes
    metadata = {
        'classes': label_encoder.classes_.tolist(),
        'img_size': IMG_SIZE,
        'num_classes': num_classes,
        'final_train_accuracy': float(train_acc),
        'final_val_accuracy': float(val_acc)
    }

    with open(MODEL_DIR / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {MODEL_DIR / 'model_metadata.json'}")

    # Step 10: Plot training history
    plot_training_history(history, MODEL_DIR / 'training_history.png')

    print("\n" + "="*50)
    print("All done! You can now use the model for real-time classification.")
    print("="*50)

if __name__ == "__main__":
    main()
