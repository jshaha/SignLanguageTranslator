#!/usr/bin/env python3
"""
Train model using landmark-annotated processed images.
Should work much better since model learns hand pose structure.
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

# Configuration - USING LANDMARK ANNOTATED DATA
DATA_DIR = Path(__file__).parent.parent / "data" / "processed_landmarks"
MODEL_DIR = Path(__file__).parent.parent / "models"
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 40
VALIDATION_SPLIT = 0.2

def load_dataset():
    """Load processed images (with landmarks annotated)"""
    print("Loading landmark-annotated dataset...")

    images = []
    labels = []

    for letter_folder in sorted(DATA_DIR.iterdir()):
        if not letter_folder.is_dir():
            continue

        letter = letter_folder.name
        image_files = list(letter_folder.glob("*.jpg"))

        print(f"Loading {len(image_files)} images for '{letter}'...")

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
    images = images / 255.0

    return images, labels

def prepare_labels(labels):
    """Encode labels"""
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    categorical_labels = keras.utils.to_categorical(encoded_labels, num_classes)

    print(f"\nClasses: {list(label_encoder.classes_)}")
    print(f"Number of classes: {num_classes}")

    return categorical_labels, label_encoder

def create_model(num_classes):
    """Simple CNN - no augmentation needed since data is already augmented"""

    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Dense
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        # Output
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

def plot_training_history(history, save_path):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nPlots saved: {save_path}")
    plt.close()

def main():
    """Main training"""
    if not DATA_DIR.exists():
        print(f"ERROR: Landmark-annotated data folder not found at {DATA_DIR}")
        print("Please run preprocess_mediapipe.py first!")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("TRAINING WITH LANDMARK-ANNOTATED IMAGES")
    print("="*60)

    # Load processed data
    images, labels = load_dataset()

    if len(images) == 0:
        print("Error: No landmark-annotated images found!")
        print("Run preprocess_mediapipe.py first")
        return

    # Prepare labels
    categorical_labels, label_encoder = prepare_labels(labels)
    num_classes = len(label_encoder.classes_)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        images, categorical_labels,
        test_size=VALIDATION_SPLIT,
        random_state=42,
        stratify=labels
    )

    print(f"\nTraining: {len(X_train)} images")
    print(f"Validation: {len(X_val)} images")

    # Create model
    print("\nBuilding model...")
    model = create_model(num_classes)
    model.summary()

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / 'best_model_processed.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # Train
    print(f"\nTraining for up to {EPOCHS} epochs...\n")

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

    print(f"\nTraining Accuracy: {train_acc*100:.2f}%")
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Gap: {(train_acc - val_acc)*100:.1f}%")

    if val_acc > 0.85:
        print("\n✓ Excellent! Landmark annotations worked!")
    elif val_acc > 0.70:
        print("\n✓ Good! Model should work well")
    elif val_acc > 0.50:
        print("\n~ Decent, but might need improvement")
    else:
        print("\n⚠ Low accuracy - check landmark images")

    # Save
    model.save(MODEL_DIR / 'sign_language_model_processed.keras')
    print(f"\nModel saved: {MODEL_DIR / 'sign_language_model_processed.keras'}")

    metadata = {
        'classes': label_encoder.classes_.tolist(),
        'img_size': IMG_SIZE,
        'num_classes': num_classes,
        'final_train_accuracy': float(train_acc),
        'final_val_accuracy': float(val_acc),
        'version': 'landmarks',
        'landmark_annotated': True
    }

    with open(MODEL_DIR / 'model_metadata_processed.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {MODEL_DIR / 'model_metadata_processed.json'}")

    plot_training_history(history, MODEL_DIR / 'training_history_processed.png')

    print("\n" + "="*60)
    print("Ready to test with real_time_classifier.py!")
    print("="*60)

if __name__ == "__main__":
    main()
