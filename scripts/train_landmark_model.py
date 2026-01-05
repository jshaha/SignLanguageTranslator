#!/usr/bin/env python3
"""
Train sign language classifier using MediaPipe landmark coordinates.
Much more efficient than CNN approach - uses hand geometry directly.
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import json

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data" / "landmark_features"
MODEL_DIR = Path(__file__).parent.parent / "models"
BATCH_SIZE = 32
EPOCHS = 100
VALIDATION_SPLIT = 0.2

def load_landmark_dataset():
    """Load landmark feature dataset"""
    print("Loading landmark dataset...")

    features_path = DATA_DIR / "features.npy"
    labels_path = DATA_DIR / "labels.npy"
    metadata_path = DATA_DIR / "metadata.json"

    if not features_path.exists():
        print(f"ERROR: Features not found at {features_path}")
        print("Please run extract_landmarks.py first!")
        return None, None, None

    X = np.load(features_path)
    y = np.load(labels_path)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"\nDataset loaded:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]} (21 landmarks × 3 coordinates)")
    print(f"  Classes: {metadata['num_classes']}")

    return X, y, metadata

def prepare_labels(labels):
    """Encode labels to categorical"""
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    categorical_labels = keras.utils.to_categorical(encoded_labels, num_classes)

    print(f"\nClasses: {list(label_encoder.classes_)}")
    print(f"Number of classes: {num_classes}")

    return categorical_labels, label_encoder

def create_landmark_model(input_dim, num_classes):
    """
    Create a dense neural network for landmark classification.
    Much simpler and faster than CNN!
    """

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        # Input normalization
        layers.BatchNormalization(),

        # Hidden layers with dropout for regularization
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),

        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),

        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

def plot_training_history(history, save_path):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy (Landmark-Based)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss (Landmark-Based)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nPlots saved: {save_path}")
    plt.close()

def main():
    """Main training function"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("TRAINING LANDMARK-BASED SIGN LANGUAGE CLASSIFIER")
    print("Using MediaPipe Hand Landmark Coordinates")
    print("="*60)

    # Load dataset
    X, y, metadata = load_landmark_dataset()

    if X is None:
        return

    # Prepare labels
    y_categorical, label_encoder = prepare_labels(y)
    num_classes = len(label_encoder.classes_)

    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_categorical,
        test_size=VALIDATION_SPLIT,
        random_state=42,
        stratify=y
    )

    print(f"\nTraining: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")

    # Normalize features
    print("\nNormalizing landmark coordinates...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Save scaler for later use
    import pickle
    with open(MODEL_DIR / 'landmark_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved: {MODEL_DIR / 'landmark_scaler.pkl'}")

    # Create model
    print("\nBuilding landmark-based model...")
    model = create_landmark_model(X.shape[1], num_classes)
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
            str(MODEL_DIR / 'best_landmark_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=12,
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

    if val_acc > 0.95:
        print("\n✓ Excellent! Landmark-based model works great!")
    elif val_acc > 0.85:
        print("\n✓ Very good! Model should work well")
    elif val_acc > 0.70:
        print("\n✓ Good performance")
    else:
        print("\n~ Could be better - may need tuning")

    # Save model
    model.save(MODEL_DIR / 'sign_language_landmark_model.keras')
    print(f"\nModel saved: {MODEL_DIR / 'sign_language_landmark_model.keras'}")

    # Save metadata
    model_metadata = {
        'classes': label_encoder.classes_.tolist(),
        'num_classes': num_classes,
        'num_features': X.shape[1],
        'final_train_accuracy': float(train_acc),
        'final_val_accuracy': float(val_acc),
        'model_type': 'landmark_based',
        'feature_description': '21 landmarks × 3 coordinates (x, y, z)',
        'uses_scaler': True
    }

    with open(MODEL_DIR / 'landmark_model_metadata.json', 'w') as f:
        json.dump(model_metadata, f, indent=2)
    print(f"Metadata saved: {MODEL_DIR / 'landmark_model_metadata.json'}")

    # Plot history
    plot_training_history(history, MODEL_DIR / 'landmark_training_history.png')

    print("\n" + "="*60)
    print("MODEL ADVANTAGES:")
    print("="*60)
    print("✓ Much smaller model size")
    print("✓ Faster training and inference")
    print("✓ Robust to lighting and background")
    print("✓ Works with any skin tone")
    print("✓ Based on hand geometry, not pixels")
    print("\nNext: Update real_time_classifier.py to use this model!")
    print("="*60)

if __name__ == "__main__":
    main()
