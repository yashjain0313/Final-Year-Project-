"""
Training Script for Disease Progression Detection Model

This script handles:
1. Data loading and preprocessing
2. Synthetic temporal sequence generation
3. Model training with callbacks
4. Evaluation and visualization
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras

# Import our custom model
from disease_progression_model import (
    DiseaseProgressionModel,
    TemporalDataGenerator,
    GradCAMVisualizer,
    create_training_callbacks
)


class DiseaseDatasetLoader:
    """
    Load and preprocess PlantVillage dataset
    """
    
    def __init__(
        self,
        dataset_path: str,
        image_size: Tuple[int, int] = (224, 224),
        sequence_length: int = 5
    ):
        """
        Initialize dataset loader
        
        Args:
            dataset_path: Path to dataset directory
            image_size: Target image size
            sequence_length: Number of images per sequence
        """
        self.dataset_path = Path(dataset_path)
        self.image_size = image_size
        self.sequence_length = sequence_length
        
        self.label_encoder = LabelEncoder()
        self.class_names = []
        
        # Temporal data generator
        self.temp_gen = TemporalDataGenerator(sequence_length=sequence_length)
    
    def load_dataset(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Load dataset and create temporal sequences
        
        Returns:
            X: Image sequences (N, seq_len, H, W, 3)
            y: Dictionary of labels
        """
        print("Loading dataset...")
        
        # Get all image paths and labels
        image_paths = []
        labels = []
        
        # Assuming dataset structure: dataset_path/class_name/image.jpg
        for class_dir in self.dataset_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                # Look for images with various extensions (case-insensitive)
                for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
                    for img_path in class_dir.glob(ext):
                        image_paths.append(str(img_path))
                        labels.append(class_name)
        
        print(f"Found {len(image_paths)} images across {len(set(labels))} classes")
        
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        self.class_names = self.label_encoder.classes_
        
        # Create sequences
        X_sequences = []
        y_disease = []
        y_severity = []
        y_progression = []
        
        print("Creating temporal sequences...")
        for i, (img_path, label) in enumerate(zip(image_paths, labels_encoded)):
            if i % 100 == 0:
                print(f"Processing {i}/{len(image_paths)}...")
            
            # Load image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.image_size)
            
            # Determine severity based on class name
            severity = self._estimate_severity(labels[i])
            
            # Create progression sequence
            sequence = self.temp_gen.create_progression_sequence(img, severity)
            
            # Calculate progression rate
            progression_rate = severity / (self.sequence_length - 1)
            
            X_sequences.append(sequence)
            y_disease.append(label)
            y_severity.append(severity)
            y_progression.append(progression_rate)
        
        # Convert to numpy arrays
        X = np.array(X_sequences, dtype=np.float32) / 255.0
        
        y = {
            'disease_classification': keras.utils.to_categorical(
                y_disease,
                num_classes=len(self.class_names)
            ),
            'severity_score': np.array(y_severity, dtype=np.float32).reshape(-1, 1),
            'progression_rate': np.array(y_progression, dtype=np.float32).reshape(-1, 1)
        }
        
        print(f"\nDataset loaded successfully!")
        print(f"X shape: {X.shape}")
        print(f"Number of classes: {len(self.class_names)}")
        
        return X, y
    
    def _estimate_severity(self, class_name: str) -> float:
        """
        Estimate disease severity from class name
        
        Args:
            class_name: Name of the class
        
        Returns:
            Severity score (0-1)
        """
        class_lower = class_name.lower()
        
        # Healthy plants
        if 'healthy' in class_lower:
            return 0.0
        
        # Severity keywords
        if 'early' in class_lower or 'mild' in class_lower:
            return 0.3
        elif 'moderate' in class_lower:
            return 0.5
        elif 'severe' in class_lower or 'advanced' in class_lower:
            return 0.9
        else:
            # Default for diseased plants
            return 0.6
    
    def split_dataset(
        self,
        X: np.ndarray,
        y: Dict[str, np.ndarray],
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42
    ) -> Tuple:
        """
        Split dataset into train/val/test
        
        Args:
            X: Input sequences
            y: Labels dictionary
            test_size: Test set proportion
            val_size: Validation set proportion
            random_state: Random seed
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            {k: v for k, v in y.items()},
            test_size=test_size,
            random_state=random_state,
            stratify=np.argmax(y['disease_classification'], axis=1)
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            {k: v for k, v in y_temp.items()},
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=np.argmax(y_temp['disease_classification'], axis=1)
        )
        
        # Reconstruct y dictionaries
        y_train_dict = {
            'disease_classification': y_train['disease_classification'],
            'severity_score': y_train['severity_score'],
            'progression_rate': y_train['progression_rate']
        }
        
        y_val_dict = {
            'disease_classification': y_val['disease_classification'],
            'severity_score': y_val['severity_score'],
            'progression_rate': y_val['progression_rate']
        }
        
        y_test_dict = {
            'disease_classification': y_test['disease_classification'],
            'severity_score': y_test['severity_score'],
            'progression_rate': y_test['progression_rate']
        }
        
        print(f"\nDataset split:")
        print(f"Train: {len(X_train)} samples")
        print(f"Val: {len(X_val)} samples")
        print(f"Test: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train_dict, y_val_dict, y_test_dict


def plot_training_history(history, save_path: str = 'training_history.png'):
    """
    Plot training history
    
    Args:
        history: Keras training history
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Disease classification accuracy
    axes[0, 0].plot(history.history['disease_classification_accuracy'], label='Train')
    axes[0, 0].plot(history.history['val_disease_classification_accuracy'], label='Val')
    axes[0, 0].set_title('Disease Classification Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Disease classification loss
    axes[0, 1].plot(history.history['disease_classification_loss'], label='Train')
    axes[0, 1].plot(history.history['val_disease_classification_loss'], label='Val')
    axes[0, 1].set_title('Disease Classification Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Severity MAE
    axes[0, 2].plot(history.history['severity_score_mae'], label='Train')
    axes[0, 2].plot(history.history['val_severity_score_mae'], label='Val')
    axes[0, 2].set_title('Severity Score MAE')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('MAE')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Severity loss
    axes[1, 0].plot(history.history['severity_score_loss'], label='Train')
    axes[1, 0].plot(history.history['val_severity_score_loss'], label='Val')
    axes[1, 0].set_title('Severity Score Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Progression rate MAE
    axes[1, 1].plot(history.history['progression_rate_mae'], label='Train')
    axes[1, 1].plot(history.history['val_progression_rate_mae'], label='Val')
    axes[1, 1].set_title('Progression Rate MAE')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Total loss
    axes[1, 2].plot(history.history['loss'], label='Train')
    axes[1, 2].plot(history.history['val_loss'], label='Val')
    axes[1, 2].set_title('Total Loss')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history saved to {save_path}")


def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: Dict[str, np.ndarray],
    class_names: List[str]
):
    """
    Evaluate model on test set
    
    Args:
        model: Trained model
        X_test: Test sequences
        y_test: Test labels
        class_names: List of class names
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Get predictions
    predictions = model.predict(X_test, verbose=1)
    
    # Disease classification metrics
    y_pred_disease = np.argmax(predictions['disease_classification'], axis=1)
    y_true_disease = np.argmax(y_test['disease_classification'], axis=1)
    
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    
    accuracy = accuracy_score(y_true_disease, y_pred_disease)
    print(f"\nDisease Classification Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(
        y_true_disease,
        y_pred_disease,
        target_names=class_names,
        digits=4
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_true_disease, y_pred_disease)
    plt.figure(figsize=(15, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix saved to confusion_matrix.png")
    
    # Severity metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    y_pred_severity = predictions['severity_score'].flatten()
    y_true_severity = y_test['severity_score'].flatten()
    
    severity_mae = mean_absolute_error(y_true_severity, y_pred_severity)
    severity_rmse = np.sqrt(mean_squared_error(y_true_severity, y_pred_severity))
    severity_r2 = r2_score(y_true_severity, y_pred_severity)
    
    print(f"\nSeverity Prediction:")
    print(f"  MAE: {severity_mae:.4f}")
    print(f"  RMSE: {severity_rmse:.4f}")
    print(f"  R² Score: {severity_r2:.4f}")
    
    # Progression rate metrics
    y_pred_progression = predictions['progression_rate'].flatten()
    y_true_progression = y_test['progression_rate'].flatten()
    
    progression_mae = mean_absolute_error(y_true_progression, y_pred_progression)
    progression_rmse = np.sqrt(mean_squared_error(y_true_progression, y_pred_progression))
    
    print(f"\nProgression Rate Prediction:")
    print(f"  MAE: {progression_mae:.4f}")
    print(f"  RMSE: {progression_rmse:.4f}")
    
    # Scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Severity scatter
    axes[0].scatter(y_true_severity, y_pred_severity, alpha=0.5)
    axes[0].plot([0, 1], [0, 1], 'r--', lw=2)
    axes[0].set_xlabel('True Severity')
    axes[0].set_ylabel('Predicted Severity')
    axes[0].set_title(f'Severity Prediction (MAE={severity_mae:.4f})')
    axes[0].grid(True)
    
    # Progression scatter
    axes[1].scatter(y_true_progression, y_pred_progression, alpha=0.5)
    min_val = min(y_true_progression.min(), y_pred_progression.min())
    max_val = max(y_true_progression.max(), y_pred_progression.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[1].set_xlabel('True Progression Rate')
    axes[1].set_ylabel('Predicted Progression Rate')
    axes[1].set_title(f'Progression Rate (MAE={progression_mae:.4f})')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('regression_plots.png', dpi=300, bbox_inches='tight')
    print("Regression plots saved to regression_plots.png")


def main():
    """
    Main training pipeline
    """
    # Configuration
    DATASET_PATH = '../../datasets/plant_disease/PlantVillage/PlantVillage'  # Nested PlantVillage folder
    IMAGE_SIZE = (224, 224)
    SEQUENCE_LENGTH = 5
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Load dataset
    print("="*50)
    print("LOADING DATASET")
    print("="*50)
    
    loader = DiseaseDatasetLoader(
        dataset_path=DATASET_PATH,
        image_size=IMAGE_SIZE,
        sequence_length=SEQUENCE_LENGTH
    )
    
    X, y = loader.load_dataset()
    
    # Split dataset
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_dataset(X, y)
    
    # Build model
    print("\n" + "="*50)
    print("BUILDING MODEL")
    print("="*50)
    
    model_builder = DiseaseProgressionModel(
        num_classes=len(loader.class_names),
        sequence_length=SEQUENCE_LENGTH,
        image_size=IMAGE_SIZE,
        cnn_backbone='efficientnet',
        lstm_units=[256, 128],
        dropout_rate=0.3
    )
    
    model = model_builder.build_model()
    model_builder.compile_model(learning_rate=LEARNING_RATE)
    
    print("\nModel Summary:")
    model_builder.summary()
    
    # Training callbacks
    callbacks = create_training_callbacks(
        model_name='disease_progression',
        patience=10
    )
    
    # Train model
    print("\n" + "="*50)
    print("TRAINING MODEL")
    print("="*50)
    
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test, loader.class_names)
    
    # Save final model
    model.save('models/disease_progression_final.h5')
    print("\nFinal model saved to models/disease_progression_final.h5")
    
    # Save class names
    np.save('models/class_names.npy', loader.class_names)
    print("Class names saved to models/class_names.npy")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)


if __name__ == '__main__':
    main()
