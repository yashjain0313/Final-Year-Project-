"""
Disease Progression Detection Model
CNN+LSTM Hybrid Architecture for Temporal Crop Disease Analysis

Author: AgroSmart Team
Date: 2025-11-27
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3, ResNet50
import numpy as np
import cv2
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt


class DiseaseProgressionModel:
    """
    CNN+LSTM Hybrid Model for Disease Progression Detection
    
    Architecture:
    - CNN Feature Extractor (EfficientNetB3/ResNet50)
    - LSTM Temporal Processor
    - Multi-Task Output (Disease Class, Severity, Progression Rate)
    """
    
    def __init__(
        self,
        num_classes: int = 38,
        sequence_length: int = 5,
        image_size: Tuple[int, int] = (224, 224),
        cnn_backbone: str = 'efficientnet',
        lstm_units: List[int] = [256, 128],
        dropout_rate: float = 0.3
    ):
        """
        Initialize the Disease Progression Model
        
        Args:
            num_classes: Number of disease classes
            sequence_length: Number of images in temporal sequence (default: 5)
            image_size: Input image dimensions
            cnn_backbone: 'efficientnet' or 'resnet50'
            lstm_units: List of LSTM layer units
            dropout_rate: Dropout rate for regularization
        """
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.cnn_backbone = cnn_backbone
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        
        self.model = None
        self.feature_extractor = None
        
    def build_model(self) -> keras.Model:
        """
        Build the complete CNN+LSTM hybrid model
        
        Returns:
            Compiled Keras model
        """
        # Input: Sequence of images
        input_sequence = layers.Input(
            shape=(self.sequence_length, *self.image_size, 3),
            name='image_sequence'
        )
        
        # CNN Feature Extractor (applied to each image in sequence)
        feature_extractor = self._build_cnn_feature_extractor()
        
        # TimeDistributed wrapper to apply CNN to each timestep
        features = layers.TimeDistributed(
            feature_extractor,
            name='time_distributed_cnn'
        )(input_sequence)
        
        # LSTM Temporal Processing
        x = features
        for i, units in enumerate(self.lstm_units[:-1]):
            x = layers.LSTM(
                units,
                return_sequences=True,
                name=f'lstm_{i+1}'
            )(x)
            x = layers.Dropout(self.dropout_rate, name=f'dropout_lstm_{i+1}')(x)
        
        # Final LSTM layer (no return_sequences)
        x = layers.LSTM(
            self.lstm_units[-1],
            return_sequences=False,
            name=f'lstm_{len(self.lstm_units)}'
        )(x)
        x = layers.Dropout(self.dropout_rate, name=f'dropout_lstm_{len(self.lstm_units)}')(x)
        
        # Dense layers
        x = layers.Dense(256, activation='relu', name='dense_1')(x)
        x = layers.Dropout(0.4, name='dropout_dense_1')(x)
        x = layers.Dense(128, activation='relu', name='dense_2')(x)
        x = layers.Dropout(0.3, name='dropout_dense_2')(x)
        
        # Multi-Task Output Heads
        
        # 1. Disease Classification
        disease_output = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='disease_classification'
        )(x)
        
        # 2. Severity Score (0-1)
        severity_output = layers.Dense(
            1,
            activation='sigmoid',
            name='severity_score'
        )(x)
        
        # 3. Progression Rate (can be negative if improving)
        progression_output = layers.Dense(
            1,
            activation='linear',
            name='progression_rate'
        )(x)
        
        # Create model
        model = keras.Model(
            inputs=input_sequence,
            outputs={
                'disease_classification': disease_output,
                'severity_score': severity_output,
                'progression_rate': progression_output
            },
            name='disease_progression_model'
        )
        
        self.model = model
        self.feature_extractor = feature_extractor
        
        return model
    
    def _build_cnn_feature_extractor(self) -> keras.Model:
        """
        Build CNN feature extractor using transfer learning
        
        Returns:
            Feature extractor model
        """
        input_img = layers.Input(shape=(*self.image_size, 3))
        
        # Load pretrained backbone
        if self.cnn_backbone == 'efficientnet':
            base_model = EfficientNetB3(
                include_top=False,
                weights='imagenet',
                input_tensor=input_img,
                pooling='avg'
            )
        elif self.cnn_backbone == 'resnet50':
            base_model = ResNet50(
                include_top=False,
                weights='imagenet',
                input_tensor=input_img,
                pooling='avg'
            )
        else:
            raise ValueError(f"Unknown backbone: {self.cnn_backbone}")
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom layers on top
        x = base_model.output
        x = layers.Dense(1024, activation='relu', name='fc1')(x)
        x = layers.Dropout(0.3, name='dropout_fc1')(x)
        features = layers.Dense(512, activation='relu', name='features')(x)
        
        feature_extractor = keras.Model(
            inputs=input_img,
            outputs=features,
            name='cnn_feature_extractor'
        )
        
        return feature_extractor
    
    def compile_model(
        self,
        learning_rate: float = 0.0001,
        loss_weights: Dict[str, float] = None
    ):
        """
        Compile the model with appropriate losses and metrics
        
        Args:
            learning_rate: Learning rate for Adam optimizer
            loss_weights: Weights for multi-task losses
        """
        if loss_weights is None:
            loss_weights = {
                'disease_classification': 1.0,
                'severity_score': 0.5,
                'progression_rate': 0.3
            }
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                'disease_classification': 'categorical_crossentropy',
                'severity_score': 'mse',
                'progression_rate': 'mae'
            },
            loss_weights=loss_weights,
            metrics={
                'disease_classification': ['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3)],
                'severity_score': ['mae', 'mse'],
                'progression_rate': ['mae']
            }
        )
    
    def unfreeze_backbone(self, num_layers: int = 20):
        """
        Unfreeze top layers of CNN backbone for fine-tuning
        
        Args:
            num_layers: Number of layers to unfreeze from the top
        """
        if self.feature_extractor is None:
            raise ValueError("Model must be built before unfreezing")
        
        # Get the base model (EfficientNet or ResNet)
        base_model = self.feature_extractor.layers[1]  # First layer is Input
        
        # Unfreeze top layers
        for layer in base_model.layers[-num_layers:]:
            layer.trainable = True
        
        print(f"Unfroze top {num_layers} layers of {self.cnn_backbone}")
    
    def summary(self):
        """Print model summary"""
        if self.model is None:
            raise ValueError("Model must be built first")
        self.model.summary()
    
    def plot_model(self, filename: str = 'model_architecture.png'):
        """
        Plot model architecture
        
        Args:
            filename: Output filename for plot
        """
        keras.utils.plot_model(
            self.model,
            to_file=filename,
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=True
        )


class TemporalDataGenerator:
    """
    Generate synthetic temporal sequences for training
    """
    
    def __init__(self, sequence_length: int = 5):
        self.sequence_length = sequence_length
    
    def create_progression_sequence(
        self,
        image: np.ndarray,
        disease_severity: float = 0.5
    ) -> np.ndarray:
        """
        Create synthetic disease progression sequence
        
        Args:
            image: Base image (numpy array)
            disease_severity: Target severity (0-1)
        
        Returns:
            Sequence of images showing progression
        """
        sequence = []
        
        for step in range(self.sequence_length):
            # Calculate intensity for this step
            intensity = (step / (self.sequence_length - 1)) * disease_severity
            
            # Apply progressive degradation
            degraded = self._apply_degradation(image.copy(), intensity)
            sequence.append(degraded)
        
        return np.array(sequence)
    
    def _apply_degradation(
        self,
        image: np.ndarray,
        intensity: float
    ) -> np.ndarray:
        """
        Apply disease-like degradation to image
        
        Args:
            image: Input image (RGB, 0-255)
            intensity: Degradation intensity (0-1)
        
        Returns:
            Degraded image
        """
        # Ensure float type
        img = image.astype(np.float32) / 255.0
        
        # 1. Color shift (yellowing/browning)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[:, :, 0] = hsv[:, :, 0] * (1 - 0.3 * intensity)  # Shift hue
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + 0.5 * intensity), 0, 1)  # Increase saturation
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # 2. Add noise (disease spots)
        if intensity > 0.2:
            noise = np.random.normal(0, 0.1 * intensity, img.shape)
            img = np.clip(img + noise, 0, 1)
        
        # 3. Reduce brightness (wilting effect)
        img = img * (1 - 0.2 * intensity)
        
        # 4. Add brown spots
        if intensity > 0.3:
            img = self._add_lesions(img, num_spots=int(10 * intensity))
        
        # Convert back to uint8
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)
    
    def _add_lesions(
        self,
        image: np.ndarray,
        num_spots: int = 5
    ) -> np.ndarray:
        """
        Add disease lesions/spots to image
        
        Args:
            image: Input image (float, 0-1)
            num_spots: Number of lesions to add
        
        Returns:
            Image with lesions
        """
        h, w = image.shape[:2]
        
        for _ in range(num_spots):
            # Random position and size
            center_x = np.random.randint(0, w)
            center_y = np.random.randint(0, h)
            radius = np.random.randint(5, 20)
            
            # Create circular mask
            y, x = np.ogrid[:h, :w]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
            # Apply brown color to lesion
            brown_color = np.array([0.4, 0.3, 0.2])  # Brown in RGB
            image[mask] = image[mask] * 0.3 + brown_color * 0.7
        
        return image


class GradCAMVisualizer:
    """
    Grad-CAM visualization for model interpretability
    """
    
    def __init__(self, model: keras.Model, layer_name: str = None):
        """
        Initialize Grad-CAM visualizer
        
        Args:
            model: Trained model
            layer_name: Name of layer to visualize (default: last conv layer)
        """
        self.model = model
        self.layer_name = layer_name
        
        if layer_name is None:
            # Find last convolutional layer
            for layer in reversed(model.layers):
                if 'conv' in layer.name.lower():
                    self.layer_name = layer.name
                    break
    
    def generate_heatmap(
        self,
        image: np.ndarray,
        class_idx: int = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap
        
        Args:
            image: Input image
            class_idx: Target class index (default: predicted class)
        
        Returns:
            Heatmap array
        """
        # Create gradient model
        grad_model = keras.Model(
            inputs=self.model.input,
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output
            ]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(np.expand_dims(image, 0))
            
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            loss = predictions[:, class_idx]
        
        # Get gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Pool gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Overlay heatmap on original image
        
        Args:
            image: Original image
            heatmap: Grad-CAM heatmap
            alpha: Transparency of heatmap
        
        Returns:
            Overlayed image
        """
        # Resize heatmap to match image
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay
        superimposed = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return superimposed


def create_training_callbacks(
    model_name: str = 'disease_progression',
    patience: int = 10
) -> List[keras.callbacks.Callback]:
    """
    Create training callbacks
    
    Args:
        model_name: Name for saved models
        patience: Patience for early stopping
    
    Returns:
        List of callbacks
    """
    callbacks = [
        # Model checkpoint
        keras.callbacks.ModelCheckpoint(
            filepath=f'models/{model_name}_best.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard
        keras.callbacks.TensorBoard(
            log_dir=f'logs/{model_name}',
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),
        
        # CSV Logger
        keras.callbacks.CSVLogger(
            filename=f'logs/{model_name}_training.csv',
            append=True
        )
    ]
    
    return callbacks


if __name__ == '__main__':
    # Example usage
    print("Building Disease Progression Model...")
    
    # Initialize model
    model_builder = DiseaseProgressionModel(
        num_classes=38,
        sequence_length=5,
        image_size=(224, 224),
        cnn_backbone='efficientnet',
        lstm_units=[256, 128],
        dropout_rate=0.3
    )
    
    # Build model
    model = model_builder.build_model()
    
    # Compile model
    model_builder.compile_model(learning_rate=0.0001)
    
    # Print summary
    print("\nModel Summary:")
    model_builder.summary()
    
    print("\nModel built successfully!")
    print(f"Total parameters: {model.count_params():,}")
