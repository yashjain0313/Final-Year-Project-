"""
Module 3: Disease Progression Detection Module (DPDM)
=====================================================
CNN-LSTM hybrid architecture for temporal plant disease analysis.

Architecture (Report Section 4.4.1):
    Input  : Sequence of T images  →  (batch, T, 224, 224, 3)
    Layer 1: TimeDistributed(MobileNetV2) → spatial features per frame
    Layer 2: LSTM(256, return_sequences=True)
    Layer 3: Dropout(0.4)
    Layer 4: LSTM(256, return_sequences=False)
    Layer 5: Dense(128, ReLU) + Dropout(0.4)
    Output : Softmax(num_classes)   →  disease class probabilities

Training strategy (Report Section 4.4.2):
    Phase 1 : CNN backbone frozen, train LSTM head  (30 epochs)
    Phase 2 : Unfreeze top-30 CNN layers, fine-tune (10 epochs, lr=5e-6)

Multi-task variant (per existing project architecture, Section ARCHITECTURE.md):
    Adds two regression heads:
        • severity_score  (Dense(1, sigmoid))
        • progression_rate (Dense(1, linear))

Author: AgroSmart Team
Report Sections: 4.4, 5.3
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
from typing import List, Tuple, Dict, Optional
import numpy as np


# ─────────────────────────────────────────────────────────────
# 1.  CNN Feature Extractor  (Report Listing 4.4, lines 6-21)
# ─────────────────────────────────────────────────────────────

def build_cnn_feature_extractor(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    backbone: str = "mobilenetv2",
    trainable: bool = False,
) -> keras.Model:
    """
    Build a CNN feature extractor using MobileNetV2 (default, report §4.4.1)
    or EfficientNetB3 (extended variant, per ARCHITECTURE.md).

    The backbone is loaded with ImageNet weights and frozen by default
    (Phase-1 training).  Use `unfreeze_top_layers()` for Phase-2.

    Args:
        input_shape: (H, W, C) — must match sequence frame size.
        backbone:    "mobilenetv2" (report default) | "efficientnetb3"
        trainable:   If False, freeze all backbone layers (Phase-1).

    Returns:
        Keras Model that maps (batch, H, W, 3) → (batch, feature_dim).
            MobileNetV2  → feature_dim = 1280
            EfficientNetB3 → feature_dim = 1536
    """
    inputs = keras.Input(shape=input_shape, name="frame_input")

    if backbone == "mobilenetv2":
        # Report Listing 4.4 uses MobileNetV2 — lightweight, phone-deployable
        x = applications.mobilenet_v2.preprocess_input(inputs)
        base = applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
        base.trainable = trainable
        x = base(x, training=False)
        x = layers.GlobalAveragePooling2D(name="gap")(x)           # → (batch, 1280)
        feature_dim = 1280

    elif backbone == "efficientnetb3":
        # Richer features; used in the extended project architecture
        base = applications.EfficientNetB3(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs,
            pooling="avg",                                           # → (batch, 1536)
        )
        base.trainable = trainable
        x = base.output
        feature_dim = 1536

    else:
        raise ValueError(f"Unknown backbone '{backbone}'. Choose 'mobilenetv2' or 'efficientnetb3'.")

    # Optional projection layers (used in ARCHITECTURE.md extended variant)
    if backbone == "efficientnetb3":
        x = layers.Dense(1024, activation="relu", name="fc_proj_1")(x)
        x = layers.Dropout(0.3, name="drop_proj_1")(x)
        x = layers.Dense(512, activation="relu", name="fc_proj_2")(x)

    return keras.Model(inputs, x, name=f"cnn_{backbone}")


# ─────────────────────────────────────────────────────────────
# 2.  Full CNN-LSTM Model  (Report Listing 4.4, lines 23-70)
# ─────────────────────────────────────────────────────────────

def build_cnn_lstm_model(
    sequence_length: int,
    image_shape: Tuple[int, int, int],
    num_classes: int,
    backbone: str = "mobilenetv2",
    lstm_units: int = 256,
    dense_units: int = 128,
    dropout_rate: float = 0.4,
    multi_task: bool = True,
) -> keras.Model:
    """
    Build and compile the complete CNN-LSTM Disease Progression model.

    Args:
        sequence_length : T  — number of weekly images per input (e.g. 5).
        image_shape     : (H, W, C) per frame — (224, 224, 3).
        num_classes     : Number of disease classes (C).
        backbone        : CNN backbone to use.
        lstm_units      : Units in each LSTM layer (report default: 256).
        dense_units     : Units in the dense head (report default: 128).
        dropout_rate    : Dropout applied after LSTM and Dense (report: 0.4).
        multi_task      : If True, also outputs severity_score and
                          progression_rate heads (per ARCHITECTURE.md).

    Returns:
        Compiled tf.keras.Model ready for Phase-1 training.
    """
    # ── Build shared CNN extractor ────────────────────────────
    cnn = build_cnn_feature_extractor(
        input_shape=image_shape,
        backbone=backbone,
        trainable=False,          # frozen for Phase-1
    )

    # ── Sequence input: (batch, T, H, W, C) ──────────────────
    sequence_input = keras.Input(
        shape=(sequence_length,) + image_shape,
        name="image_sequence",
    )

    # ── TimeDistributed CNN: apply CNN to each frame ──────────
    features = layers.TimeDistributed(cnn, name="time_distributed_cnn")(
        sequence_input
    )  # → (batch, T, feature_dim)

    # ── Stacked LSTM layers ───────────────────────────────────
    x = layers.LSTM(
        lstm_units,
        return_sequences=True,
        name="lstm_layer_1",
    )(features)
    x = layers.Dropout(dropout_rate, name="dropout_lstm")(x)

    x = layers.LSTM(
        lstm_units,
        return_sequences=False,
        name="lstm_layer_2",
    )(x)                          # → (batch, lstm_units)

    # ── Dense classification head ─────────────────────────────
    x = layers.Dense(dense_units, activation="relu", name="dense_1")(x)
    x = layers.Dropout(dropout_rate, name="dropout_dense")(x)

    # ── Primary output: disease classification ────────────────
    disease_output = layers.Dense(
        num_classes,
        activation="softmax",
        name="disease_classification",
    )(x)

    outputs = {"disease_classification": disease_output}
    loss     = {"disease_classification": "categorical_crossentropy"}
    metrics  = {"disease_classification": ["accuracy"]}
    loss_weights = {"disease_classification": 1.0}

    # ── Optional regression heads (multi-task variant) ────────
    if multi_task:
        severity_output = layers.Dense(
            1, activation="sigmoid", name="severity_score"
        )(x)
        progression_output = layers.Dense(
            1, activation="linear", name="progression_rate"
        )(x)

        outputs["severity_score"]   = severity_output
        outputs["progression_rate"] = progression_output

        loss["severity_score"]   = "mse"
        loss["progression_rate"] = "mae"

        metrics["severity_score"]   = ["mae"]
        metrics["progression_rate"] = ["mae"]

        loss_weights["severity_score"]   = 0.5
        loss_weights["progression_rate"] = 0.3

    # ── Assemble model ────────────────────────────────────────
    model = keras.Model(
        inputs=sequence_input,
        outputs=outputs if multi_task else disease_output,
        name="CNN_LSTM_Disease_Detector",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=loss if multi_task else "categorical_crossentropy",
        metrics=metrics if multi_task else ["accuracy"],
        loss_weights=loss_weights if multi_task else None,
    )

    return model


# ─────────────────────────────────────────────────────────────
# 3.  Fine-tuning helper  (Report Listing 4.5, lines 76-93)
# ─────────────────────────────────────────────────────────────

def prepare_for_finetuning(
    model: keras.Model,
    num_layers_to_unfreeze: int = 30,
    new_lr: float = 5e-6,
    multi_task: bool = True,
) -> keras.Model:
    """
    Phase-2: Unfreeze the top N layers of the CNN backbone and recompile
    with a very low learning rate for fine-tuning.

    Args:
        model:                   The model returned by build_cnn_lstm_model().
        num_layers_to_unfreeze:  Number of layers to unfreeze from the top
                                 (report default: 30).
        new_lr:                  Fine-tuning learning rate (report: 5e-6).
        multi_task:              Must match the flag used during build.

    Returns:
        Recompiled model (in-place modification).
    """
    # Locate the backbone inside the TimeDistributed wrapper
    td_layer = model.get_layer("time_distributed_cnn")
    cnn_model = td_layer.layer                         # the shared CNN Model
    # Find the actual pretrained base (e.g. "mobilenetv2_1.00_224")
    base_model = None
    for lyr in cnn_model.layers:
        if hasattr(lyr, "layers"):                     # it's a Model/Sequential
            base_model = lyr
            break

    if base_model is None:
        raise RuntimeError("Could not locate backbone inside TimeDistributed CNN.")

    # Unfreeze entire CNN model first, then re-freeze lower layers
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    total_trainable = sum(
        1 for l in base_model.layers if l.trainable
    )
    print(f"[Fine-tune] Unfroze top {num_layers_to_unfreeze} backbone layers.")
    print(f"[Fine-tune] Trainable backbone layers: {total_trainable}")

    # Recompile with lower lr
    loss = {
        "disease_classification": "categorical_crossentropy",
        "severity_score": "mse",
        "progression_rate": "mae",
    }
    metrics = {
        "disease_classification": ["accuracy"],
        "severity_score": ["mae"],
        "progression_rate": ["mae"],
    }
    loss_weights = {
        "disease_classification": 1.0,
        "severity_score": 0.5,
        "progression_rate": 0.3,
    }

    if not multi_task:
        loss = "categorical_crossentropy"
        metrics = ["accuracy"]
        loss_weights = None

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=new_lr),
        loss=loss,
        metrics=metrics,
        loss_weights=loss_weights,
    )
    return model


# ─────────────────────────────────────────────────────────────
# 4.  Grad-CAM for interpretability  (per ARCHITECTURE.md §5)
# ─────────────────────────────────────────────────────────────

class GradCAMVisualizer:
    """
    Grad-CAM visualisation for a single frame within the sequence.
    Helps explain which regions of the leaf drove the classification.
    """

    def __init__(self, model: keras.Model, conv_layer_name: Optional[str] = None):
        self.model = model
        self.conv_layer_name = conv_layer_name

        if conv_layer_name is None:
            # Auto-detect last conv layer inside the TimeDistributed CNN
            self.conv_layer_name = self._find_last_conv_layer()

        print(f"[GradCAM] Using layer: {self.conv_layer_name}")

    def _find_last_conv_layer(self) -> str:
        for layer in reversed(self.model.layers):
            if "conv" in layer.name.lower():
                return layer.name
        raise RuntimeError("No convolutional layer found in model.")

    def generate_heatmap(
        self,
        sequence: np.ndarray,   # (T, H, W, 3) float32 in [0,1]
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for a given sequence.

        Returns:
            heatmap: float32 array in [0, 1], shape (H, W).
        """
        import cv2

        grad_model = keras.Model(
            inputs=self.model.input,
            outputs=[
                self.model.get_layer(self.conv_layer_name).output,
                self.model.output if not isinstance(self.model.output, dict)
                else self.model.output["disease_classification"],
            ],
        )

        seq_batch = np.expand_dims(sequence, 0)  # (1, T, H, W, 3)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(seq_batch)
            if class_idx is None:
                class_idx = int(tf.argmax(predictions[0]))
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()

    def overlay_heatmap(
        self,
        frame: np.ndarray,     # (H, W, 3) uint8 or float32
        heatmap: np.ndarray,   # (h, w) float32
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Overlay Grad-CAM heatmap on the original frame."""
        import cv2

        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)

        heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        superimposed = cv2.addWeighted(frame, 1 - alpha, heatmap_color, alpha, 0)
        return superimposed


# ─────────────────────────────────────────────────────────────
# 5.  Quick smoke-test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  CNN-LSTM DISEASE PROGRESSION MODEL — SMOKE TEST")
    print("  Report Section: 4.4")
    print("=" * 60)

    SEQUENCE_LENGTH = 5
    IMAGE_SHAPE = (224, 224, 3)
    NUM_CLASSES = 5        # Healthy, EarlyBlight, LateBlight, LeafMold, BrownSpot

    model = build_cnn_lstm_model(
        sequence_length=SEQUENCE_LENGTH,
        image_shape=IMAGE_SHAPE,
        num_classes=NUM_CLASSES,
        backbone="mobilenetv2",
        lstm_units=256,
        dense_units=128,
        dropout_rate=0.4,
        multi_task=True,
    )

    model.summary()
    print(f"\nTotal parameters : {model.count_params():,}")
    print(f"Trainable params : {sum(np.prod(v.shape) for v in model.trainable_variables):,}")

    # Dummy forward pass
    dummy_input = np.random.rand(2, SEQUENCE_LENGTH, *IMAGE_SHAPE).astype(np.float32)
    outputs = model(dummy_input, training=False)

    print("\n[Smoke test] Output shapes:")
    if isinstance(outputs, dict):
        for key, val in outputs.items():
            print(f"  {key}: {val.shape}")
    else:
        print(f"  disease_classification: {outputs.shape}")

    print("\n✅ Model built successfully!")
