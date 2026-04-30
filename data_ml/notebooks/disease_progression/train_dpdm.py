"""
Training Script — Disease Progression Detection Module (DPDM)
=============================================================
Implements the two-phase training strategy described in the project report
(Section 4.4.2, Listing 4.5):

    Phase 1 : CNN backbone frozen  →  train LSTM + Dense head  (30 epochs)
    Phase 2 : Unfreeze top-30 CNN layers  →  fine-tune end-to-end  (10 epochs)

Input data:
    Synthetic .npy sequences produced by synthetic_sequence_generator.py
    Each file has shape  (5, 224, 224, 3),  dtype float32,  range [0, 1].

Dataset split (report §5.1.2):
    70 % training  |  15 % validation  |  15 % test

Usage:
    python train_dpdm.py

Author: AgroSmart Team
Report: Sections 4.4.2, 5.3
"""

import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from pathlib import Path
from typing import Tuple, Dict, List

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from cnn_lstm_model import build_cnn_lstm_model, prepare_for_finetuning

# ─────────────────────────────────────────────────────────────────────────────
# Configuration  (mirrors report Listing 4.5 constants)
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    # Data
    "data_dir"          : "../../datasets/synthetic_sequences",
    "image_shape"       : (224, 224, 3),
    "sequence_length"   : 5,

    # Model
    "backbone"          : "mobilenetv2",   # "mobilenetv2" | "efficientnetb3"
    "lstm_units"        : 256,
    "dense_units"       : 128,
    "dropout_rate"      : 0.4,
    "multi_task"        : True,

    # Training — Phase 1 (frozen backbone)
    "batch_size"        : 16,
    "epochs_phase1"     : 30,
    "lr_phase1"         : 1e-4,
    "patience_es"       : 5,               # EarlyStopping patience
    "patience_lr"       : 3,               # ReduceLROnPlateau patience

    # Training — Phase 2 (fine-tuning)
    "epochs_phase2"     : 10,
    "lr_phase2"         : 5e-6,
    "unfreeze_layers"   : 30,

    # Outputs
    "model_save_path"   : "models/dpdm/cnn_lstm_disease.keras",
    "history_save_path" : "models/dpdm/training_history.json",
    "class_names_path"  : "models/dpdm/class_names.npy",

    # Reproducibility
    "random_seed"       : 42,
}

np.random.seed(CFG["random_seed"])
tf.random.set_seed(CFG["random_seed"])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loader
# ─────────────────────────────────────────────────────────────────────────────

class SequenceDatasetLoader:
    """
    Loads all .npy sequences from the synthetic dataset directory.

    Expected directory structure:
        data_dir/
            <ClassName_A>/
                seq_0000001.npy
                seq_0000002.npy
                ...
            <ClassName_B>/
                ...

    Each .npy file contains one sample: shape (T, H, W, C) float32.
    """

    def __init__(self, data_dir: str, sequence_length: int = 5):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.label_binarizer = LabelBinarizer()
        self.class_names: List[str] = []

    # ── Public ───────────────────────────────────────────────────────────────

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Scan the dataset directory and load all sequences.

        Returns:
            X         : float32 array  (N, T, H, W, C)
            y_onehot  : float32 array  (N, num_classes)   — disease labels
            y_severity: float32 array  (N, 1)              — severity score
            y_prograte: float32 array  (N, 1)              — progression rate
        """
        class_dirs = sorted([
            d for d in self.data_dir.iterdir() if d.is_dir()
        ])

        if not class_dirs:
            raise FileNotFoundError(
                f"No class sub-directories found in {self.data_dir}.\n"
                "Run synthetic_sequence_generator.py first."
            )

        self.class_names = [d.name for d in class_dirs]
        print(f"Classes found ({len(self.class_names)}): {self.class_names}")

        X_list, y_raw, sev_list, prog_list = [], [], [], []

        for label_idx, cls_dir in enumerate(class_dirs):
            npy_files = list(cls_dir.glob("*.npy"))
            if not npy_files:
                print(f"  [WARN] No .npy files in {cls_dir.name} — skipping.")
                continue

            severity = _severity_from_class_name(cls_dir.name)
            progression_rate = severity / max(self.sequence_length - 1, 1)

            print(f"  Loading {cls_dir.name}: {len(npy_files)} sequences "
                  f"(severity={severity:.2f})")

            for npy_path in npy_files:
                seq = np.load(str(npy_path))

                # Shape guard
                expected = (self.sequence_length,) + (224, 224, 3)
                if seq.shape != expected:
                    # Try to accommodate different image sizes gracefully
                    if seq.ndim == 4 and seq.shape[0] == self.sequence_length:
                        pass   # will be caught downstream if mismatched
                    else:
                        continue

                X_list.append(seq)
                y_raw.append(label_idx)
                sev_list.append(severity)
                prog_list.append(progression_rate)

        if not X_list:
            raise RuntimeError("No valid sequences were loaded. Check data_dir.")

        X = np.array(X_list, dtype=np.float32)
        y_raw_arr = np.array(y_raw, dtype=np.int32)
        y_onehot = self.label_binarizer.fit_transform(y_raw_arr).astype(np.float32)
        y_severity = np.array(sev_list, dtype=np.float32).reshape(-1, 1)
        y_prograte = np.array(prog_list, dtype=np.float32).reshape(-1, 1)

        print(f"\nDataset loaded:")
        print(f"  X shape     : {X.shape}")
        print(f"  y_onehot    : {y_onehot.shape}")
        print(f"  num_classes : {y_onehot.shape[1]}")

        return X, y_onehot, y_severity, y_prograte

    def split(
        self,
        X: np.ndarray,
        y_onehot: np.ndarray,
        y_severity: np.ndarray,
        y_prograte: np.ndarray,
    ):
        """
        Stratified 70/15/15 split (report §5.1.2).

        Returns:
            (X_train, X_val, X_test,
             y_train_dict, y_val_dict, y_test_dict)
        """
        y_raw = np.argmax(y_onehot, axis=1)

        # 70 % train | 30 % temp
        X_train, X_temp, y_tr, y_tmp, s_tr, s_tmp, p_tr, p_tmp = \
            train_test_split(
                X, y_onehot, y_severity, y_prograte,
                test_size=0.30,
                random_state=CFG["random_seed"],
                stratify=y_raw,
            )

        # Split temp equally → 15 % val | 15 % test
        y_raw_tmp = np.argmax(y_tmp, axis=1)
        X_val, X_test, y_va, y_te, s_va, s_te, p_va, p_te = \
            train_test_split(
                X_temp, y_tmp, s_tmp, p_tmp,
                test_size=0.50,
                random_state=CFG["random_seed"],
                stratify=y_raw_tmp,
            )

        def _pack(y, s, p):
            return {
                "disease_classification": y,
                "severity_score":         s,
                "progression_rate":       p,
            }

        print(f"\nDataset split:")
        print(f"  Train : {len(X_train):>5} samples")
        print(f"  Val   : {len(X_val):>5} samples")
        print(f"  Test  : {len(X_test):>5} samples")

        return (
            X_train, X_val, X_test,
            _pack(y_tr, s_tr, p_tr),
            _pack(y_va, s_va, p_va),
            _pack(y_te, s_te, p_te),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

def make_callbacks(save_path: str, phase: int = 1) -> List[keras.callbacks.Callback]:
    """
    Construct training callbacks (report Listing 4.5, lines 52-63).

    Args:
        save_path : Where to checkpoint the best model weights.
        phase     : 1 or 2 (affects monitor metric name prefix).

    Returns:
        List of Keras callbacks.
    """
    os.makedirs(Path(save_path).parent, exist_ok=True)

    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=CFG["patience_es"],
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            monitor="val_disease_classification_accuracy" if CFG["multi_task"]
                     else "val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=CFG["patience_lr"],
            min_lr=1e-8,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            filename=save_path.replace(".keras", f"_phase{phase}.csv"),
            append=False,
        ),
        keras.callbacks.TensorBoard(
            log_dir=f"models/dpdm/logs/phase{phase}",
            histogram_freq=1,
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _severity_from_class_name(class_name: str) -> float:
    """
    Infer a numeric severity label [0, 1] from the class folder name.
    Mirrors the heuristic used in the existing train_model.py.
    """
    name = class_name.lower()
    if "healthy" in name:
        return 0.0
    if "early" in name or "mild" in name:
        return 0.3
    if "moderate" in name:
        return 0.5
    if "severe" in name or "advanced" in name or "late" in name:
        return 0.8
    return 0.6                # generic diseased default


def _merge_histories(h1: dict, h2: dict) -> dict:
    """Concatenate two Keras history dicts for unified plotting."""
    merged = {}
    for key in h1:
        merged[key] = h1[key] + h2.get(key, [])
    # Include any keys only in h2
    for key in h2:
        if key not in merged:
            merged[key] = [None] * len(h1.get(list(h1.keys())[0], [])) + h2[key]
    return merged


def _print_banner(text: str):
    bar = "=" * 60
    print(f"\n{bar}\n  {text}\n{bar}")


# ─────────────────────────────────────────────────────────────────────────────
# Main training pipeline  (Report Listing 4.5)
# ─────────────────────────────────────────────────────────────────────────────

def main():
    start_time = time.time()

    _print_banner("LOADING DATASET")

    loader = SequenceDatasetLoader(
        data_dir=CFG["data_dir"],
        sequence_length=CFG["sequence_length"],
    )
    X, y_onehot, y_severity, y_prograte = loader.load()
    num_classes = y_onehot.shape[1]

    X_train, X_val, X_test, y_train, y_val, y_test = loader.split(
        X, y_onehot, y_severity, y_prograte
    )

    # Save class names for inference
    os.makedirs(Path(CFG["class_names_path"]).parent, exist_ok=True)
    np.save(CFG["class_names_path"], loader.class_names)
    print(f"\nClass names saved → {CFG['class_names_path']}")

    # ── Build model ───────────────────────────────────────────
    _print_banner("BUILDING MODEL")

    model = build_cnn_lstm_model(
        sequence_length = CFG["sequence_length"],
        image_shape     = CFG["image_shape"],
        num_classes     = num_classes,
        backbone        = CFG["backbone"],
        lstm_units      = CFG["lstm_units"],
        dense_units     = CFG["dense_units"],
        dropout_rate    = CFG["dropout_rate"],
        multi_task      = CFG["multi_task"],
    )
    model.summary()

    # ── Phase 1: Frozen backbone ──────────────────────────────
    _print_banner(f"PHASE 1 — Training LSTM head (CNN frozen) — {CFG['epochs_phase1']} epochs")

    callbacks_p1 = make_callbacks(CFG["model_save_path"], phase=1)

    history_p1 = model.fit(
        X_train,
        y_train if CFG["multi_task"] else y_train["disease_classification"],
        validation_data=(
            X_val,
            y_val if CFG["multi_task"] else y_val["disease_classification"],
        ),
        epochs      = CFG["epochs_phase1"],
        batch_size  = CFG["batch_size"],
        callbacks   = callbacks_p1,
        verbose     = 1,
    )

    # ── Phase 2: Fine-tuning ──────────────────────────────────
    _print_banner(f"PHASE 2 — Fine-tuning top-{CFG['unfreeze_layers']} CNN layers — {CFG['epochs_phase2']} epochs")

    model = prepare_for_finetuning(
        model                  = model,
        num_layers_to_unfreeze = CFG["unfreeze_layers"],
        new_lr                 = CFG["lr_phase2"],
        multi_task             = CFG["multi_task"],
    )

    callbacks_p2 = make_callbacks(CFG["model_save_path"], phase=2)

    history_p2 = model.fit(
        X_train,
        y_train if CFG["multi_task"] else y_train["disease_classification"],
        validation_data=(
            X_val,
            y_val if CFG["multi_task"] else y_val["disease_classification"],
        ),
        epochs      = CFG["epochs_phase2"],
        batch_size  = CFG["batch_size"],
        callbacks   = callbacks_p2,
        verbose     = 1,
    )

    # ── Final evaluation on held-out test set ─────────────────
    _print_banner("FINAL TEST SET EVALUATION")

    results = model.evaluate(
        X_test,
        y_test if CFG["multi_task"] else y_test["disease_classification"],
        batch_size = CFG["batch_size"],
        verbose    = 1,
    )

    # Pretty-print metric names and values
    metric_names = model.metrics_names
    print("\n── Test Results ─────────────────────────────────────")
    for name, val in zip(metric_names, results):
        print(f"  {name:<50} {val:.4f}")

    # ── Save training history ─────────────────────────────────
    combined_history = _merge_histories(
        history_p1.history, history_p2.history
    )
    with open(CFG["history_save_path"], "w") as f:
        # Convert numpy float values to plain Python floats for JSON
        serialisable = {
            k: [float(v) for v in vals]
            for k, vals in combined_history.items()
        }
        json.dump(serialisable, f, indent=2)
    print(f"\nTraining history saved → {CFG['history_save_path']}")

    # ── Save final model ──────────────────────────────────────
    final_path = CFG["model_save_path"].replace(".keras", "_final.keras")
    model.save(final_path)
    print(f"Final model saved    → {final_path}")

    elapsed = (time.time() - start_time) / 60
    _print_banner(f"TRAINING COMPLETE  ({elapsed:.1f} min)")


if __name__ == "__main__":
    main()
