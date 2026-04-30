"""
Evaluation Script — Disease Progression Detection Module (DPDM)
===============================================================
Reproduces the experimental results from Report Section 5.3.

Outputs:
    1. Overall accuracy / precision / recall / F1  (Table 5.2)
    2. Per-class F1-score breakdown               (Table 5.3)
    3. CNN-LSTM vs. Static CNN baseline comparison
    4. Confusion matrix heatmap
    5. Training dynamics curves (accuracy + loss)
    6. Severity & progression rate regression scatter plots

Usage:
    python evaluate_dpdm.py

Author: AgroSmart Team
Report: Sections 5.3.1 – 5.3.4
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — no GUI required
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import tensorflow as tf
from tensorflow import keras

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from cnn_lstm_model import build_cnn_lstm_model
from train_dpdm import SequenceDatasetLoader, CFG, _print_banner

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
EVAL_CFG = {
    "model_path"        : "models/dpdm/cnn_lstm_disease_final.keras",
    "class_names_path"  : "models/dpdm/class_names.npy",
    "history_path"      : "models/dpdm/training_history.json",
    "output_dir"        : "models/dpdm/eval_results",
    "data_dir"          : CFG["data_dir"],
    "sequence_length"   : CFG["sequence_length"],
    "batch_size"        : CFG["batch_size"],
    "multi_task"        : CFG["multi_task"],
}

os.makedirs(EVAL_CFG["output_dir"], exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Static CNN Baseline  (report §5.3.1 comparison)
# ─────────────────────────────────────────────────────────────────────────────

def build_static_cnn_baseline(
    image_shape: tuple,
    num_classes: int,
) -> keras.Model:
    """
    MobileNetV2 + GlobalAveragePooling + Dense head trained on single frames.
    This is the "Static CNN Baseline" in Report Table 5.2.
    """
    from tensorflow.keras import applications, layers

    inputs = keras.Input(shape=image_shape, name="single_frame")
    x = applications.mobilenet_v2.preprocess_input(inputs)
    base = applications.MobileNetV2(
        input_shape=image_shape,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs, outputs, name="Static_CNN_Baseline")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_evaluate_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    image_shape: tuple,
    epochs: int = 30,
    batch_size: int = 16,
) -> dict:
    """
    Train Static CNN baseline on middle-frame of each sequence, then evaluate.

    Returns dict with accuracy, precision, recall, f1, y_pred.
    """
    print("\n[Baseline] Extracting middle frames from sequences...")

    mid = EVAL_CFG["sequence_length"] // 2   # frame index 2 for T=5

    # Use only the middle frame (frame index 2)
    X_tr_f  = X_train[:, mid, ...]   # (N, H, W, C)
    X_va_f  = X_val[:,   mid, ...]
    X_te_f  = X_test[:,  mid, ...]

    model = build_static_cnn_baseline(image_shape, num_classes)

    cb = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=0
        ),
    ]

    model.fit(
        X_tr_f, y_train,
        validation_data=(X_va_f, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb,
        verbose=0,
    )

    y_pred_prob = model.predict(X_te_f, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    return {
        "accuracy"  : accuracy_score(y_true, y_pred),
        "precision" : precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall"    : recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1"        : f1_score(y_true, y_pred, average="macro", zero_division=0),
        "y_pred"    : y_pred,
        "y_true"    : y_true,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(history: dict, output_dir: str):
    """
    Plot training and validation curves — accuracy and loss per epoch.
    Reproduces the dynamics described in Report §5.3.3.
    """
    epochs = range(1, len(history.get("loss", [])) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("CNN-LSTM Training Dynamics", fontsize=16, fontweight="bold")

    # ── Disease classification accuracy ──────────────────────
    acc_key = "disease_classification_accuracy"
    val_acc_key = "val_disease_classification_accuracy"
    if acc_key not in history:
        acc_key, val_acc_key = "accuracy", "val_accuracy"

    if acc_key in history:
        axes[0, 0].plot(epochs, history[acc_key], label="Train", color="#2196F3")
        axes[0, 0].plot(epochs, history.get(val_acc_key, []), label="Val", color="#FF9800")
        axes[0, 0].set_title("Disease Classification Accuracy")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)

    # ── Total loss ────────────────────────────────────────────
    if "loss" in history:
        axes[0, 1].plot(epochs, history["loss"], label="Train", color="#2196F3")
        axes[0, 1].plot(epochs, history.get("val_loss", []), label="Val", color="#FF9800")
        axes[0, 1].set_title("Total Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # ── Severity MAE ──────────────────────────────────────────
    sev_key = "severity_score_mae"
    if sev_key in history:
        axes[1, 0].plot(epochs, history[sev_key], label="Train", color="#4CAF50")
        axes[1, 0].plot(epochs, history.get(f"val_{sev_key}", []), label="Val", color="#F44336")
        axes[1, 0].set_title("Severity Score MAE")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("MAE")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # ── Progression rate MAE ──────────────────────────────────
    prog_key = "progression_rate_mae"
    if prog_key in history:
        axes[1, 1].plot(epochs, history[prog_key], label="Train", color="#9C27B0")
        axes[1, 1].plot(epochs, history.get(f"val_{prog_key}", []), label="Val", color="#FF5722")
        axes[1, 1].set_title("Progression Rate MAE")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("MAE")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "training_dynamics.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training dynamics plot → {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str,
    output_dir: str,
    filename: str = "confusion_matrix.png",
):
    """Seaborn annotated confusion matrix."""
    from typing import List
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(max(14, len(class_names) * 2), 6))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Counts")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")
    axes[0].tick_params(axis="x", rotation=45)

    # Normalised
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title("Normalised")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix → {save_path}")


def plot_per_class_f1_comparison(
    baseline_f1: np.ndarray,
    cnn_lstm_f1: np.ndarray,
    class_names: List[str],
    output_dir: str,
):
    """
    Bar chart comparing per-class F1: Static CNN vs CNN-LSTM.
    Reproduces Report Figure 5.2.
    """
    from typing import List
    x = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 1.5), 6))
    bars1 = ax.bar(x - width / 2, baseline_f1, width, label="Static CNN Baseline",
                   color="#90CAF9", edgecolor="white")
    bars2 = ax.bar(x + width / 2, cnn_lstm_f1, width, label="CNN-LSTM (Proposed)",
                   color="#1565C0", edgecolor="white")

    ax.set_xlabel("Disease Class")
    ax.set_ylabel("F1-Score")
    ax.set_title("Per-Class F1-Score: CNN-LSTM vs. Static CNN Baseline\n(Report Figure 5.2)")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylim(0.80, 1.0)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate improvement arrows
    for i, (b, c) in enumerate(zip(baseline_f1, cnn_lstm_f1)):
        delta = c - b
        if delta > 0:
            ax.annotate(f"+{delta:.3f}",
                        xy=(x[i] + width / 2, c + 0.003),
                        ha="center", fontsize=8, color="darkgreen")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "per_class_f1_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Per-class F1 comparison → {save_path}")


def plot_regression_scatter(
    y_true_sev: np.ndarray,
    y_pred_sev: np.ndarray,
    y_true_prog: np.ndarray,
    y_pred_prog: np.ndarray,
    output_dir: str,
):
    """Scatter plots for severity and progression rate regression."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, y_true, y_pred, label in [
        (axes[0], y_true_sev,  y_pred_sev,  "Severity Score"),
        (axes[1], y_true_prog, y_pred_prog, "Progression Rate"),
    ]:
        mae = mean_absolute_error(y_true, y_pred)
        r2  = r2_score(y_true, y_pred)
        ax.scatter(y_true, y_pred, alpha=0.4, s=15, color="#1565C0")
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", lw=1.5, label="Perfect")
        ax.set_xlabel(f"True {label}")
        ax.set_ylabel(f"Predicted {label}")
        ax.set_title(f"{label}\nMAE={mae:.4f}  R²={r2:.4f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "regression_scatter.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Regression scatter → {save_path}")


def plot_comparison_table(baseline: dict, cnn_lstm: dict, output_dir: str):
    """
    Save a matplotlib table image replicating Report Table 5.2.
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    b_vals = [
        f"{baseline['accuracy']*100:.2f}%",
        f"{baseline['precision']*100:.2f}%",
        f"{baseline['recall']*100:.2f}%",
        f"{baseline['f1']*100:.2f}%",
    ]
    c_vals = [
        f"{cnn_lstm['accuracy']*100:.2f}%",
        f"{cnn_lstm['precision']*100:.2f}%",
        f"{cnn_lstm['recall']*100:.2f}%",
        f"{cnn_lstm['f1']*100:.2f}%",
    ]
    delta = [
        f"+{(cnn_lstm['accuracy']  - baseline['accuracy'])  * 100:.2f}%",
        f"+{(cnn_lstm['precision'] - baseline['precision']) * 100:.2f}%",
        f"+{(cnn_lstm['recall']    - baseline['recall'])    * 100:.2f}%",
        f"+{(cnn_lstm['f1']        - baseline['f1'])        * 100:.2f}%",
    ]

    fig, ax = plt.subplots(figsize=(9, 2.5))
    ax.axis("off")
    table_data = [["Metric", "Static CNN", "CNN-LSTM", "Δ Improvement"]] + \
                 list(zip(metrics, b_vals, c_vals, delta))

    tbl = ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.5, 2)

    # Style header
    for j in range(4):
        tbl[0, j].set_facecolor("#1565C0")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight CNN-LSTM column
    for i in range(1, len(metrics) + 1):
        tbl[i, 2].set_facecolor("#E3F2FD")

    ax.set_title("Table 5.2 — Disease Progression Detection: CNN-LSTM vs. Static CNN",
                 fontsize=11, pad=20, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(output_dir, "comparison_table_5_2.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Comparison table (Table 5.2) → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    from typing import List
    _print_banner("DPDM — EVALUATION PIPELINE  (Report §5.3)")

    # ── Load test data ────────────────────────────────────────
    loader = SequenceDatasetLoader(
        data_dir=EVAL_CFG["data_dir"],
        sequence_length=EVAL_CFG["sequence_length"],
    )
    X, y_onehot, y_severity, y_prograte = loader.load()
    _, _, X_test, _, _, y_test = loader.split(X, y_onehot, y_severity, y_prograte)

    # Also keep train/val for baseline training
    X_train, X_val, _, y_train_raw, y_val_raw, _ = loader.split(
        X, y_onehot, y_severity, y_prograte
    )

    # Load class names
    if Path(EVAL_CFG["class_names_path"]).exists():
        class_names: List[str] = list(
            np.load(EVAL_CFG["class_names_path"], allow_pickle=True)
        )
    else:
        class_names = loader.class_names

    num_classes = y_onehot.shape[1]
    image_shape = CFG["image_shape"]

    # ── Load trained CNN-LSTM model ───────────────────────────
    model_path = EVAL_CFG["model_path"]
    if not Path(model_path).exists():
        print(f"\n[WARN] Trained model not found at {model_path}.")
        print("       Building an untrained model for structural demonstration.")
        model = build_cnn_lstm_model(
            sequence_length = EVAL_CFG["sequence_length"],
            image_shape     = image_shape,
            num_classes     = num_classes,
            multi_task      = EVAL_CFG["multi_task"],
        )
    else:
        print(f"\nLoading model from: {model_path}")
        model = keras.models.load_model(model_path)

    # ── CNN-LSTM predictions ──────────────────────────────────
    _print_banner("CNN-LSTM — Inference on test set")

    raw_preds = model.predict(X_test, batch_size=EVAL_CFG["batch_size"], verbose=1)

    if isinstance(raw_preds, dict):
        disease_probs = raw_preds["disease_classification"]
        pred_severity = raw_preds["severity_score"].flatten()
        pred_prograte = raw_preds["progression_rate"].flatten()
    else:
        disease_probs = raw_preds
        pred_severity = np.zeros(len(X_test))
        pred_prograte = np.zeros(len(X_test))

    y_pred_cnn_lstm = np.argmax(disease_probs, axis=1)
    y_true          = np.argmax(y_test["disease_classification"], axis=1)

    cnn_lstm_metrics = {
        "accuracy"  : accuracy_score(y_true, y_pred_cnn_lstm),
        "precision" : precision_score(y_true, y_pred_cnn_lstm, average="macro", zero_division=0),
        "recall"    : recall_score(y_true, y_pred_cnn_lstm, average="macro", zero_division=0),
        "f1"        : f1_score(y_true, y_pred_cnn_lstm, average="macro", zero_division=0),
        "y_pred"    : y_pred_cnn_lstm,
        "y_true"    : y_true,
    }

    # Per-class F1
    cnn_lstm_per_class_f1 = f1_score(
        y_true, y_pred_cnn_lstm, average=None, zero_division=0
    )

    # ── Static CNN Baseline ───────────────────────────────────
    _print_banner("Static CNN Baseline — Training & Evaluation")

    baseline_metrics = train_and_evaluate_baseline(
        X_train = X_train,
        y_train = y_train_raw["disease_classification"],
        X_val   = X_val,
        y_val   = y_val_raw["disease_classification"],
        X_test  = X_test,
        y_test  = y_test["disease_classification"],
        num_classes = num_classes,
        image_shape = image_shape,
        epochs      = 30,
        batch_size  = EVAL_CFG["batch_size"],
    )
    baseline_per_class_f1 = f1_score(
        baseline_metrics["y_true"],
        baseline_metrics["y_pred"],
        average=None,
        zero_division=0,
    )

    # ── Print results (mirroring Report Tables 5.2 & 5.3) ─────
    _print_banner("RESULTS — Report Table 5.2")
    print(f"\n{'Model':<35} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 75)
    for name, m in [("Static CNN Baseline", baseline_metrics),
                    ("CNN-LSTM (Proposed)",  cnn_lstm_metrics)]:
        print(f"{name:<35} {m['accuracy']*100:>9.2f}% {m['precision']*100:>9.2f}% "
              f"{m['recall']*100:>9.2f}% {m['f1']*100:>9.2f}%")

    delta_acc = (cnn_lstm_metrics["accuracy"] - baseline_metrics["accuracy"]) * 100
    print(f"\n  Δ Accuracy (CNN-LSTM − Baseline): {delta_acc:+.2f}%")

    _print_banner("RESULTS — Report Table 5.3  (Per-Class F1)")
    print(f"\n{'Class':<35} {'Static CNN F1':>14} {'CNN-LSTM F1':>12} {'Δ':>10}")
    print("-" * 73)
    for i, cls in enumerate(class_names[:len(cnn_lstm_per_class_f1)]):
        b = baseline_per_class_f1[i] if i < len(baseline_per_class_f1) else 0.0
        c = cnn_lstm_per_class_f1[i]
        print(f"{cls:<35} {b:>14.4f} {c:>12.4f} {c - b:>+10.4f}")
    print(f"\n{'Macro Average':<35} {baseline_metrics['f1']:>14.4f} "
          f"{cnn_lstm_metrics['f1']:>12.4f} "
          f"{cnn_lstm_metrics['f1'] - baseline_metrics['f1']:>+10.4f}")

    print("\nDetailed CNN-LSTM classification report:")
    print(classification_report(
        y_true, y_pred_cnn_lstm,
        target_names=class_names[:num_classes],
        digits=4,
        zero_division=0,
    ))

    # ── Save JSON summary ─────────────────────────────────────
    summary = {
        "baseline":  {k: float(v) if isinstance(v, (float, np.floating)) else None
                      for k, v in baseline_metrics.items() if k not in ("y_pred", "y_true")},
        "cnn_lstm":  {k: float(v) if isinstance(v, (float, np.floating)) else None
                      for k, v in cnn_lstm_metrics.items() if k not in ("y_pred", "y_true")},
        "per_class_f1_cnn_lstm": dict(zip(class_names, cnn_lstm_per_class_f1.tolist())),
    }
    with open(os.path.join(EVAL_CFG["output_dir"], "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ── Plots ─────────────────────────────────────────────────
    _print_banner("GENERATING PLOTS")

    plot_confusion_matrix(
        y_true, y_pred_cnn_lstm, class_names[:num_classes],
        "CNN-LSTM Confusion Matrix",
        EVAL_CFG["output_dir"],
        "confusion_matrix_cnn_lstm.png",
    )
    plot_confusion_matrix(
        baseline_metrics["y_true"], baseline_metrics["y_pred"],
        class_names[:num_classes],
        "Static CNN Baseline Confusion Matrix",
        EVAL_CFG["output_dir"],
        "confusion_matrix_baseline.png",
    )

    if len(cnn_lstm_per_class_f1) == len(baseline_per_class_f1):
        plot_per_class_f1_comparison(
            baseline_per_class_f1, cnn_lstm_per_class_f1,
            class_names[:num_classes], EVAL_CFG["output_dir"],
        )

    if EVAL_CFG["multi_task"]:
        true_sev  = y_test["severity_score"].flatten()
        true_prog = y_test["progression_rate"].flatten()
        plot_regression_scatter(
            true_sev, pred_severity, true_prog, pred_prograte,
            EVAL_CFG["output_dir"],
        )

    # Training history curves
    if Path(EVAL_CFG["history_path"]).exists():
        with open(EVAL_CFG["history_path"]) as f:
            history = json.load(f)
        plot_training_history(history, EVAL_CFG["output_dir"])
    else:
        print("  [INFO] No training history file found — skipping dynamics plot.")

    plot_comparison_table(baseline_metrics, cnn_lstm_metrics, EVAL_CFG["output_dir"])

    _print_banner("EVALUATION COMPLETE")
    print(f"All outputs saved to: {EVAL_CFG['output_dir']}/")


if __name__ == "__main__":
    main()
