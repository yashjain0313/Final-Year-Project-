"""
Module 1: Crop Recommendation Engine (CRE) — Training Script
=============================================================
Implements Report Section 4.2.3 / Listing 4.2 exactly.

Model: Gradient Boosting Classifier with GridSearchCV
Dataset: Kaggle Crop Recommendation Dataset (2200 samples, 22 crops)
Split: 80 % train | 20 % test (stratified)

Result targets (Report Table 5.1):
    GradientBoosting  → 99.09 % accuracy   (best of all baselines)
    RandomForest      → 97.05 %
    XGBoost           → 96.59 %

Artefacts saved:
    models/crop_recommendation/gradient_boosting_crop.pkl
    models/crop_recommendation/standard_scaler.pkl
    models/crop_recommendation/label_encoder.pkl
    models/crop_recommendation/eval_results/

Usage:
    cd data_ml/notebooks/crop_recomendation
    python train_cre.py

Author: AgroSmart Team
Report: Sections 4.2.3, 5.2
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (
    train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

# Optional — install xgboost if available for full Table 5.1 reproduction
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[INFO] xgboost not installed — XGBoost baseline will be skipped.")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
_HERE         = Path(__file__).parent
DATASET_PATH  = str(_HERE / "../../datasets/crop_recommendation/Crop_recommendation.csv")
MODEL_DIR     = str(_HERE / "../../models/crop_recommendation")
EVAL_DIR      = os.path.join(MODEL_DIR, "eval_results")
MODEL_PATH    = os.path.join(MODEL_DIR, "gradient_boosting_crop.pkl")
SCALER_PATH   = os.path.join(MODEL_DIR, "standard_scaler.pkl")
ENCODER_PATH  = os.path.join(MODEL_DIR, "label_encoder.pkl")

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET   = "label"
RANDOM_SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR,  exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prepare(csv_path: str):
    """
    Load the Kaggle Crop Recommendation CSV and apply standard preprocessing.

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test,
        label_encoder, scaler, class_names
    """
    print(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"  Shape      : {df.shape}")
    print(f"  Crops      : {df[TARGET].nunique()}")
    print(f"  Missing    : {df.isnull().sum().sum()}")
    print(f"  Crop list  : {sorted(df[TARGET].unique())}\n")

    X = df[FEATURES].values
    y = df[TARGET].values

    # Encode string labels → integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Stratified 80/20 split (report §5.1.1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y_encoded,
    )
    print(f"Train samples : {len(X_train)}")
    print(f"Test  samples : {len(X_test)}\n")

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return (
        X_train_scaled, X_test_scaled,
        y_train, y_test,
        label_encoder, scaler,
        label_encoder.classes_,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Proposed model: Gradient Boosting + GridSearchCV  (Report Listing 4.2)
# ─────────────────────────────────────────────────────────────────────────────

def train_gradient_boosting(X_train, y_train):
    """
    Train a Gradient Boosting Classifier with hyperparameter search.
    Matches Report Listing 4.2 exactly.

    Returns:
        best_estimator, grid_search object
    """
    print("=" * 55)
    print("  Gradient Boosting — GridSearchCV (Report Listing 4.2)")
    print("=" * 55)

    param_grid = {
        "n_estimators"    : [100, 200, 300],
        "learning_rate"   : [0.05, 0.1, 0.2],
        "max_depth"       : [3, 4, 5],
        "min_samples_split": [2, 5],
        "subsample"       : [0.8, 1.0],
    }

    base_model = GradientBoostingClassifier(random_state=RANDOM_SEED)

    grid_search = GridSearchCV(
        estimator  = base_model,
        param_grid = param_grid,
        cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
        scoring    = "accuracy",
        n_jobs     = -1,
        verbose    = 1,
        refit      = True,
    )

    t0 = time.time()
    grid_search.fit(X_train, y_train)
    elapsed = (time.time() - t0) / 60

    print(f"\n  Best params    : {grid_search.best_params_}")
    print(f"  Best CV score  : {grid_search.best_score_ * 100:.2f}%")
    print(f"  Search time    : {elapsed:.1f} min\n")

    return grid_search.best_estimator_, grid_search


# ─────────────────────────────────────────────────────────────────────────────
# Baseline models  (Report Table 5.1 comparison)
# ─────────────────────────────────────────────────────────────────────────────

def train_baselines(X_train, y_train, X_test, y_test, class_names):
    """
    Train & evaluate baseline models for Table 5.1 reproduction.

    Models:
        • Random Forest
        • XGBoost (if installed)
        • k-NN, SVM, Decision Tree, Naïve Bayes (lightweight quick baselines)

    Returns:
        dict mapping model_name -> metric_dict
    """
    from sklearn.neighbors      import KNeighborsClassifier
    from sklearn.svm            import SVC
    from sklearn.tree           import DecisionTreeClassifier
    from sklearn.naive_bayes    import GaussianNB

    baselines = {
        "k-Nearest Neighbours"  : KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": SVC(kernel="rbf", C=10, gamma="scale",
                                      probability=True, random_state=RANDOM_SEED),
        "Decision Tree"         : DecisionTreeClassifier(random_state=RANDOM_SEED),
        "Naïve Bayes"           : GaussianNB(),
        "Random Forest"         : RandomForestClassifier(
                                      n_estimators=200, random_state=RANDOM_SEED,
                                      n_jobs=-1),
    }
    if XGBOOST_AVAILABLE:
        baselines["XGBoost"] = XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=RANDOM_SEED, n_jobs=-1,
        )

    results = {}
    print("\n── Baseline model evaluation ────────────────────────────")
    print(f"{'Model':<30} {'Accuracy':>10} {'F1-Macro':>10}")
    print("-" * 52)

    for name, clf in baselines.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
        results[name] = {
            "accuracy" : acc,
            "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall"   : recall_score(y_test, y_pred, average="macro", zero_division=0),
            "f1"       : f1,
        }
        print(f"  {name:<28} {acc*100:>9.2f}% {f1*100:>9.2f}%")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, class_names, model_name="Gradient Boosting"):
    """Full evaluation: accuracy + classification report + confusion matrix."""
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"\n{'='*55}")
    print(f"  {model_name} — Test Set Results")
    print(f"{'='*55}")
    print(f"  Accuracy  : {acc  * 100:.2f}%")
    print(f"  Precision : {prec * 100:.2f}%")
    print(f"  Recall    : {rec  * 100:.2f}%")
    print(f"  F1-Score  : {f1   * 100:.2f}%")
    print(f"\nClassification Report:\n")
    print(classification_report(
        y_test, y_pred, target_names=class_names, digits=4, zero_division=0
    ))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "y_pred": y_pred}


def plot_confusion_matrix(y_test, y_pred, class_names, output_dir):
    """Annotated confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(max(12, len(class_names)), max(10, len(class_names) - 2)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                linewidths=0.5)
    ax.set_title("CRE — Gradient Boosting Confusion Matrix", fontsize=14, pad=15)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix_cre.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix → {path}")


def plot_model_comparison(baseline_results: dict, gb_metrics: dict, output_dir: str):
    """
    Horizontal bar chart: accuracy comparison of all models.
    Reproduces Report Figure 5.1.
    """
    all_models = dict(baseline_results)
    all_models["Gradient Boosting (Proposed)"] = gb_metrics

    # Sort by accuracy ascending (so GB is at top)
    sorted_items = sorted(all_models.items(), key=lambda x: x[1]["accuracy"])
    names  = [k for k, _ in sorted_items]
    values = [v["accuracy"] * 100 for _, v in sorted_items]
    colors = ["#90CAF9"] * (len(names) - 1) + ["#1565C0"]  # highlight GB

    fig, ax = plt.subplots(figsize=(10, max(5, len(names) * 0.7)))
    bars = ax.barh(names, values, color=colors, edgecolor="white", height=0.6)

    # Annotate values
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=9,
        )

    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Crop Recommendation Engine — Model Comparison\n(Report Figure 5.1)", pad=15)
    ax.set_xlim(70, 102)
    ax.axvline(99.09, color="red", linestyle="--", linewidth=1, alpha=0.5,
               label="Report target (99.09%)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "model_accuracy_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Model comparison chart → {path}")


def plot_feature_importance(model, output_dir):
    """Bar chart of Gradient Boosting feature importances."""
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(FEATURES)),
           importances[indices],
           color="#1565C0", alpha=0.85)
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels([FEATURES[i] for i in indices])
    ax.set_ylabel("Importance")
    ax.set_title("Gradient Boosting — Feature Importances")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Feature importance plot → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  CROP RECOMMENDATION ENGINE — TRAINING")
    print("  Report: Section 4.2.3 / Listing 4.2")
    print("=" * 55 + "\n")

    # ── Load data ─────────────────────────────────────────────
    (X_train_sc, X_test_sc, y_train, y_test,
     label_encoder, scaler, class_names) = load_and_prepare(DATASET_PATH)

    # ── Baseline models (Table 5.1) ───────────────────────────
    baseline_results = train_baselines(
        X_train_sc, y_train, X_test_sc, y_test, class_names
    )

    # ── Proposed: Gradient Boosting + GridSearchCV ─────────────
    best_model, grid_search = train_gradient_boosting(X_train_sc, y_train)

    # ── Evaluate ──────────────────────────────────────────────
    gb_metrics = evaluate_model(
        best_model, X_test_sc, y_test, class_names,
        model_name="Gradient Boosting (Proposed)",
    )

    # ── Print Table 5.1 ───────────────────────────────────────
    print("\n── Report Table 5.1 — Full Comparison ───────────────")
    print(f"{'Model':<35} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print("-" * 71)
    for name, m in baseline_results.items():
        print(f"  {name:<33} {m['accuracy']*100:>7.2f}% {m['precision']*100:>7.2f}% "
              f"{m['recall']*100:>7.2f}% {m['f1']*100:>7.2f}%")
    print(f"  {'Gradient Boosting (Proposed)':<33} "
          f"{gb_metrics['accuracy']*100:>7.2f}% {gb_metrics['precision']*100:>7.2f}% "
          f"{gb_metrics['recall']*100:>7.2f}% {gb_metrics['f1']*100:>7.2f}%")

    # ── Serialise artefacts (Report Listing 4.2, lines 54-57) ─
    joblib.dump(best_model,    MODEL_PATH)
    joblib.dump(scaler,        SCALER_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)
    print(f"\n✅ Artefacts saved:")
    print(f"   {MODEL_PATH}")
    print(f"   {SCALER_PATH}")
    print(f"   {ENCODER_PATH}")

    # ── Save evaluation JSON ───────────────────────────────────
    summary = {
        "gradient_boosting": {k: float(v) for k, v in gb_metrics.items()
                              if k != "y_pred"},
        "baselines": {
            name: {k: float(v) for k, v in m.items()}
            for name, m in baseline_results.items()
        },
        "best_params": grid_search.best_params_,
    }
    with open(os.path.join(EVAL_DIR, "cre_eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ── Plots ─────────────────────────────────────────────────
    print("\n── Generating plots ─────────────────────────────────")
    plot_confusion_matrix(y_test, gb_metrics["y_pred"], class_names, EVAL_DIR)
    plot_model_comparison(baseline_results, gb_metrics, EVAL_DIR)
    plot_feature_importance(best_model, EVAL_DIR)

    print("\n" + "=" * 55)
    print("  TRAINING COMPLETE")
    print("=" * 55)


if __name__ == "__main__":
    main()
