"""
Inference Module — Disease Progression Detection Module (DPDM)
==============================================================
This is the production-facing prediction interface consumed by
`backend/disease_progression_api.py`.

Responsibilities:
  • Load the saved CNN-LSTM model once at startup (low-latency reuse).
  • Accept a list of raw image arrays (3-5 frames) or a single 5-frame
    numpy sequence.
  • Return a structured result dict with disease, severity, progression
    rate, and treatment recommendation — mirroring the API response schema
    described in the project report (ARCHITECTURE.md §API Workflow).

Usage (standalone):
    python predict_dpdm.py --images path/day1.jpg path/day2.jpg path/day3.jpg

Usage (from Flask API):
    from predict_dpdm import DPDMPredictor
    predictor = DPDMPredictor(model_path="...", class_names_path="...")
    result = predictor.predict_from_images([img1, img2, img3])

Author: AgroSmart Team
Report: Sections 4.4, API Workflow
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Union
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH       = "models/dpdm/cnn_lstm_disease_final.keras"
DEFAULT_CLASS_NAMES_PATH = "models/dpdm/class_names.npy"
TARGET_SIZE              = (224, 224)
SEQUENCE_LENGTH          = 5     # model always expects 5 frames


# ─────────────────────────────────────────────────────────────────────────────
# Severity / progression helpers  (mirroring disease_progression_api.py)
# ─────────────────────────────────────────────────────────────────────────────

def _severity_level(score: float) -> str:
    if score < 0.2:  return "Healthy"
    if score < 0.4:  return "Mild"
    if score < 0.6:  return "Moderate"
    if score < 0.8:  return "Severe"
    return "Critical"


def _progression_status(rate: float) -> str:
    if rate < 0.05:  return "Stable"
    if rate < 0.15:  return "Slowly Worsening"
    if rate < 0.25:  return "Moderately Worsening"
    return "Rapidly Worsening"


def _urgency(severity: float, rate: float) -> str:
    if severity > 0.7 or rate > 0.20: return "high"
    if severity > 0.4 or rate > 0.10: return "medium"
    return "low"


# Disease-specific treatment advice (extends generic recommendations)
DISEASE_TREATMENTS: Dict[str, List[str]] = {
    "blight":    ["Remove affected leaves", "Apply Mancozeb 75% WP at 2 g/L"],
    "late":      ["Apply Metalaxyl-M + Mancozeb at 2.5 g/L immediately",
                  "Remove severely infected material; burn or bury residues"],
    "mold":      ["Improve canopy airflow", "Apply copper-based fungicide"],
    "rust":      ["Apply sulfur-based fungicide (wettable sulfur 80%)"],
    "spot":      ["Avoid overhead irrigation", "Apply Chlorothalonil at label rate"],
    "bacterial": ["Apply copper hydroxide spray", "Avoid working in wet fields"],
    "mosaic":    ["Remove infected plants to prevent vector spread",
                  "Control aphid populations with insecticidal soap"],
    "healthy":   ["Maintain good agricultural practices", "Monitor weekly"],
}


def _build_recommendation(disease: str, severity: float, rate: float) -> Dict:
    urg = _urgency(severity, rate)
    advice = {
        "urgency":    urg,
        "actions":    [],
        "treatments": [],
        "monitoring": [],
    }
    if urg == "high":
        advice["actions"]    = ["Immediate intervention required",
                                 "Consult agricultural expert within 24 h"]
        advice["treatments"] = ["Apply appropriate fungicide / pesticide"]
        advice["monitoring"] = ["Monitor daily"]
    elif urg == "medium":
        advice["actions"]    = ["Take action within 1-2 days"]
        advice["treatments"] = ["Apply preventive treatment"]
        advice["monitoring"] = ["Monitor every 2 days"]
    else:
        advice["actions"]    = ["Continue monitoring"]
        advice["treatments"] = ["Maintain good agricultural practices"]
        advice["monitoring"] = ["Monitor weekly"]

    # Disease-specific additions
    d_lower = disease.lower()
    for keyword, tips in DISEASE_TREATMENTS.items():
        if keyword in d_lower:
            advice["treatments"].extend(tips)
            break

    return advice


# ─────────────────────────────────────────────────────────────────────────────
# Core predictor class
# ─────────────────────────────────────────────────────────────────────────────

class DPDMPredictor:
    """
    Singleton-friendly inference wrapper for the CNN-LSTM disease progression model.

    Thread-safety note:
        TensorFlow models are not thread-safe for concurrent `predict()` calls.
        Use one instance per worker thread or protect calls with a mutex.
    """

    def __init__(
        self,
        model_path:       str = DEFAULT_MODEL_PATH,
        class_names_path: str = DEFAULT_CLASS_NAMES_PATH,
        image_size:       tuple = TARGET_SIZE,
    ):
        import tensorflow as tf
        from tensorflow import keras

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Trained model not found: {model_path}\n"
                "Run train_dpdm.py first."
            )
        if not Path(class_names_path).exists():
            raise FileNotFoundError(
                f"Class names file not found: {class_names_path}"
            )

        print(f"[DPDMPredictor] Loading model from: {model_path}")
        self.model       = keras.models.load_model(model_path)
        self.class_names = list(np.load(class_names_path, allow_pickle=True))
        self.image_size  = image_size

        print(f"[DPDMPredictor] ✓ Ready  |  {len(self.class_names)} classes  "
              f"|  sequence length: {SEQUENCE_LENGTH}")

    # ── Public API ────────────────────────────────────────────────────────────

    def predict_from_images(
        self,
        images: List[np.ndarray],
        top_k: int = 3,
    ) -> Dict:
        """
        Run inference from a list of raw image arrays.

        Args:
            images : List of 3–5 BGR or RGB uint8 numpy arrays (H × W × 3).
                     If fewer than 5 are provided the last frame is repeated
                     to reach SEQUENCE_LENGTH (padding strategy used in
                     disease_progression_api.py).
            top_k  : Number of top predictions to include in the response.

        Returns:
            dict — see `_build_response()` for the full schema.
        """
        if len(images) < 3:
            raise ValueError(
                f"At least 3 images are required; {len(images)} provided."
            )

        sequence = self._prepare_sequence(images)
        return self._run_inference(sequence, top_k)

    def predict_from_sequence(
        self,
        sequence: np.ndarray,
        top_k: int = 3,
    ) -> Dict:
        """
        Run inference from a pre-built float32 sequence array.

        Args:
            sequence : float32 array (T, H, W, 3) with values in [0, 1].
            top_k    : Number of top predictions in the response.

        Returns:
            dict — same schema as `predict_from_images`.
        """
        if sequence.ndim != 4 or sequence.shape[0] != SEQUENCE_LENGTH:
            raise ValueError(
                f"Expected sequence shape ({SEQUENCE_LENGTH}, H, W, 3), "
                f"got {sequence.shape}."
            )
        return self._run_inference(sequence, top_k)

    def predict_from_paths(
        self,
        image_paths: List[str],
        top_k: int = 3,
    ) -> Dict:
        """
        Convenience method: load images from file paths then predict.

        Args:
            image_paths : List of 3–5 file paths.
            top_k       : Number of top predictions.

        Returns:
            dict — same schema as `predict_from_images`.
        """
        images = [self._load_image(p) for p in image_paths]
        return self.predict_from_images(images, top_k)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_image(self, path: str) -> np.ndarray:
        """Load image file → uint8 RGB numpy array."""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _prepare_sequence(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Resize, normalise, and pad/trim images to SEQUENCE_LENGTH.

        Returns:
            float32 array  (SEQUENCE_LENGTH, H, W, 3),  values in [0, 1].
        """
        frames = []
        for img in images:
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            resized = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LINEAR)
            frames.append(resized.astype(np.float32) / 255.0)

        # Pad by repeating last frame
        while len(frames) < SEQUENCE_LENGTH:
            frames.append(frames[-1])

        # Trim to SEQUENCE_LENGTH
        frames = frames[:SEQUENCE_LENGTH]

        return np.stack(frames, axis=0)   # (T, H, W, 3)

    def _run_inference(self, sequence: np.ndarray, top_k: int) -> Dict:
        """
        Forward pass + result construction.
        """
        import tensorflow as tf

        batch = np.expand_dims(sequence, 0)   # (1, T, H, W, 3)
        raw = self.model(batch, training=False)

        # Handle both dict output (multi-task) and plain tensor (single-task)
        if isinstance(raw, dict):
            disease_probs = raw["disease_classification"].numpy()[0]
            severity      = float(raw["severity_score"].numpy()[0][0])
            prog_rate     = float(raw["progression_rate"].numpy()[0][0])
        else:
            disease_probs = raw.numpy()[0]
            severity      = 0.0
            prog_rate     = 0.0

        top_idx    = int(np.argmax(disease_probs))
        confidence = float(disease_probs[top_idx])
        disease    = self.class_names[top_idx]

        # Top-K predictions
        top_k_idx = np.argsort(disease_probs)[::-1][:top_k]
        top_k_preds = [
            {"disease": self.class_names[i], "confidence": float(disease_probs[i])}
            for i in top_k_idx
        ]

        return self._build_response(
            disease, confidence, severity, prog_rate, top_k_preds
        )

    def _build_response(
        self,
        disease:     str,
        confidence:  float,
        severity:    float,
        prog_rate:   float,
        top_k_preds: List[Dict],
    ) -> Dict:
        """
        Build the structured result dict (matches API response schema in
        disease_progression_api.py and Report ARCHITECTURE.md §API Workflow).
        """
        return {
            "success": True,
            "analysis": {
                "disease":             disease,
                "confidence":          round(confidence, 4),
                "severity_score":      round(severity, 4),
                "severity_level":      _severity_level(severity),
                "progression_rate":    round(prog_rate, 4),
                "progression_status":  _progression_status(prog_rate),
                "top_predictions":     top_k_preds,
            },
            "recommendation": _build_recommendation(disease, severity, prog_rate),
            "analyzed_at": datetime.now().isoformat(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(
        description="DPDM — CNN-LSTM disease progression prediction"
    )
    parser.add_argument(
        "--images", nargs="+", required=True,
        metavar="IMG",
        help="3–5 leaf image file paths in chronological order.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to trained model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--classes",
        default=DEFAULT_CLASS_NAMES_PATH,
        help=f"Path to class_names.npy (default: {DEFAULT_CLASS_NAMES_PATH})",
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Number of top predictions to display (default: 3)",
    )
    args = parser.parse_args()

    predictor = DPDMPredictor(
        model_path=args.model,
        class_names_path=args.classes,
    )

    import json
    result = predictor.predict_from_paths(args.images, top_k=args.top_k)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _cli()
