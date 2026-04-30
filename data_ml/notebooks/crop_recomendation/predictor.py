"""
Module 1: Crop Recommendation Engine (CRE) — Inference Module
=============================================================
Implements Report Section 4.2.1 / Listing 4.1.

Responsibilities:
  • Load trained Gradient Boosting model + scaler + label encoder at startup.
  • Fetch live weather from OpenWeatherMap API using GPS coordinates.
  • Return top-K ranked crop recommendations with confidence scores.

Usage (standalone):
    python predictor.py --lat 28.7 --lon 77.1 --N 90 --P 42 --K 43 --ph 6.5

Usage (from Flask backend):
    from predictor import CREPredictor
    predictor = CREPredictor()
    results = predictor.recommend_crop(N=90, P=42, K=43, ph=6.5,
                                       lat=28.7, lon=77.1, top_k=5)

Author: AgroSmart Team
Report: Section 4.2.1–4.2.2 / Listing 4.1
"""

import os
import sys
import json
import argparse
import numpy as np
import joblib
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Default paths  (relative to this file)
# ─────────────────────────────────────────────────────────────────────────────
_BASE = Path(__file__).parent
MODEL_PATH   = _BASE / "../../models/crop_recommendation/gradient_boosting_crop.pkl"
SCALER_PATH  = _BASE / "../../models/crop_recommendation/standard_scaler.pkl"
ENCODER_PATH = _BASE / "../../models/crop_recommendation/label_encoder.pkl"

# ── OpenWeatherMap (report §4.2.2) ────────────────────────────────────────────
OWM_API_KEY  = os.environ.get("OWM_API_KEY", "YOUR_OPENWEATHERMAP_API_KEY")
OWM_ENDPOINT = "https://api.openweathermap.org/data/2.5/weather"
OWM_TIMEOUT  = 5   # seconds

# ── Input feature order ────────────────────────────────────────────────────────
FEATURE_NAMES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# ── Validation ranges (from Kaggle dataset statistics) ────────────────────────
FEATURE_RANGES: Dict[str, Tuple[float, float]] = {
    "N"          : (0,    140),
    "P"          : (5,    145),
    "K"          : (5,    205),
    "temperature": (8.0,  43.0),
    "humidity"   : (14.0, 100.0),
    "ph"         : (3.5,  9.9),
    "rainfall"   : (20.0, 298.0),
}


# ─────────────────────────────────────────────────────────────────────────────
# Weather fetch  (Report Listing 4.1, lines 18-34)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_live_weather(lat: float, lon: float) -> Dict[str, float]:
    """
    Fetch current temperature, humidity, and rainfall from OpenWeatherMap.

    Args:
        lat: GPS latitude  (float)
        lon: GPS longitude (float)

    Returns:
        dict with keys 'temperature' (°C), 'humidity' (%), 'rainfall' (mm/h)

    Raises:
        RuntimeError: if the API call fails and no fallback is available.
    """
    if OWM_API_KEY == "YOUR_OPENWEATHERMAP_API_KEY":
        raise RuntimeError(
            "OWM_API_KEY not configured. "
            "Set the OWM_API_KEY environment variable or pass weather manually."
        )

    url = (
        f"{OWM_ENDPOINT}"
        f"?lat={lat}&lon={lon}&appid={OWM_API_KEY}&units=metric"
    )
    try:
        response = requests.get(url, timeout=OWM_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return {
            "temperature": float(data["main"]["temp"]),
            "humidity"   : float(data["main"]["humidity"]),
            "rainfall"   : float(data.get("rain", {}).get("1h", 0.0)),
        }
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"OpenWeatherMap API error: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Predictor class  (Report Listing 4.1, lines 36-60)
# ─────────────────────────────────────────────────────────────────────────────

class CREPredictor:
    """
    Crop Recommendation Engine — production inference wrapper.

    Loads the three serialised artefacts (model, scaler, encoder) once at
    startup and exposes a simple `recommend_crop()` interface consumed by
    the Flask backend route.
    """

    def __init__(
        self,
        model_path:   str = str(MODEL_PATH),
        scaler_path:  str = str(SCALER_PATH),
        encoder_path: str = str(ENCODER_PATH),
    ):
        for label, path in [
            ("Model",         model_path),
            ("Scaler",        scaler_path),
            ("Label encoder", encoder_path),
        ]:
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"{label} not found: {path}\n"
                    "Run train_cre.py first."
                )

        self.model         = joblib.load(model_path)
        self.scaler        = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)
        print(f"[CREPredictor] ✓ Ready  |  {len(self.label_encoder.classes_)} crops")

    # ── Public API ────────────────────────────────────────────────────────────

    def recommend_crop(
        self,
        nitrogen:     float,
        phosphorus:   float,
        potassium:    float,
        ph:           float,
        lat:          Optional[float] = None,
        lon:          Optional[float] = None,
        temperature:  Optional[float] = None,
        humidity:     Optional[float] = None,
        rainfall:     Optional[float] = None,
        top_k:        int = 5,
    ) -> List[Dict]:
        """
        Return top-K ranked crop recommendations with confidence scores.
        Matches Report Listing 4.1 exactly.

        Args:
            nitrogen, phosphorus, potassium, ph: soil parameters.
            lat, lon    : GPS coordinates — used to fetch live weather.
                          Omit if passing temperature/humidity/rainfall directly.
            temperature : °C  (overrides live fetch if provided).
            humidity    : %   (overrides live fetch if provided).
            rainfall    : mm/h (overrides live fetch if provided).
            top_k       : Number of ranked recommendations to return.

        Returns:
            List of dicts: [{"crop": str, "confidence": float}, ...]
        """
        # ── Weather resolution ────────────────────────────────
        if temperature is None or humidity is None or rainfall is None:
            if lat is None or lon is None:
                raise ValueError(
                    "Provide either (lat, lon) for live weather fetch, "
                    "or explicit (temperature, humidity, rainfall)."
                )
            weather = fetch_live_weather(lat, lon)
            temperature = weather["temperature"]
            humidity    = weather["humidity"]
            rainfall    = weather["rainfall"]

        # ── Validation ────────────────────────────────────────
        inputs = {
            "N"          : nitrogen,
            "P"          : phosphorus,
            "K"          : potassium,
            "temperature": temperature,
            "humidity"   : humidity,
            "ph"         : ph,
            "rainfall"   : rainfall,
        }
        self._validate(inputs)

        # ── Inference (Report Listing 4.1, lines 45-59) ───────
        feature_vector = np.array([[
            nitrogen, phosphorus, potassium,
            temperature, humidity, ph, rainfall,
        ]])
        feature_scaled   = self.scaler.transform(feature_vector)
        probabilities    = self.model.predict_proba(feature_scaled)[0]
        top_k_indices    = np.argsort(probabilities)[::-1][:top_k]

        recommendations = [
            {
                "crop"      : str(self.label_encoder.classes_[i]),
                "confidence": round(float(probabilities[i]), 4),
            }
            for i in top_k_indices
        ]
        return recommendations

    def recommend_with_metadata(
        self,
        nitrogen: float, phosphorus: float, potassium: float, ph: float,
        lat: float = None, lon: float = None,
        temperature: float = None, humidity: float = None, rainfall: float = None,
        top_k: int = 5,
    ) -> Dict:
        """
        Same as recommend_crop but wraps the response with metadata —
        used by the Flask API route.
        """
        # Resolve weather first so we can echo it back
        if temperature is None or humidity is None or rainfall is None:
            weather = fetch_live_weather(lat, lon)
            temperature = weather["temperature"]
            humidity    = weather["humidity"]
            rainfall    = weather["rainfall"]

        recommendations = self.recommend_crop(
            nitrogen, phosphorus, potassium, ph,
            temperature=temperature, humidity=humidity, rainfall=rainfall,
            top_k=top_k,
        )
        return {
            "success"        : True,
            "recommendations": recommendations,
            "input_used": {
                "nitrogen"   : nitrogen,
                "phosphorus" : phosphorus,
                "potassium"  : potassium,
                "ph"         : ph,
                "temperature": round(temperature, 2),
                "humidity"   : round(humidity,    2),
                "rainfall"   : round(rainfall,    3),
            },
            "weather_source" : "live_owm" if lat is not None else "manual",
            "gps"            : {"lat": lat, "lon": lon} if lat is not None else None,
            "generated_at"   : datetime.now().isoformat(),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _validate(self, inputs: Dict[str, float]):
        """Range-check input features and raise ValueError if out of bounds."""
        for feat, val in inputs.items():
            lo, hi = FEATURE_RANGES[feat]
            if not (lo <= val <= hi):
                raise ValueError(
                    f"'{feat}' value {val} is out of expected range [{lo}, {hi}]."
                )


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(
        description="CRE — Gradient Boosting Crop Recommendation"
    )
    parser.add_argument("--N",   type=float, required=True,  help="Nitrogen (0-140)")
    parser.add_argument("--P",   type=float, required=True,  help="Phosphorus (5-145)")
    parser.add_argument("--K",   type=float, required=True,  help="Potassium (5-205)")
    parser.add_argument("--ph",  type=float, required=True,  help="pH (3.5-9.9)")
    parser.add_argument("--lat", type=float, default=None,   help="GPS latitude")
    parser.add_argument("--lon", type=float, default=None,   help="GPS longitude")
    parser.add_argument("--temperature", type=float, default=None, help="°C (manual override)")
    parser.add_argument("--humidity",    type=float, default=None, help="% (manual override)")
    parser.add_argument("--rainfall",    type=float, default=None, help="mm/h (manual override)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of recommendations")
    args = parser.parse_args()

    predictor = CREPredictor()
    result = predictor.recommend_with_metadata(
        nitrogen    = args.N,
        phosphorus  = args.P,
        potassium   = args.K,
        ph          = args.ph,
        lat         = args.lat,
        lon         = args.lon,
        temperature = args.temperature,
        humidity    = args.humidity,
        rainfall    = args.rainfall,
        top_k       = args.top_k,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _cli()
