"""
CineVision AI - Prediction engine for movie success forecasting.
"""

import time
from typing import Any, Dict, Optional

import joblib
import numpy as np

from utils.helpers import (
    ENCODER_PATH,
    FEATURE_NAMES_PATH,
    METRICS_PATH,
    MODEL_PATH,
    SCALER_PATH,
    models_exist,
)
from utils.preprocessing import prepare_single_prediction


class MoviePredictor:
    """
    Encapsulates model loading, inference, and revenue/ROI estimation.
    """

    def __init__(self):
        self.model = None
        self.encoders = None
        self.scaler = None
        self.feature_cols = None
        self.metrics = None
        self.target_encoder = None
        self._loaded = False

    def load(self) -> bool:
        """Load all model artifacts from disk."""
        if not models_exist():
            return False
        try:
            self.model = joblib.load(MODEL_PATH)
            self.encoders = joblib.load(ENCODER_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            self.feature_cols = joblib.load(FEATURE_NAMES_PATH)
            self.metrics = joblib.load(METRICS_PATH)
            self.target_encoder = self.metrics.get("target_encoder")
            self._loaded = True
            return True
        except Exception:
            self._loaded = False
            return False

    @property
    def is_ready(self) -> bool:
        return self._loaded

    def _estimate_revenue(self, input_data: dict, category: str) -> float:
        """
        Estimate box office revenue based on input features and predicted category.
        Uses a heuristic model calibrated to category tiers.
        """
        budget = float(input_data.get("budget", 1_000_000))
        marketing = float(input_data.get("marketing_budget", budget * 0.25))
        lead_pop = float(input_data.get("lead_actor_popularity", 50))
        screens = int(input_data.get("num_screens", 2000))
        franchise = input_data.get("franchise", "No")
        ott = input_data.get("ott_release", "No")
        competition = input_data.get("competition_level", "Medium")

        category_multipliers = {
            "Flop": 0.7,
            "Average": 1.5,
            "Hit": 3.0,
            "Super Hit": 5.5,
            "Blockbuster": 9.0,
        }
        base = category_multipliers.get(category, 2.0)

        star_factor = 1 + (lead_pop / 100) * 0.4
        screen_factor = 1 + (screens - 2500) / 8000
        franchise_boost = 1.35 if franchise == "Yes" else 1.0
        ott_penalty = 0.88 if ott == "Yes" else 1.0
        marketing_factor = 1 + (marketing / max(budget, 1)) * 0.25
        competition_map = {"Low": 1.12, "Medium": 1.0, "High": 0.88}

        revenue = (
            budget * base * star_factor * screen_factor
            * franchise_boost * ott_penalty * marketing_factor
            * competition_map.get(competition, 1.0)
        )
        return max(revenue, budget * 0.3)

    def predict(self, input_data: dict) -> Dict[str, Any]:
        """
        Run full prediction pipeline on input movie features.
        Returns category, probabilities, revenue, ROI, and confidence.
        """
        if not self._loaded:
            if not self.load():
                raise RuntimeError(
                    "Model not found. Please run 'python train_model.py' first."
                )

        start = time.perf_counter()

        X = prepare_single_prediction(
            input_data, self.encoders, self.scaler, self.feature_cols
        )

        # Raw prediction
        pred_encoded = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]

        # Decode category label
        if self.target_encoder is not None:
            category = self.target_encoder.inverse_transform([pred_encoded])[0]
            class_labels = self.target_encoder.classes_
        else:
            class_labels = self.metrics.get("class_labels", [])
            category = class_labels[pred_encoded] if pred_encoded < len(class_labels) else "Average"

        category_probabilities = {
            label: float(prob) for label, prob in zip(class_labels, proba)
        }

        success_probability = float(max(proba))
        confidence_score = float(success_probability)

        predicted_revenue = self._estimate_revenue(input_data, category)
        budget = float(input_data.get("budget", 1))
        roi = ((predicted_revenue - budget) / budget) * 100

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "category": category,
            "success_probability": success_probability,
            "predicted_revenue": predicted_revenue,
            "roi": roi,
            "confidence_score": confidence_score,
            "category_probabilities": category_probabilities,
            "prediction_time_ms": elapsed_ms,
        }


# Module-level singleton for reuse across Streamlit pages
_predictor: Optional[MoviePredictor] = None


def get_predictor() -> MoviePredictor:
    """Return shared MoviePredictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = MoviePredictor()
        _predictor.load()
    return _predictor


def predict_movie(input_data: dict) -> Dict[str, Any]:
    """Convenience function for one-shot prediction."""
    return get_predictor().predict(input_data)
