"""
CineVision AI - Data preprocessing, feature engineering, and dataset management.
"""

import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils.helpers import DATASET_PATH, PROJECT_ROOT

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Drama",
    "Fantasy", "Horror", "Mystery", "Romance", "Sci-Fi", "Thriller",
]
LANGUAGES = ["English", "Hindi", "Spanish", "French", "Japanese", "Korean", "Tamil", "Telugu"]
COUNTRIES = ["USA", "India", "UK", "France", "Japan", "South Korea", "Germany", "Canada"]
CERTIFICATIONS = ["G", "PG", "PG-13", "R", "UA", "A", "U"]
COMPETITION_LEVELS = ["Low", "Medium", "High"]
CATEGORIES = ["Flop", "Average", "Hit", "Super Hit", "Blockbuster"]

CATEGORICAL_COLS = [
    "genre", "language", "country", "certification_rating",
    "competition_level", "franchise", "ott_release",
    "budget_category", "runtime_category", "release_season",
]

NUMERIC_COLS = [
    "budget", "runtime", "release_month", "director_experience",
    "lead_actor_popularity", "supporting_cast_rating", "marketing_budget",
    "production_house_rating", "num_screens", "genre_popularity",
]

# Outcome columns used for target derivation only — NOT model inputs
OUTCOME_COLS = ["revenue", "roi", "revenue_budget_ratio"]

FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS

TARGET_COL = "category"


def _season_from_month(month: int) -> str:
    """Map release month to season name."""
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


def _budget_category(budget: float) -> str:
    """Categorize budget into tiers."""
    if budget < 10_000_000:
        return "Low"
    if budget < 50_000_000:
        return "Medium"
    if budget < 100_000_000:
        return "High"
    return "Blockbuster"


def _runtime_category(runtime: int) -> str:
    """Categorize runtime into tiers."""
    if runtime < 90:
        return "Short"
    if runtime <= 120:
        return "Standard"
    if runtime <= 150:
        return "Long"
    return "Epic"


def _compute_success_score(row: dict) -> float:
    """Compute a latent success score from pre-release features (0-100)."""
    genre_pop = row.get("genre_popularity", 70)
    budget = max(row.get("budget", 1), 1)
    marketing_ratio = min(row.get("marketing_budget", 0) / budget, 0.5)

    score = (
        genre_pop * 0.08
        + row.get("lead_actor_popularity", 50) * 0.12
        + row.get("supporting_cast_rating", 5) * 2.0
        + min(row.get("director_experience", 10), 25) * 0.5
        + row.get("production_house_rating", 5) * 1.5
        + marketing_ratio * 20
        + (row.get("num_screens", 2000) - 500) / 4500 * 12
    )
    if row.get("franchise") == "Yes":
        score += 6
    if row.get("ott_release") == "Yes":
        score -= 3
    competition_penalty = {"Low": 4, "Medium": 0, "High": -5}
    score += competition_penalty.get(row.get("competition_level", "Medium"), 0)
    season_boost = {"Summer": 3, "Fall": 2, "Winter": 0, "Spring": 1}
    score += season_boost.get(row.get("release_season", "Summer"), 0)
    return float(np.clip(score, 0, 100))


def _category_from_score(score: float, rng) -> str:
    """Map success score to category with slight randomness."""
    noise = rng.normal(0, 6)
    adjusted = score + noise
    if adjusted < 30:
        return "Flop"
    if adjusted < 45:
        return "Average"
    if adjusted < 58:
        return "Hit"
    if adjusted < 72:
        return "Super Hit"
    return "Blockbuster"


def generate_synthetic_dataset(n_samples: int = 2500, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic movie dataset with engineered features.
    Used when no external dataset is available.
    """
    rng = np.random.default_rng(seed)
    records = []

    genre_popularity_map = {
        "Action": 85, "Adventure": 78, "Animation": 72, "Comedy": 80,
        "Crime": 65, "Drama": 70, "Fantasy": 75, "Horror": 68,
        "Mystery": 62, "Romance": 74, "Sci-Fi": 77, "Thriller": 73,
    }

    for i in range(n_samples):
        genre = rng.choice(GENRES)
        budget = float(rng.lognormal(mean=16.5, sigma=0.8))
        budget = np.clip(budget, 500_000, 350_000_000)

        runtime = int(rng.integers(80, 180))
        release_month = int(rng.integers(1, 13))
        director_exp = int(rng.integers(1, 30))
        lead_pop = float(rng.uniform(20, 100))
        cast_rating = float(rng.uniform(3, 10))
        marketing = budget * float(rng.uniform(0.1, 0.5))
        prod_rating = float(rng.uniform(4, 10))
        competition = rng.choice(COMPETITION_LEVELS, p=[0.3, 0.45, 0.25])
        screens = int(rng.integers(500, 5000))
        franchise = rng.choice(["Yes", "No"], p=[0.25, 0.75])
        ott = rng.choice(["Yes", "No"], p=[0.4, 0.6])

        success_score = _compute_success_score({
            "genre_popularity": genre_popularity_map[genre],
            "lead_actor_popularity": lead_pop,
            "supporting_cast_rating": cast_rating,
            "director_experience": director_exp,
            "production_house_rating": prod_rating,
            "marketing_budget": marketing,
            "budget": budget,
            "num_screens": screens,
            "franchise": franchise,
            "ott_release": ott,
            "competition_level": competition,
            "release_season": _season_from_month(release_month),
        })

        category = _category_from_score(success_score, rng)

        # Revenue correlates with success score
        category_multiplier = {
            "Flop": 0.6, "Average": 1.4, "Hit": 2.8,
            "Super Hit": 5.0, "Blockbuster": 8.5,
        }
        # Temporary category for revenue; final category assigned via quantiles
        temp_cat = _category_from_score(success_score, rng)
        revenue = budget * category_multiplier[temp_cat] * float(rng.uniform(0.85, 1.15))
        roi = ((revenue - budget) / budget) * 100
        ratio = revenue / budget

        records.append({
            "title": f"Movie_{i + 1}",
            "budget": round(budget, 2),
            "genre": genre,
            "language": rng.choice(LANGUAGES),
            "runtime": runtime,
            "release_month": release_month,
            "director_experience": director_exp,
            "lead_actor_popularity": round(lead_pop, 2),
            "supporting_cast_rating": round(cast_rating, 2),
            "marketing_budget": round(marketing, 2),
            "production_house_rating": round(prod_rating, 2),
            "competition_level": competition,
            "num_screens": screens,
            "franchise": franchise,
            "ott_release": ott,
            "country": rng.choice(COUNTRIES),
            "certification_rating": rng.choice(CERTIFICATIONS),
            "revenue": round(revenue, 2),
            "roi": round(roi, 2),
            "revenue_budget_ratio": round(ratio, 4),
            "budget_category": _budget_category(budget),
            "runtime_category": _runtime_category(runtime),
            "release_season": _season_from_month(release_month),
            "genre_popularity": genre_popularity_map[genre],
            "_success_score": success_score,
        })

    df = pd.DataFrame(records)
    return assign_categories_by_quantile(df)


def assign_categories_by_quantile(df: pd.DataFrame, score_col: str = "_success_score") -> pd.DataFrame:
    """Assign balanced categories using score quantiles."""
    df = df.copy()
    if score_col not in df.columns:
        df[score_col] = df.apply(lambda r: _compute_success_score(r.to_dict()), axis=1)

    quantiles = [0, 0.15, 0.35, 0.55, 0.75, 1.0]
    labels = CATEGORIES
    df[TARGET_COL] = pd.cut(
        df[score_col],
        bins=df[score_col].quantile(quantiles).values,
        labels=labels,
        include_lowest=True,
        duplicates="drop",
    )
    df[TARGET_COL] = df[TARGET_COL].astype(str)
    df.drop(columns=[score_col], inplace=True, errors="ignore")
    return df


def download_tmdb_sample() -> Optional[pd.DataFrame]:
    """
    Attempt to download TMDB 5000 movie dataset from GitHub mirror.
    Returns None if download fails.
    """
    url = (
        "https://raw.githubusercontent.com/duyetdev/brickhouse/master"
        "/data/tmdb_5000_movies.csv"
    )
    try:
        temp_path = PROJECT_ROOT / "dataset" / "_tmdb_temp.csv"
        urllib.request.urlretrieve(url, temp_path)
        raw = pd.read_csv(temp_path)
        temp_path.unlink(missing_ok=True)

        # Map TMDB columns to our schema where possible
        df = pd.DataFrame()
        df["title"] = raw.get("title", raw.index.astype(str))
        df["budget"] = pd.to_numeric(raw.get("budget", 0), errors="coerce").fillna(1_000_000)
        df["budget"] = df["budget"].replace(0, 1_000_000)
        df["revenue"] = pd.to_numeric(raw.get("revenue", 0), errors="coerce").fillna(0)

        rng = np.random.default_rng(42)
        n = len(df)
        df["genre"] = rng.choice(GENRES, n)
        df["language"] = rng.choice(LANGUAGES, n)
        df["runtime"] = pd.to_numeric(raw.get("runtime", 120), errors="coerce").fillna(120).astype(int)
        df["release_month"] = rng.integers(1, 13, n)
        df["director_experience"] = rng.integers(1, 25, n)
        df["lead_actor_popularity"] = rng.uniform(30, 95, n).round(2)
        df["supporting_cast_rating"] = rng.uniform(4, 9, n).round(2)
        df["marketing_budget"] = (df["budget"] * rng.uniform(0.15, 0.4, n)).round(2)
        df["production_house_rating"] = rng.uniform(5, 10, n).round(2)
        df["competition_level"] = rng.choice(COMPETITION_LEVELS, n)
        df["num_screens"] = rng.integers(800, 4500, n)
        df["franchise"] = rng.choice(["Yes", "No"], n, p=[0.2, 0.8])
        df["ott_release"] = rng.choice(["Yes", "No"], n, p=[0.35, 0.65])
        df["country"] = rng.choice(COUNTRIES, n)
        df["certification_rating"] = rng.choice(CERTIFICATIONS, n)

        df = engineer_features(df)
        df["_success_score"] = df.apply(
            lambda r: _compute_success_score(r.to_dict()), axis=1
        )
        df = assign_categories_by_quantile(df)
        return df
    except Exception:
        return None


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to a movie dataframe."""
    df = df.copy()
    df["roi"] = ((df["revenue"] - df["budget"]) / df["budget"].replace(0, 1)) * 100
    df["revenue_budget_ratio"] = df["revenue"] / df["budget"].replace(0, 1)
    df["budget_category"] = df["budget"].apply(_budget_category)
    df["runtime_category"] = df["runtime"].apply(_runtime_category)
    df["release_season"] = df["release_month"].apply(_season_from_month)

    genre_pop = {
        "Action": 85, "Adventure": 78, "Animation": 72, "Comedy": 80,
        "Crime": 65, "Drama": 70, "Fantasy": 75, "Horror": 68,
        "Mystery": 62, "Romance": 74, "Sci-Fi": 77, "Thriller": 73,
    }
    df["genre_popularity"] = df["genre"].map(genre_pop).fillna(70)
    return df


def load_or_create_dataset(force_regenerate: bool = False) -> pd.DataFrame:
    """
    Load movies.csv or create it from external/synthetic sources.
    Performs cleaning: missing values, duplicates, feature engineering.
    """
    if DATASET_PATH.exists() and not force_regenerate:
        df = pd.read_csv(DATASET_PATH)
    else:
        df = download_tmdb_sample()
        if df is None or len(df) < 100:
            df = generate_synthetic_dataset()
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATASET_PATH, index=False)

    df = clean_dataset(df)
    df.to_csv(DATASET_PATH, index=False)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, remove duplicates, and engineer features."""
    df = df.copy()
    df = df.drop_duplicates(subset=["title"], keep="first")

    # Fill numeric missing values with median
    all_numeric = NUMERIC_COLS + OUTCOME_COLS
    for col in all_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing values with mode
    cat_defaults = {
        "genre": "Drama", "language": "English", "country": "USA",
        "certification_rating": "PG-13", "competition_level": "Medium",
        "franchise": "No", "ott_release": "No",
    }
    for col, default in cat_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)

    df = engineer_features(df)

    if TARGET_COL not in df.columns or df[TARGET_COL].isna().any():
        df["_success_score"] = df.apply(
            lambda r: _compute_success_score(r.to_dict()), axis=1
        )
        df = assign_categories_by_quantile(df)

    return df


def encode_features(
    df: pd.DataFrame,
    encoders: Optional[dict] = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Label-encode categorical columns.
    Returns encoded dataframe and encoder dictionary.
    """
    if encoders is None:
        encoders = {}

    df_encoded = df.copy()
    for col in CATEGORICAL_COLS:
        if col not in df_encoded.columns:
            continue
        if fit:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df_encoded[col] = df_encoded[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    return df_encoded, encoders


def scale_features(
    X: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    fit: bool = True,
) -> Tuple[np.ndarray, StandardScaler]:
    """Standard-scale numeric feature matrix."""
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler


def prepare_training_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, dict, StandardScaler, List[str]]:
    """
    Full pipeline: encode categoricals, scale features, return X, y.
    """
    df_encoded, encoders = encode_features(df, fit=True)
    feature_cols = [c for c in FEATURE_COLS if c in df_encoded.columns]
    X = df_encoded[feature_cols]
    y = df_encoded[TARGET_COL].values

    X_scaled, scaler = scale_features(X, fit=True)
    return X_scaled, y, encoders, scaler, feature_cols


def prepare_single_prediction(
    input_data: dict,
    encoders: dict,
    scaler: StandardScaler,
    feature_cols: List[str],
) -> np.ndarray:
    """Transform a single prediction input dict into scaled feature vector."""
    row = dict(input_data)

    # Engineer derived features from pre-release inputs only
    budget = float(row.get("budget", 1))
    runtime = int(row.get("runtime", 120))
    release_month = int(row.get("release_month", 6))

    row["budget_category"] = _budget_category(budget)
    row["runtime_category"] = _runtime_category(runtime)
    row["release_season"] = _season_from_month(release_month)

    genre_pop = {
        "Action": 85, "Adventure": 78, "Animation": 72, "Comedy": 80,
        "Crime": 65, "Drama": 70, "Fantasy": 75, "Horror": 68,
        "Mystery": 62, "Romance": 74, "Sci-Fi": 77, "Thriller": 73,
    }
    row["genre_popularity"] = genre_pop.get(row.get("genre", "Drama"), 70)

    df_single = pd.DataFrame([row])
    df_encoded, _ = encode_features(df_single, encoders=encoders, fit=False)

    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    X = df_encoded[feature_cols]
    X_scaled, _ = scale_features(X, scaler=scaler, fit=False)
    return X_scaled
