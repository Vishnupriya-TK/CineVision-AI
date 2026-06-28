"""
CineVision AI - Model training script.
Compares multiple ML algorithms and saves the best model.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import (
    ENCODER_PATH,
    FEATURE_NAMES_PATH,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    SCALER_PATH,
    ensure_directories,
)
from utils.preprocessing import TARGET_COL, load_or_create_dataset, prepare_training_data

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def get_models() -> dict:
    """Return dictionary of candidate classifiers."""
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=6, random_state=42
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            eval_metric="mlogloss",
            verbosity=0,
        )
    return models


def compute_feature_importance(model, feature_names: list) -> dict:
    """Extract feature importance from trained model."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)
    return dict(zip(feature_names, importances.tolist()))


def train_and_evaluate():
    """Main training pipeline: load data, train models, save best."""
    ensure_directories()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  CineVision AI - Model Training Pipeline")
    print("=" * 60)

    # Load and prepare data
    print("\n[1/5] Loading and preprocessing dataset...")
    df = load_or_create_dataset()
    print(f"      Dataset shape: {df.shape}")
    print(f"      Categories: {df[TARGET_COL].value_counts().to_dict()}")

    X, y, encoders, scaler, feature_cols = prepare_training_data(df)

    # Encode target labels for XGBoost compatibility
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Compare models
    print("\n[2/5] Comparing machine learning algorithms...")
    models = get_models()
    results = {}

    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_weighted")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[name] = {
            "model": model,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "y_pred": y_pred,
        }
        print(
            f"      {name:20s} | CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f}) "
            f"| Test Acc: {results[name]['accuracy']:.4f}"
        )

    # Select best model by F1 score
    best_name = max(results, key=lambda k: results[k]["f1"])
    best = results[best_name]
    best_model = best["model"]
    y_pred = best["y_pred"]

    print(f"\n[3/5] Best model: {best_name} (F1: {best['f1']:.4f})")

    # Detailed metrics
    class_labels = target_encoder.classes_
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_labels, output_dict=True)
    feature_importance = compute_feature_importance(best_model, feature_cols)

    # ROC curves (one-vs-rest)
    y_test_bin = label_binarize(y_test, classes=range(len(class_labels)))
    y_proba = best_model.predict_proba(X_test)
    fpr_dict, tpr_dict, auc_dict = {}, {}, {}
    for i, label in enumerate(class_labels):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        fpr_dict[label] = fpr.tolist()
        tpr_dict[label] = tpr.tolist()
        auc_dict[label] = float(roc_auc_score(y_test_bin[:, i], y_proba[:, i]))

    print("\n[4/5] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_labels))

    print("      Confusion Matrix:")
    print(cm)

    print("\n      Top 5 Feature Importances:")
    sorted_fi = sorted(feature_importance.items(), key=lambda x: -x[1])[:5]
    for feat, imp in sorted_fi:
        print(f"        {feat}: {imp:.4f}")

    # Save artifacts
    print("\n[5/5] Saving model artifacts...")
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(encoders, ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_cols, FEATURE_NAMES_PATH)

    metrics = {
        "best_model_name": best_name,
        "accuracy": best["accuracy"],
        "precision": best["precision"],
        "recall": best["recall"],
        "f1_score": best["f1"],
        "cv_mean": best["cv_mean"],
        "cv_std": best["cv_std"],
        "confusion_matrix": cm.tolist(),
        "class_labels": class_labels.tolist(),
        "classification_report": report,
        "feature_importance": feature_importance,
        "feature_names": feature_cols,
        "roc_fpr": fpr_dict,
        "roc_tpr": tpr_dict,
        "roc_auc": auc_dict,
        "all_model_scores": {
            name: {
                "accuracy": r["accuracy"],
                "f1": r["f1"],
                "precision": r["precision"],
                "recall": r["recall"],
            }
            for name, r in results.items()
        },
        "target_encoder": target_encoder,
        "has_xgboost": HAS_XGBOOST,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "dataset_size": len(df),
    }
    joblib.dump(metrics, METRICS_PATH)

    print(f"\n      Saved: {MODEL_PATH}")
    print(f"      Saved: {ENCODER_PATH}")
    print(f"      Saved: {SCALER_PATH}")
    print(f"      Saved: {METRICS_PATH}")
    print("\n" + "=" * 60)
    print(f"  Training complete! Best model: {best_name}")
    print(f"  Accuracy: {best['accuracy']:.2%} | F1: {best['f1']:.4f}")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    train_and_evaluate()
