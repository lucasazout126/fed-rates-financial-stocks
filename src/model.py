"""
Purpose:
Train and evaluate machine learning models that predict
financial sector performance after Fed rate changes.

Responsibilities:
- Load engineered features
- Split data into train/test sets
- Train classification models (Logistic Regression, Random Forest, etc.)
- Evaluate using accuracy, precision, recall, and F1
- Save trained model to disk for later use by the web app

Output:
model.pkl
"""

import os
import pickle
import psycopg2
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline

load_dotenv()

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     os.getenv("DB_PORT", "5432"),
    "dbname":   os.getenv("DB_NAME", "fed_rates_db"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "best_model.pkl")

# ── Feature columns used for training ────────────────────────────────────────

FEATURE_COLS = [
    "rate_before",
    "change_bp",
    "abs_change_bp",
    "pre_return_10d",
    "pre_return_30d",
    "pre_volatility_30d",
    "pre_rel_return_10d",
    "pre_rel_return_30d",
]

CATEGORICAL_COLS = {
    "direction":          {"hike": 1, "cut": -1, "hold": 0},
    "rate_level_regime":  {"low": 0, "mid": 1, "high": 2},
}

TARGET_COL = "outperformed"

# ── Load data ─────────────────────────────────────────────────────────────────

def load_features() -> pd.DataFrame:
    with psycopg2.connect(**DB_CONFIG) as conn:
        df = pd.read_sql(
            f"""
            SELECT {', '.join(FEATURE_COLS + list(CATEGORICAL_COLS.keys()) + [TARGET_COL])}
            FROM features
            ORDER BY fomc_date, ticker
            """,
            conn
        )
    return df


def prepare_data(df: pd.DataFrame):
    """Encode categoricals, drop nulls, split X/y."""
    for col, mapping in CATEGORICAL_COLS.items():
        df[col] = df[col].map(mapping)

    df = df.dropna()

    all_feature_cols = FEATURE_COLS + list(CATEGORICAL_COLS.keys())
    X = df[all_feature_cols].astype(float)
    y = df[TARGET_COL].astype(int)

    return X, y


# ── Model definitions ─────────────────────────────────────────────────────────

def get_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42))
        ]),
    }


# ── Cross-validation evaluation ───────────────────────────────────────────────

def evaluate_models(X, y):
    """Run stratified k-fold CV on all models and return results."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1"]

    results = {}
    print("\n── Cross-Validation Results (5-fold) ──────────────────────\n")

    for name, model in get_models().items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
        results[name] = {
            "accuracy":  scores["test_accuracy"].mean(),
            "precision": scores["test_precision"].mean(),
            "recall":    scores["test_recall"].mean(),
            "f1":        scores["test_f1"].mean(),
        }
        print(f"{name}")
        for metric, val in results[name].items():
            print(f"  {metric:<12} {val:.4f}")
        print()

    return results


# ── Train best model on full data ─────────────────────────────────────────────

def train_best_model(X, y, results: dict):
    """Pick the model with the highest F1 and retrain on all data."""
    best_name = max(results, key=lambda k: results[k]["f1"])
    print(f"Best model by F1: {best_name} ({results[best_name]['f1']:.4f})\n")

    best_model = get_models()[best_name]
    best_model.fit(X, y)

    # Final evaluation on training set (for reference)
    y_pred = best_model.predict(X)
    print("── Full-Data Evaluation (training set) ────────────────────\n")
    print(classification_report(y, y_pred, target_names=["Underperform", "Outperform"]))

    print("Confusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}\n")

    return best_name, best_model


# ── Feature importance ────────────────────────────────────────────────────────

def print_feature_importance(model_name, model, feature_names):
    clf = model.named_steps["clf"]
    print("── Feature Importance ──────────────────────────────────────\n")

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        print("  (not available for this model type)")
        return

    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    for feat, imp in pairs:
        bar = "█" * int(imp * 40)
        print(f"  {feat:<25} {imp:.4f}  {bar}")
    print()


# ── Save model ────────────────────────────────────────────────────────────────

def save_model(model_name, model, feature_names):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    payload = {
        "model_name":    model_name,
        "model":         model,
        "feature_names": feature_names,
        "feature_cols":  FEATURE_COLS + list(CATEGORICAL_COLS.keys()),
        "cat_mappings":  CATEGORICAL_COLS,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"✅ Model saved to {MODEL_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== ML Model Training & Evaluation ===\n")

    print("Loading features from database …")
    df = load_features()
    print(f"  → {len(df)} rows loaded.\n")

    print("Preparing data …")
    X, y = prepare_data(df)
    feature_names = list(X.columns)
    print(f"  → {X.shape[0]} samples, {X.shape[1]} features.")
    print(f"  → Class balance: {y.value_counts().to_dict()}\n")

    results = evaluate_models(X, y)

    best_name, best_model = train_best_model(X, y, results)

    print_feature_importance(best_name, best_model, feature_names)

    save_model(best_name, best_model, feature_names)

    print("\n✅ Model training complete.")