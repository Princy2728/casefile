"""
LIFECYCLE-X AI — ML Training Script
Trains a Random Forest classifier on the synthetic auth dataset.
Run this ONCE to generate model/rf_model.joblib and model/encoders.joblib.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    classification_report
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "auth_logs.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.joblib")
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.joblib")

os.makedirs(MODEL_DIR, exist_ok=True)

SENTINEL_DATE = "2099-12-31"   # used for active employees (no termination date)
REFERENCE_DATE = datetime(2026, 11, 20)

CATEGORICAL_COLS = [
    "employment_status",
    "session_type",
    "device_category",
    "mfa_status",
    "application_sensitivity",
]

FEATURE_COLS = [
    "employment_status_enc",
    "days_since_termination",
    "session_type_enc",
    "token_age_hours",
    "device_category_enc",
    "mfa_status_enc",
    "hr_idp_sync_delay_hours",
    "application_sensitivity_enc",
    "revocation_acknowledged",
    "auth_success",
]

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=" * 60)
print("LIFECYCLE-X AI — ML Training")
print("=" * 60)

if not os.path.exists(DATA_PATH):
    print(f"❌ Dataset not found at {DATA_PATH}")
    print("   Run generate_dataset.py first.")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded {len(df)} records from {DATA_PATH}")
print(f"   Class distribution:\n{df['risk_label'].value_counts().to_string()}\n")

# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------

# Fill missing termination_date with sentinel
df["termination_date"] = df["termination_date"].fillna(SENTINEL_DATE)

# Parse dates robustly
def parse_date_safe(d):
    try:
        return datetime.strptime(str(d)[:10], "%Y-%m-%d")
    except Exception:
        return datetime.strptime(SENTINEL_DATE, "%Y-%m-%d")

df["termination_date_dt"] = df["termination_date"].apply(parse_date_safe)

# days_since_termination: negative = still employed; large positive = long since terminated
df["days_since_termination"] = (REFERENCE_DATE - df["termination_date_dt"]).dt.days if hasattr(
    (REFERENCE_DATE - df["termination_date_dt"]), "dt") else df["termination_date_dt"].apply(
    lambda d: (REFERENCE_DATE - d).days
)

# ---------------------------------------------------------------------------
# 3. Encode categoricals
# ---------------------------------------------------------------------------
encoders = {}
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    print(f"   Encoded {col}: {list(le.classes_)}")

print()

# ---------------------------------------------------------------------------
# 4. Prepare features & target
# ---------------------------------------------------------------------------
X = df[FEATURE_COLS].copy()
y = df["risk_label"]

# Safety: fill any remaining NaN
X = X.fillna(0)

# ---------------------------------------------------------------------------
# 5. Train/test split (stratified)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
print(f"Train class dist: {dict(pd.Series(y_train).value_counts())}")
print(f"Test  class dist: {dict(pd.Series(y_test).value_counts())}\n")

# ---------------------------------------------------------------------------
# 6. Train Random Forest
# ---------------------------------------------------------------------------
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train, y_train)
print("✅ Model trained.\n")

# ---------------------------------------------------------------------------
# 7. Evaluate
# ---------------------------------------------------------------------------
y_pred = clf.predict(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
cm   = confusion_matrix(y_test, y_pred)

print("─" * 40)
print("MODEL PERFORMANCE")
print("─" * 40)
print(f"  Accuracy  : {acc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print()
print("Confusion Matrix:")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
print()
print("Full Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal (0)", "High Risk (1)"]))

# ---------------------------------------------------------------------------
# 8. Feature importances
# ---------------------------------------------------------------------------
importances = dict(zip(FEATURE_COLS, clf.feature_importances_))
sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
print("Feature Importances (descending):")
for feat, imp in sorted_imp:
    bar = "█" * int(imp * 50)
    print(f"  {feat:<35} {imp:.4f} {bar}")

# ---------------------------------------------------------------------------
# 9. Save model and encoders
# ---------------------------------------------------------------------------
metrics = {
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1_score": f1,
    "confusion_matrix": cm,
    "feature_importances": importances,
    "feature_cols": FEATURE_COLS,
}

joblib.dump(clf, MODEL_PATH)
joblib.dump({"encoders": encoders, "metrics": metrics, "feature_cols": FEATURE_COLS,
             "sentinel_date": SENTINEL_DATE, "reference_date": REFERENCE_DATE},
            ENCODERS_PATH)

print(f"\n✅ Model saved   : {MODEL_PATH}")
print(f"✅ Encoders saved : {ENCODERS_PATH}")
print("\nTraining complete.")
