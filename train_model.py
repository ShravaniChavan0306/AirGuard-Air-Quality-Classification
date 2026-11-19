
import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------
# 1. PROJECT SETTINGS
# ----------------------------------

DATA_PATH = "data/city_day.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3"
]

CLASS_NAMES = ["Good", "Moderate", "Poor"]

# ----------------------------------
# 2. LOAD DATASET
# ----------------------------------

print("\nLoading air quality dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}. "
        "Please check that city_day.csv is inside the data folder."
    )

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Original dataset shape:", df.shape)

# Check required columns
required_columns = FEATURES + ["AQI_Bucket"]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )

# ----------------------------------
# 3. PREPARE TARGET CLASSES
# ----------------------------------

def convert_aqi_category(category):

    if pd.isna(category):
        return np.nan

    category = str(category).strip().lower()

    if category in ["good", "satisfactory"]:
        return "Good"

    elif category == "moderate":
        return "Moderate"

    elif category in ["poor", "very poor", "severe"]:
        return "Poor"

    else:
        return np.nan


df["Target"] = df["AQI_Bucket"].apply(convert_aqi_category)

# Remove rows without a valid target
df = df.dropna(subset=["Target"]).copy()

# ----------------------------------
# 4. PREPARE INPUT FEATURES
# ----------------------------------

X = df[FEATURES].copy()

# Convert feature columns into numeric values
for column in FEATURES:
    X[column] = pd.to_numeric(
        X[column], errors="coerce"
    )

y = df["Target"]

print("\nClass distribution:")
print(y.value_counts())

# Check that all three classes are available
if set(y.unique()) != set(CLASS_NAMES):
    raise ValueError(
        "The dataset must contain all three classes: "
        "Good, Moderate and Poor."
    )

# ----------------------------------
# 5. SPLIT DATASET
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ----------------------------------
# 6. DEFINE THREE ML MODELS
# ----------------------------------

models = {

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(
            strategy="median",
            keep_empty_features=True
        )),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
    ]),

    "SVM": Pipeline([
        ("imputer", SimpleImputer(
            strategy="median",
            keep_empty_features=True
        )),
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Neural Network": Pipeline([
        ("imputer", SimpleImputer(
            strategy="median",
            keep_empty_features=True
        )),
        ("scaler", StandardScaler()),
        ("classifier", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            random_state=42
        ))
    ])
}

# ----------------------------------
# 7. TRAIN AND EVALUATE MODELS
# ----------------------------------

all_metrics = {}

print("\n========== MODEL TRAINING ==========")

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    )
    recall = recall_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    )
    f1 = f1_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    )

    all_metrics[name] = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=CLASS_NAMES,
            output_dict=True,
            zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=CLASS_NAMES
        ).tolist()
    }

    print(f"{name} Results:")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=CLASS_NAMES,
        zero_division=0
    ))

    # Save trained model
    filename = name.lower().replace(" ", "_") + ".joblib"

    model_path = os.path.join(MODEL_DIR, filename)

    joblib.dump(model, model_path)

    print("Saved model:", model_path)

# ----------------------------------
# 8. SAVE MODEL METRICS AND DEFAULTS
# ----------------------------------

feature_defaults = X.median().fillna(0).to_dict()

metadata = {
    "features": FEATURES,
    "classes": CLASS_NAMES,
    "defaults": {
        key: float(value)
        for key, value in feature_defaults.items()
    },
    "metrics": all_metrics
}

with open(
    os.path.join(MODEL_DIR, "metrics.json"),
    "w"
) as file:
    json.dump(metadata, file, indent=4)

# ----------------------------------
# 9. FINAL RESULTS
# ----------------------------------

print("\n========== FINAL MODEL COMPARISON ==========")

for name, result in all_metrics.items():

    print(
        f"{name}: "
        f"Accuracy = {result['accuracy'] * 100:.2f}% | "
        f"F1 Score = {result['f1_score'] * 100:.2f}%"
    )

print("\nAll three models trained successfully!")
print("Trained models and metrics saved in the models folder.")
