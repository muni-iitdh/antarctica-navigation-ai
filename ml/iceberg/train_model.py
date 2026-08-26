from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor


# ---------------------------------------------------------
# File locations
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "iceberg"
    / "processed"
    / "iceberg_train.csv"
)

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "iceberg"
    / "processed"
    / "iceberg_test.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "iceberg"
    / "processed"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "iceberg_ml_predictions.csv"
)


# ---------------------------------------------------------
# Features and targets
# ---------------------------------------------------------

FEATURES = [
    "previous_latitude",
    "previous_longitude",
    "distance_km",
    "speed_kmh",
    "bearing_deg",
    "time_difference_days",
]

TARGETS = [
    "target_latitude",
    "target_longitude",
]

EARTH_RADIUS_KM = 6371.0


# ---------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------

def haversine_distance(lat1, lon1, lat2, lon2):

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(delta_lon / 2) ** 2
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a)
    )

    return EARTH_RADIUS_KM * c


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=== Iceberg Random Forest Model ===")
    print()

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    print(f"Loading training data: {TRAIN_FILE}")

    train = pd.read_csv(TRAIN_FILE)

    print(f"Training rows: {len(train)}")

    print()
    print(f"Loading test data: {TEST_FILE}")

    test = pd.read_csv(TEST_FILE)

    print(f"Test rows: {len(test)}")

    # -----------------------------------------------------
    # Prepare features and targets
    # -----------------------------------------------------

    X_train = train[FEATURES]
    y_train = train[TARGETS]

    X_test = test[FEATURES]
    y_test = test[TARGETS]

    # -----------------------------------------------------
    # Train Random Forest
    # -----------------------------------------------------

    print()
    print("Training Random Forest...")

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    print("Training complete.")

    # -----------------------------------------------------
    # Predict
    # -----------------------------------------------------

    predictions = model.predict(X_test)

    test["predicted_latitude"] = predictions[:, 0]
    test["predicted_longitude"] = predictions[:, 1]

    # -----------------------------------------------------
    # Calculate position error
    # -----------------------------------------------------

    test["prediction_error_km"] = haversine_distance(
        test["predicted_latitude"],
        test["predicted_longitude"],
        test["target_latitude"],
        test["target_longitude"]
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    errors = test["prediction_error_km"]

    mae = errors.mean()

    median_error = errors.median()

    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    within_1km = (
        errors <= 1
    ).mean() * 100

    within_5km = (
        errors <= 5
    ).mean() * 100

    within_10km = (
        errors <= 10
    ).mean() * 100

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    print()
    print("=== Random Forest Results ===")

    print(
        f"Test observations: "
        f"{len(test)}"
    )

    print(
        f"MAE: "
        f"{mae:.2f} km"
    )

    print(
        f"Median error: "
        f"{median_error:.2f} km"
    )

    print(
        f"RMSE: "
        f"{rmse:.2f} km"
    )

    print(
        f"Within 1 km: "
        f"{within_1km:.2f}%"
    )

    print(
        f"Within 5 km: "
        f"{within_5km:.2f}%"
    )

    print(
        f"Within 10 km: "
        f"{within_10km:.2f}%"
    )

    # -----------------------------------------------------
    # Feature importance
    # -----------------------------------------------------

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURES
    ).sort_values(
        ascending=False
    )

    print()
    print("=== Feature Importance ===")
    print(importance.to_string())

    # -----------------------------------------------------
    # Largest errors
    # -----------------------------------------------------

    print()
    print("=== Largest Prediction Errors ===")

    print(
        test[
            [
                "Iceberg",
                "Last Update",
                "prediction_error_km"
            ]
        ]
        .sort_values(
            "prediction_error_km",
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )

    # -----------------------------------------------------
    # Save predictions
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    test.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("Saved predictions to:")
    print(OUTPUT_FILE)

    print()
    print("=== Random Forest Complete ===")


if __name__ == "__main__":
    main()