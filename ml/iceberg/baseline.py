from pathlib import Path
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# File locations
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

OUTPUT_FILE = OUTPUT_DIR / "iceberg_baseline_predictions.csv"


EARTH_RADIUS_KM = 6371.0


# ---------------------------------------------------------
# Predict destination from starting point, speed and bearing
# ---------------------------------------------------------

def destination_point(lat, lon, distance_km, bearing_deg):
    """
    Calculate the destination point after travelling
    distance_km along bearing_deg from lat/lon.

    Uses a spherical Earth approximation.
    """

    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    bearing = np.radians(bearing_deg)

    angular_distance = distance_km / EARTH_RADIUS_KM

    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(angular_distance)
        + np.cos(lat1)
        * np.sin(angular_distance)
        * np.cos(bearing)
    )

    lon2 = lon1 + np.arctan2(
        np.sin(bearing) * np.sin(angular_distance) * np.cos(lat1),
        np.cos(angular_distance)
        - np.sin(lat1) * np.sin(lat2)
    )

    lat2 = np.degrees(lat2)
    lon2 = np.degrees(lon2)

    # Normalize longitude to [-180, 180].
    lon2 = (lon2 + 180) % 360 - 180

    return lat2, lon2


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

    print("=== Iceberg Constant-Velocity Baseline ===")
    print()

    print(f"Loading: {TEST_FILE}")

    df = pd.read_csv(
        TEST_FILE,
        parse_dates=["previous_date", "Last Update"]
    )

    print(f"Loaded {len(df)} test observations.")

    # -----------------------------------------------------
    # Predict how far the iceberg should travel during
    # the observed time interval.
    # -----------------------------------------------------

    df["predicted_distance_km"] = (
    df["speed_kmh"]
    * df["time_difference_days"]
    * 24
)

    # -----------------------------------------------------
    # Predict destination.
    # -----------------------------------------------------

    predictions = df.apply(
        lambda row: destination_point(
            row["previous_latitude"],
            row["previous_longitude"],
            row["predicted_distance_km"],
            row["bearing_deg"]
        ),
        axis=1
    )

    df["predicted_latitude"] = [
        point[0] for point in predictions
    ]

    df["predicted_longitude"] = [
        point[1] for point in predictions
    ]

    # -----------------------------------------------------
    # Calculate prediction error.
    # -----------------------------------------------------

    df["prediction_error_km"] = haversine_distance(
        df["predicted_latitude"],
        df["predicted_longitude"],
        df["target_latitude"],
        df["target_longitude"]
    )

    # -----------------------------------------------------
    # Save predictions.
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    mae = df["prediction_error_km"].mean()
    median_error = df["prediction_error_km"].median()
    rmse = np.sqrt(
        np.mean(
            df["prediction_error_km"] ** 2
        )
    )

    print()
    print("=== Baseline Results ===")
    print(f"Test observations: {len(df)}")
    print(f"MAE: {mae:.2f} km")
    print(f"Median error: {median_error:.2f} km")
    print(f"RMSE: {rmse:.2f} km")

    print()
    print("Largest prediction errors:")

    print(
        df[
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

    print()
    print("Saved predictions to:")
    print(OUTPUT_FILE)

    print()
    print("=== Baseline Complete ===")


if __name__ == "__main__":
    main()