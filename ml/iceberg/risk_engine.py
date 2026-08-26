from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# File locations
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "iceberg"
    / "processed"
    / "iceberg_baseline_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "iceberg"
    / "processed"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "iceberg_risk.csv"
)


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

EARTH_RADIUS_KM = 6371.0

# Prototype warning radius.
#
# This is NOT a guaranteed safety radius.
# It is an operational prototype threshold used to
# demonstrate route-risk classification.
WARNING_RADIUS_KM = 10.0


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
# Risk classification
# ---------------------------------------------------------

def classify_risk(distance_km):

    if distance_km <= 2:
        return "CRITICAL"

    if distance_km <= 5:
        return "HIGH"

    if distance_km <= WARNING_RADIUS_KM:
        return "MEDIUM"

    return "LOW"


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=== Antarctic Iceberg Risk Engine ===")
    print()

    print(f"Loading: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df)} iceberg predictions.")

    # -----------------------------------------------------
    # Calculate distance from previous/current vessel
    # position to predicted iceberg position.
    #
    # For now this is a prototype hazard layer.
    # Later the route planner will provide actual route
    # coordinates instead of a single vessel position.
    # -----------------------------------------------------

    # Use the previous observed position as the reference
    # point for this first prototype.
    df["predicted_distance_from_reference_km"] = (
        haversine_distance(
            df["previous_latitude"],
            df["previous_longitude"],
            df["predicted_latitude"],
            df["predicted_longitude"]
        )
    )

    # -----------------------------------------------------
    # Classify risk
    # -----------------------------------------------------

    df["risk_level"] = (
        df["predicted_distance_from_reference_km"]
        .apply(classify_risk)
    )

    # -----------------------------------------------------
    # Save
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
    # Summary
    # -----------------------------------------------------

    print()
    print("=== Risk Summary ===")

    print(
        df["risk_level"]
        .value_counts()
        .to_string()
    )

    print()
    print("Closest predicted movements:")

    print(
        df[
            [
                "Iceberg",
                "Last Update",
                "predicted_distance_from_reference_km",
                "risk_level"
            ]
        ]
        .sort_values(
            "predicted_distance_from_reference_km"
        )
        .head(15)
        .to_string(index=False)
    )

    print()
    print("Saved risk dataset to:")
    print(OUTPUT_FILE)

    print()
    print("=== Risk Engine Complete ===")


if __name__ == "__main__":
    main()