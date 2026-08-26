from pathlib import Path
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
    / "iceberg_motion.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "iceberg"
    / "processed"
)

OUTPUT_FILE = OUTPUT_DIR / "iceberg_training.csv"


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=== Iceberg ML Training Dataset Creation ===")
    print()

    print(f"Loading: {INPUT_FILE}")

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Last Update", "previous_date"]
    )

    print(f"Loaded {len(df)} observations.")

    # -----------------------------------------------------
    # Keep only observations with a valid weekly interval.
    # -----------------------------------------------------

    training = df[
        df["valid_weekly_interval"] == True
    ].copy()

    print(
        f"Weekly-valid movement observations: "
        f"{len(training)}"
    )

    # -----------------------------------------------------
    # Remove rows where previous position is unavailable.
    # -----------------------------------------------------

    training = training.dropna(
        subset=[
            "previous_latitude",
            "previous_longitude",
            "Latitude",
            "Longitude"
        ]
    ).copy()

    # -----------------------------------------------------
    # Create explicit target variables.
    #
    # Features describe the iceberg BEFORE the movement.
    # Targets describe where it was observed AFTER the movement.
    # -----------------------------------------------------

    training["target_latitude"] = training["Latitude"]
    training["target_longitude"] = training["Longitude"]

    # -----------------------------------------------------
    # Select model features.
    # -----------------------------------------------------

    feature_columns = [
        "previous_latitude",
        "previous_longitude",
        "distance_km",
        "speed_kmh",
        "bearing_deg",
        "time_difference_days"
    ]

    target_columns = [
        "target_latitude",
        "target_longitude"
    ]

    training = training[
        ["Iceberg", "previous_date", "Last Update"]
        + feature_columns
        + target_columns
    ].copy()

    # -----------------------------------------------------
    # Remove rows with missing model values.
    # -----------------------------------------------------

    training = training.dropna(
        subset=feature_columns + target_columns
    ).copy()

    # -----------------------------------------------------
    # Sort chronologically.
    # -----------------------------------------------------

    training = training.sort_values(
        ["Iceberg", "Last Update"]
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # Save.
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    training.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(f"Final training rows: {len(training)}")
    print(f"Unique icebergs: {training['Iceberg'].nunique()}")

    print()
    print("Feature columns:")
    for column in feature_columns:
        print(f"  - {column}")

    print()
    print("Target columns:")
    for column in target_columns:
        print(f"  - {column}")

    print()
    print("Saved training dataset to:")
    print(OUTPUT_FILE)

    print()
    print("=== Training Dataset Creation Complete ===")


if __name__ == "__main__":
    main()