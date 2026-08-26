from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "sea_ice"
    / "processed"
    / "iceberg_sea_ice.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "sea_ice"
    / "processed"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "sea_ice_risk.csv"
)


# ---------------------------------------------------------
# Prototype thresholds
#
# These are NOT scientifically validated safety limits.
# They are only for the SIH prototype.
# ---------------------------------------------------------

def classify_sea_ice_risk(concentration):

    if concentration >= 0.80:
        return "VERY_HIGH"

    if concentration >= 0.50:
        return "HIGH"

    if concentration >= 0.20:
        return "MEDIUM"

    return "LOW"


def calculate_risk_score(concentration):

    # Continuous prototype score: 0 → 100
    #
    # 0.0 concentration → 0
    # 1.0 concentration → 100

    return concentration * 100.0


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=== Antarctic Sea-Ice Risk Engine ===")
    print()

    print(f"Loading: {INPUT_FILE}")

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Last Update"]
    )

    print(
        f"Loaded {len(df)} matched observations."
    )

    # -----------------------------------------------------
    # Calculate risk
    # -----------------------------------------------------

    df["sea_ice_risk_score"] = (
        df["ice_concentration"]
        .clip(0, 1)
        .apply(calculate_risk_score)
    )

    df["sea_ice_risk_level"] = (
        df["ice_concentration"]
        .apply(classify_sea_ice_risk)
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
    print("=== Sea-Ice Risk Summary ===")

    print(
        df["sea_ice_risk_level"]
        .value_counts()
    )

    print()
    print("Risk score statistics:")

    print(
        df["sea_ice_risk_score"]
        .describe()
    )

    print()
    print("Highest sea-ice risk observations:")

    print(
        df[
            [
                "Iceberg",
                "Last Update",
                "predicted_latitude",
                "predicted_longitude",
                "ice_concentration",
                "sea_ice_risk_score",
                "sea_ice_risk_level",
            ]
        ]
        .sort_values(
            "sea_ice_risk_score",
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )

    print()
    print("Saved:")
    print(OUTPUT_FILE)

    print()
    print("=== Sea-Ice Risk Complete ===")


if __name__ == "__main__":
    main()