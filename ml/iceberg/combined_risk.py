from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ICEBERG_FILE = (
    PROJECT_ROOT
    / "data"
    / "iceberg"
    / "processed"
    / "iceberg_risk.csv"
)

SEA_ICE_FILE = (
    PROJECT_ROOT
    / "data"
    / "sea_ice"
    / "processed"
    / "sea_ice_risk.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "iceberg"
    / "processed"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "combined_environmental_risk.csv"
)


# ---------------------------------------------------------
# Prototype iceberg risk scores
#
# These convert the EXISTING risk_level into a numeric
# value for combination with sea-ice exposure.
#
# They are prototype values, not maritime safety standards.
# ---------------------------------------------------------

ICEBERG_RISK_SCORES = {
    "LOW": 25,
    "MEDIUM": 50,
    "HIGH": 75,
    "CRITICAL": 100,
}


ICEBERG_WEIGHT = 0.70
SEA_ICE_WEIGHT = 0.30


def classify_combined_risk(score):

    if score >= 80:
        return "CRITICAL"

    if score >= 60:
        return "HIGH"

    if score >= 30:
        return "MEDIUM"

    return "LOW"


def main():

    print("=== Combined Environmental Risk Engine ===")
    print()

    # -----------------------------------------------------
    # Load iceberg risk
    # -----------------------------------------------------

    print(f"Loading iceberg risk: {ICEBERG_FILE}")

    iceberg = pd.read_csv(
        ICEBERG_FILE,
        parse_dates=["Last Update"]
    )

    print(
        f"Iceberg risk observations: {len(iceberg)}"
    )

    # -----------------------------------------------------
    # Load sea-ice risk
    # -----------------------------------------------------

    print()
    print(f"Loading sea-ice risk: {SEA_ICE_FILE}")

    sea_ice = pd.read_csv(
        SEA_ICE_FILE,
        parse_dates=["Last Update"]
    )

    print(
        f"Sea-ice observations: {len(sea_ice)}"
    )

    # -----------------------------------------------------
    # Convert existing iceberg risk level to numeric score
    # -----------------------------------------------------

    iceberg["iceberg_risk_score"] = (
        iceberg["risk_level"]
        .map(ICEBERG_RISK_SCORES)
    )

    # Check for unexpected risk levels.
    unknown_levels = iceberg[
        iceberg["iceberg_risk_score"].isna()
    ]["risk_level"].dropna().unique()

    if len(unknown_levels) > 0:

        raise ValueError(
            f"Unknown iceberg risk levels: "
            f"{unknown_levels.tolist()}"
        )

    # -----------------------------------------------------
    # Select sea-ice columns
    # -----------------------------------------------------

    sea_ice_subset = sea_ice[
        [
            "Iceberg",
            "Last Update",
            "ice_concentration",
            "sea_ice_risk_score",
            "sea_ice_risk_level",
            "sea_ice_distance_km",
        ]
    ].copy()

    # -----------------------------------------------------
    # Join.
    #
    # LEFT JOIN is intentional:
    # missing sea-ice data != zero risk.
    # -----------------------------------------------------

    df = iceberg.merge(
        sea_ice_subset,
        on=[
            "Iceberg",
            "Last Update",
        ],
        how="left"
    )

    # -----------------------------------------------------
    # Sea-ice availability
    # -----------------------------------------------------

    df["sea_ice_available"] = (
        df["sea_ice_risk_score"].notna()
    )

    # -----------------------------------------------------
    # Combined risk
    #
    # Only calculate when BOTH environmental signals
    # are available.
    # -----------------------------------------------------

    df["combined_risk_score"] = pd.NA
    df["combined_risk_level"] = pd.NA

    available = df["sea_ice_available"]

    df.loc[
        available,
        "combined_risk_score"
    ] = (
        ICEBERG_WEIGHT
        * df.loc[
            available,
            "iceberg_risk_score"
        ]
        + SEA_ICE_WEIGHT
        * df.loc[
            available,
            "sea_ice_risk_score"
        ]
    )

    df.loc[
        available,
        "combined_risk_level"
    ] = (
        df.loc[
            available,
            "combined_risk_score"
        ]
        .astype(float)
        .apply(classify_combined_risk)
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
    print("=== Combined Risk Summary ===")

    print(
        f"Total iceberg observations: "
        f"{len(df)}"
    )

    print(
        f"Sea-ice available: "
        f"{int(df['sea_ice_available'].sum())}"
    )

    print(
        f"Sea-ice unavailable: "
        f"{int((~df['sea_ice_available']).sum())}"
    )

    available_df = df[
        df["sea_ice_available"]
    ].copy()

    if len(available_df) > 0:

        print()
        print("Combined risk levels:")

        print(
            available_df[
                "combined_risk_level"
            ].value_counts()
        )

        print()
        print("Highest combined-risk observations:")

        print(
            available_df[
                [
                    "Iceberg",
                    "Last Update",
                    "predicted_latitude",
                    "predicted_longitude",
                    "risk_level",
                    "iceberg_risk_score",
                    "ice_concentration",
                    "sea_ice_risk_score",
                    "combined_risk_score",
                    "combined_risk_level",
                ]
            ]
            .sort_values(
                "combined_risk_score",
                ascending=False
            )
            .head(10)
            .to_string(index=False)
        )

    print()
    print("Saved:")
    print(OUTPUT_FILE)

    print()
    print("=== Combined Risk Complete ===")


if __name__ == "__main__":
    main()