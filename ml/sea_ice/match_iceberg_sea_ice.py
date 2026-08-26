from pathlib import Path

import numpy as np
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
    / "iceberg_baseline_predictions.csv"
)

SEA_ICE_FILE = (
    PROJECT_ROOT
    / "data"
    / "sea_ice"
    / "processed"
    / "sea_ice_concentration.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "sea_ice"
    / "processed"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "iceberg_sea_ice.csv"
)


# ---------------------------------------------------------
# Distance calculation
# ---------------------------------------------------------

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

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

    return 6371.0 * c


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=== Iceberg / Sea-Ice Matching ===")
    print()

    print(
        f"Loading iceberg predictions: "
        f"{ICEBERG_FILE}"
    )

    icebergs = pd.read_csv(
        ICEBERG_FILE,
        parse_dates=["Last Update"]
    )

    print(
        f"Iceberg predictions: "
        f"{len(icebergs)}"
    )

    print()

    print(
        f"Loading sea-ice data: "
        f"{SEA_ICE_FILE}"
    )

    sea_ice = pd.read_csv(
        SEA_ICE_FILE,
        parse_dates=["date"]
    )

    print(
        f"Sea-ice observations: "
        f"{len(sea_ice)}"
    )

    # -----------------------------------------------------
    # Prepare dates
    # -----------------------------------------------------

    icebergs["date"] = (
        icebergs["Last Update"]
        .dt.normalize()
    )

    sea_ice["date"] = (
        sea_ice["date"]
        .dt.normalize()
    )

    # -----------------------------------------------------
    # Match each iceberg prediction with the nearest
    # sea-ice grid cell on the same date.
    # -----------------------------------------------------

    results = []

    sea_ice_by_date = {
        date: group
        for date, group
        in sea_ice.groupby("date")
    }

    for _, iceberg in icebergs.iterrows():

        date = iceberg["date"]

        if date not in sea_ice_by_date:

            continue

        grid = sea_ice_by_date[date]

        distances = haversine_distance(
            iceberg["predicted_latitude"],
            iceberg["predicted_longitude"],
            grid["latitude"].values,
            grid["longitude"].values
        )

        nearest_index = np.argmin(
            distances
        )

        nearest = grid.iloc[
            nearest_index
        ]

        result = iceberg.copy()

        result[
            "sea_ice_latitude"
        ] = nearest["latitude"]

        result[
            "sea_ice_longitude"
        ] = nearest["longitude"]

        result[
            "ice_concentration"
        ] = nearest[
            "ice_concentration"
        ]

        result[
            "sea_ice_distance_km"
        ] = distances[
            nearest_index
        ]

        results.append(result)

    # -----------------------------------------------------
    # Create output
    # -----------------------------------------------------

    result_df = pd.DataFrame(results)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print()
    print("=== Matching Summary ===")

    print(
        f"Matched predictions: "
        f"{len(result_df)}"
    )

    if len(result_df) > 0:

        print(
            f"Average nearest-grid distance: "
            f"{result_df['sea_ice_distance_km'].mean():.2f} km"
        )

        print(
            f"Average ice concentration: "
            f"{result_df['ice_concentration'].mean():.3f}"
        )

        print()
        print("Ice concentration percentiles:")

        print(
            result_df[
                "ice_concentration"
            ].describe(
                percentiles=[
                    0.25,
                    0.50,
                    0.75,
                    0.90,
                    0.95
                ]
            )
        )

    print()
    print("Saved:")
    print(OUTPUT_FILE)

    print()
    print("=== Matching Complete ===")


if __name__ == "__main__":
    main()