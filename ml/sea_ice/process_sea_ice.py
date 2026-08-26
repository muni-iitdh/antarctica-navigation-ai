from pathlib import Path

import pandas as pd
import xarray as xr


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "sea_ice"
    / "raw"
    / "sea_ice_20260814_20260820.nc"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "sea_ice"
    / "processed"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "sea_ice_concentration.csv"
)


# ---------------------------------------------------------
# Processing
# ---------------------------------------------------------

def main():

    print("=== Antarctic Sea-Ice Processing ===")
    print()

    print(f"Loading: {INPUT_FILE}")

    ds = xr.open_dataset(INPUT_FILE)

    print()
    print("Dataset:")
    print(ds)

    # Convert the ice concentration grid into a DataFrame.
    df = ds["icec"].to_dataframe(
        name="ice_concentration"
    ).reset_index()

    # Rename coordinates to our project convention.
    df = df.rename(
        columns={
            "time": "date",
            "lat": "latitude",
            "lon": "longitude",
        }
    )

    # -----------------------------------------------------
    # Normalize longitude.
    #
    # NOAA file uses approximately:
    # 0 → 360 degrees
    #
    # Our iceberg data uses:
    # -180 → +180 degrees
    #
    # Convert NOAA longitude into the same convention.
    # -----------------------------------------------------

    df["longitude"] = (
        (df["longitude"] + 180) % 360
    ) - 180

    # -----------------------------------------------------
    # Keep Antarctic region only.
    # -----------------------------------------------------

    df = df[
        df["latitude"] <= -50
    ].copy()

    # Remove missing values.
    df = df.dropna(
        subset=[
            "date",
            "latitude",
            "longitude",
            "ice_concentration",
        ]
    )

    # Sort for easier inspection and later matching.
    df = df.sort_values(
        [
            "date",
            "latitude",
            "longitude",
        ]
    ).reset_index(drop=True)

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
    print("=== Sea-Ice Summary ===")

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Dates: "
        f"{df['date'].min()} → "
        f"{df['date'].max()}"
    )

    print(
        f"Latitude: "
        f"{df['latitude'].min():.3f} → "
        f"{df['latitude'].max():.3f}"
    )

    print(
        f"Longitude: "
        f"{df['longitude'].min():.3f} → "
        f"{df['longitude'].max():.3f}"
    )

    print(
        f"Sea-ice concentration: "
        f"{df['ice_concentration'].min():.3f} → "
        f"{df['ice_concentration'].max():.3f}"
    )

    print()
    print("Dates available:")

    print(
        df["date"]
        .drop_duplicates()
        .to_string(index=False)
    )

    print()
    print("Saved processed dataset to:")
    print(OUTPUT_FILE)

    print()
    print("=== Sea-Ice Processing Complete ===")


if __name__ == "__main__":
    main()