from pathlib import Path
import pandas as pd


# Location of the raw weekly iceberg CSV files
RAW_DIR = Path(__file__).parent / "raw"

# Location for the processed dataset
OUTPUT_DIR = Path(__file__).parent / "processed"
OUTPUT_FILE = OUTPUT_DIR / "iceberg_tracks.csv"


def load_iceberg_files():
    """Read all weekly iceberg CSV files from the raw directory."""

    files = sorted((RAW_DIR / "historical").glob("AntarcticIcebergs_*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No iceberg CSV files found in {RAW_DIR}"
        )

    print(f"Found {len(files)} iceberg files.")

    dataframes = []

    for file in files:
        print(f"Reading: {file.name}")

        df = pd.read_csv(file)

        # Store the source filename so we can trace each observation.
        df["source_file"] = file.name

        dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)


def clean_data(df):
    """Clean and standardize the iceberg observations."""

    # Remove accidental whitespace from column names.
    df.columns = df.columns.str.strip()

    # Standardize the iceberg ID column.
    df["Iceberg"] = df["Iceberg"].astype(str).str.strip()

    # Convert latitude and longitude to numeric values.
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

    # Convert the update date.
    df["Last Update"] = pd.to_datetime(
        df["Last Update"],
        errors="coerce"
    )

    # Remove observations where essential information is missing.
    df = df.dropna(
        subset=[
            "Iceberg",
            "Latitude",
            "Longitude",
            "Last Update",
        ]
    )

    # Keep only observations within valid geographic ranges.
    df = df[
        df["Latitude"].between(-90, 90)
        & df["Longitude"].between(-180, 180)
    ]

    # Sort observations chronologically for each iceberg.
    df = df.sort_values(
        ["Iceberg", "Last Update"]
    ).reset_index(drop=True)

    return df


def save_data(df):
    """Save the cleaned trajectory dataset."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_columns = [
        "Iceberg",
        "Last Update",
        "Latitude",
        "Longitude",
        "Length (NM)",
        "Width (NM)",
        "Area (sqNM)",
        "source_file",
    ]

    # Keep only columns that actually exist.
    output_columns = [
        column
        for column in output_columns
        if column in df.columns
    ]

    df[output_columns].to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(f"Saved processed dataset to:")
    print(OUTPUT_FILE)


def main():
    print("=== Antarctic Iceberg Processing ===")
    print()

    df = load_iceberg_files()

    print()
    print(f"Total raw observations: {len(df)}")

    df = clean_data(df)

    print(f"Clean observations: {len(df)}")
    print(f"Unique iceberg IDs: {df['Iceberg'].nunique()}")

    print()
    print("Observations per iceberg:")
    print(
        df.groupby("Iceberg")
        .size()
        .describe()
    )

    save_data(df)

    print()
    print("Processing complete.")


if __name__ == "__main__":
    main()