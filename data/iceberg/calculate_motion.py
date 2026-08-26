from pathlib import Path
import pandas as pd
import numpy as np


# File locations
ICEBERG_DIR = Path(__file__).parent
INPUT_FILE = ICEBERG_DIR / "processed" / "iceberg_tracks.csv"
OUTPUT_FILE = ICEBERG_DIR / "processed" / "iceberg_motion.csv"


EARTH_RADIUS_KM = 6371.0


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two
    latitude/longitude points in kilometers.
    """

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


def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the initial bearing from point 1 to point 2.

    Bearing:
        0°   = North
        90°  = East
        180° = South
        270° = West
    """

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    delta_lon = np.radians(lon2 - lon1)

    x = np.sin(delta_lon) * np.cos(lat2)

    y = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1)
        * np.cos(lat2)
        * np.cos(delta_lon)
    )

    bearing = np.degrees(
        np.arctan2(x, y)
    )

    return (bearing + 360) % 360


def calculate_motion(df):
    """
    Calculate movement features for every iceberg.
    """

    # Make sure observations are ordered correctly.
    df = df.sort_values(
        ["Iceberg", "Last Update"]
    ).copy()

    # Previous observation for the same iceberg.
    df["previous_latitude"] = (
        df.groupby("Iceberg")["Latitude"]
        .shift(1)
    )

    df["previous_longitude"] = (
        df.groupby("Iceberg")["Longitude"]
        .shift(1)
    )

    df["previous_date"] = (
        df.groupby("Iceberg")["Last Update"]
        .shift(1)
    )

    # Calculate changes in position.
    df["delta_lat"] = (
        df["Latitude"]
        - df["previous_latitude"]
    )

    df["delta_lon"] = (
        df["Longitude"]
        - df["previous_longitude"]
    )

    # Calculate distance travelled.
    df["distance_km"] = haversine_distance(
        df["previous_latitude"],
        df["previous_longitude"],
        df["Latitude"],
        df["Longitude"]
    )

    # Calculate time difference in hours.
    df["time_difference_hours"] = (
        df["Last Update"]
        - df["previous_date"]
    ).dt.total_seconds() / 3600

    # Calculate average movement speed.
    df["speed_kmh"] = (
        df["distance_km"]
        / df["time_difference_hours"]
    )

    # Calculate movement direction.
    df["bearing_deg"] = calculate_bearing(
        df["previous_latitude"],
        df["previous_longitude"],
        df["Latitude"],
        df["Longitude"]
    )

    return df


def main():

    print("=== Antarctic Iceberg Motion Calculation ===")
    print()

    print(f"Loading: {INPUT_FILE}")

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Last Update"]
    )

    print(f"Loaded {len(df)} observations.")

    df = calculate_motion(df)

    # The first observation for each iceberg has no previous
    # observation, so its movement values are undefined.
    movement_columns = [
        "previous_latitude",
        "previous_longitude",
        "previous_date",
        "delta_lat",
        "delta_lon",
        "distance_km",
        "time_difference_hours",
        "speed_kmh",
        "bearing_deg",
    ]

    # Save the complete dataset, including the first observation
    # for every iceberg.
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(f"Saved motion dataset to:")
    print(OUTPUT_FILE)

    print()
    print("=== Motion Summary ===")

    valid_motion = df[
        df["distance_km"].notna()
    ]

    print(
        f"Valid movement observations: "
        f"{len(valid_motion)}"
    )

    print(
        f"Average distance: "
        f"{valid_motion['distance_km'].mean():.2f} km"
    )

    print(
        f"Average speed: "
        f"{valid_motion['speed_kmh'].mean():.2f} km/h"
    )

    print()
    print("Processing complete.")


if __name__ == "__main__":
    main()