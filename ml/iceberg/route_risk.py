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

OUTPUT_FILE = OUTPUT_DIR / "route_risk_test.csv"


EARTH_RADIUS_KM = 6371.0

# Prototype hazard thresholds.
CRITICAL_RADIUS_KM = 2.0
HIGH_RADIUS_KM = 5.0
MEDIUM_RADIUS_KM = 10.0


# ---------------------------------------------------------
# Distance between two geographic points
# ---------------------------------------------------------

def haversine_distance(lat1, lon1, lat2, lon2):

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

    return EARTH_RADIUS_KM * c


# ---------------------------------------------------------
# Distance from a point to a route segment
#
# For the prototype we approximate the local Antarctic
# area using latitude/longitude converted to kilometres.
# ---------------------------------------------------------

def point_to_segment_distance(
    point_lat,
    point_lon,
    start_lat,
    start_lon,
    end_lat,
    end_lon
):

    mean_lat = np.radians(
        (start_lat + end_lat + point_lat) / 3
    )

    km_per_degree_lat = 111.32

    km_per_degree_lon = (
        111.32 * np.cos(mean_lat)
    )

    px = point_lon * km_per_degree_lon
    py = point_lat * km_per_degree_lat

    x1 = start_lon * km_per_degree_lon
    y1 = start_lat * km_per_degree_lat

    x2 = end_lon * km_per_degree_lon
    y2 = end_lat * km_per_degree_lat

    dx = x2 - x1
    dy = y2 - y1

    segment_length_squared = dx * dx + dy * dy

    if segment_length_squared == 0:
        return np.sqrt(
            (px - x1) ** 2 +
            (py - y1) ** 2
        )

    t = (
        (px - x1) * dx +
        (py - y1) * dy
    ) / segment_length_squared

    t = max(0.0, min(1.0, t))

    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    return np.sqrt(
        (px - closest_x) ** 2 +
        (py - closest_y) ** 2
    )


# ---------------------------------------------------------
# Risk classification
# ---------------------------------------------------------

def classify_risk(distance_km):

    if distance_km <= CRITICAL_RADIUS_KM:
        return "CRITICAL"

    if distance_km <= HIGH_RADIUS_KM:
        return "HIGH"

    if distance_km <= MEDIUM_RADIUS_KM:
        return "MEDIUM"

    return "LOW"


# ---------------------------------------------------------
# Route analysis
# ---------------------------------------------------------

def analyze_route(
    df,
    start_lat,
    start_lon,
    end_lat,
    end_lon
):

    results = []

    for _, row in df.iterrows():

        distance = point_to_segment_distance(
            row["predicted_latitude"],
            row["predicted_longitude"],
            start_lat,
            start_lon,
            end_lat,
            end_lon
        )

        risk = classify_risk(distance)

        results.append({
            "Iceberg": row["Iceberg"],
            "Last Update": row["Last Update"],
            "predicted_latitude": row["predicted_latitude"],
            "predicted_longitude": row["predicted_longitude"],
            "route_clearance_km": distance,
            "risk_level": risk
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=== Antarctic Route Risk Analysis ===")
    print()

    print(f"Loading: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(
        f"Loaded {len(df)} predicted iceberg positions."
    )

    # -----------------------------------------------------
    # DEMONSTRATION ROUTE
    #
    # These are only test coordinates used to validate
    # the route-risk calculation.
    #
    # The frontend will eventually provide real
    # start/end coordinates.
    # -----------------------------------------------------

    start_lat = -60.0
    start_lon = -60.0

    end_lat = -65.0
    end_lon = -45.0

    print()
    print("Test route:")
    print(
        f"Start:      ({start_lat}, {start_lon})"
    )
    print(
        f"Destination: ({end_lat}, {end_lon})"
    )

    # -----------------------------------------------------
    # Analyze route
    # -----------------------------------------------------

    results = analyze_route(
        df,
        start_lat,
        start_lon,
        end_lat,
        end_lon
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print()
    print("=== Route Risk Summary ===")

    print(
        results["risk_level"]
        .value_counts()
        .to_string()
    )

    print()
    print("Closest iceberg trajectories:")

    print(
        results
        .sort_values("route_clearance_km")
        .head(15)
        .to_string(index=False)
    )

    print()
    print(
        f"Minimum route clearance: "
        f"{results['route_clearance_km'].min():.2f} km"
    )

    print()
    print("Saved route analysis to:")
    print(OUTPUT_FILE)

    print()
    print("=== Route Risk Analysis Complete ===")


if __name__ == "__main__":
    main()