from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Paths
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

OUTPUT_FILE = OUTPUT_DIR / "route_comparison.csv"


EARTH_RADIUS_KM = 6371.0


# ---------------------------------------------------------
# Prototype risk thresholds
# ---------------------------------------------------------

CRITICAL_RADIUS_KM = 2.0
HIGH_RADIUS_KM = 5.0
MEDIUM_RADIUS_KM = 10.0


# ---------------------------------------------------------
# Test start/end
# ---------------------------------------------------------

START = (-60.0, -60.0)
END = (-65.0, -45.0)


# ---------------------------------------------------------
# Candidate routes
#
# Route 1 = direct/shortest
# Route 2 = northern detour
# ---------------------------------------------------------

ROUTES = {
    "Direct": [
        START,
        END,
    ],

    "Safer Detour": [
        START,
        (-58.5, -55.0),
        (-61.0, -48.0),
        END,
    ],
}


# ---------------------------------------------------------
# Haversine distance
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
# Local point-to-segment distance
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

    km_per_lat = 111.32
    km_per_lon = 111.32 * np.cos(mean_lat)

    px = point_lon * km_per_lon
    py = point_lat * km_per_lat

    x1 = start_lon * km_per_lon
    y1 = start_lat * km_per_lat

    x2 = end_lon * km_per_lon
    y2 = end_lat * km_per_lat

    dx = x2 - x1
    dy = y2 - y1

    length_squared = dx * dx + dy * dy

    if length_squared == 0:
        return np.sqrt(
            (px - x1) ** 2 +
            (py - y1) ** 2
        )

    t = (
        (px - x1) * dx +
        (py - y1) * dy
    ) / length_squared

    t = max(0.0, min(1.0, t))

    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    return np.sqrt(
        (px - closest_x) ** 2 +
        (py - closest_y) ** 2
    )


# ---------------------------------------------------------
# Route distance
# ---------------------------------------------------------

def route_distance(route):

    total = 0.0

    for i in range(len(route) - 1):

        lat1, lon1 = route[i]
        lat2, lon2 = route[i + 1]

        total += haversine_distance(
            lat1,
            lon1,
            lat2,
            lon2
        )

    return total


# ---------------------------------------------------------
# Risk classification
# ---------------------------------------------------------

def classify_risk(distance):

    if distance <= CRITICAL_RADIUS_KM:
        return "CRITICAL"

    if distance <= HIGH_RADIUS_KM:
        return "HIGH"

    if distance <= MEDIUM_RADIUS_KM:
        return "MEDIUM"

    return "LOW"


# ---------------------------------------------------------
# Analyze route
# ---------------------------------------------------------

def analyze_route(name, route, icebergs):

    route_length = route_distance(route)

    clearances = []

    for _, iceberg in icebergs.iterrows():

        minimum_distance = float("inf")

        for i in range(len(route) - 1):

            start_lat, start_lon = route[i]
            end_lat, end_lon = route[i + 1]

            distance = point_to_segment_distance(
                iceberg["predicted_latitude"],
                iceberg["predicted_longitude"],
                start_lat,
                start_lon,
                end_lat,
                end_lon
            )

            minimum_distance = min(
                minimum_distance,
                distance
            )

        clearances.append(minimum_distance)

    clearances = np.array(clearances)

    risk_levels = np.array(
        [classify_risk(d) for d in clearances]
    )

    critical = np.sum(
        risk_levels == "CRITICAL"
    )

    high = np.sum(
        risk_levels == "HIGH"
    )

    medium = np.sum(
        risk_levels == "MEDIUM"
    )

    # Weighted prototype risk score.
    #
    # Higher score = more route hazard exposure.
    risk_score = (
        critical * 10
        + high * 5
        + medium * 2
    )

    return {
        "route": name,
        "distance_km": route_length,
        "minimum_clearance_km": clearances.min(),
        "critical_encounters": critical,
        "high_encounters": high,
        "medium_encounters": medium,
        "risk_score": risk_score,
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=== Antarctic Route Comparison ===")
    print()

    print(f"Loading: {INPUT_FILE}")

    icebergs = pd.read_csv(INPUT_FILE)

    print(
        f"Loaded {len(icebergs)} predicted iceberg positions."
    )

    results = []

    for name, route in ROUTES.items():

        result = analyze_route(
            name,
            route,
            icebergs
        )

        results.append(result)

    results = pd.DataFrame(results)

    # -----------------------------------------------------
    # Rank routes.
    #
    # First minimize risk score.
    # If risk is equal, prefer shorter route.
    # -----------------------------------------------------

    results = results.sort_values(
        ["risk_score", "distance_km"]
    ).reset_index(drop=True)

    results["recommended"] = False

    if len(results) > 0:
        results.loc[0, "recommended"] = True

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
    # Display
    # -----------------------------------------------------

    print()
    print("=== Route Results ===")
    print(
        results.to_string(index=False)
    )

    print()
    print("=== Recommendation ===")

    recommended = results.iloc[0]

    print(
        f"Recommended route: "
        f"{recommended['route']}"
    )

    print(
        f"Distance: "
        f"{recommended['distance_km']:.2f} km"
    )

    print(
        f"Minimum clearance: "
        f"{recommended['minimum_clearance_km']:.2f} km"
    )

    print(
        f"Risk score: "
        f"{recommended['risk_score']:.0f}"
    )

    print()
    print("Saved comparison to:")
    print(OUTPUT_FILE)

    print()
    print("=== Route Comparison Complete ===")


if __name__ == "__main__":
    main()