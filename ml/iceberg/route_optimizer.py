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

OUTPUT_FILE = OUTPUT_DIR / "route_optimizer_results.csv"


EARTH_RADIUS_KM = 6371.0


# ---------------------------------------------------------
# Prototype thresholds
# ---------------------------------------------------------

CRITICAL_RADIUS_KM = 2.0
HIGH_RADIUS_KM = 5.0
MEDIUM_RADIUS_KM = 10.0


# ---------------------------------------------------------
# Demonstration start/end
# ---------------------------------------------------------

START = (-60.0, -60.0)
END = (-65.0, -45.0)


# ---------------------------------------------------------
# Geographic distance
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
# Point → route segment distance
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
# Generate candidate routes
#
# We create routes by moving a waypoint perpendicular
# to the direct start→end direction.
# ---------------------------------------------------------

def generate_candidate_routes(
    start,
    end
):

    start_lat, start_lon = start
    end_lat, end_lon = end

    midpoint_lat = (
        start_lat + end_lat
    ) / 2

    midpoint_lon = (
        start_lon + end_lon
    ) / 2

    # Latitude offsets in degrees.
    #
    # Positive = northward detour.
    # Negative = southward detour.
    offsets = [
        0,
        0.5,
        1.0,
        1.5,
        -0.5,
        -1.0,
        -1.5,
    ]

    routes = {}

    for offset in offsets:

        if offset == 0:

            routes["Direct"] = [
                start,
                end
            ]

        elif offset > 0:

            name = f"North +{offset:.1f}deg"

            waypoint = (
                midpoint_lat + offset,
                midpoint_lon
            )

            routes[name] = [
                start,
                waypoint,
                end
            ]

        else:

            name = f"South {offset:.1f}deg"

            waypoint = (
                midpoint_lat + offset,
                midpoint_lon
            )

            routes[name] = [
                start,
                waypoint,
                end
            ]

    return routes


# ---------------------------------------------------------
# Evaluate one route
# ---------------------------------------------------------

def evaluate_route(
    name,
    route,
    icebergs
):

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

    critical = np.sum(
        clearances <= CRITICAL_RADIUS_KM
    )

    high = np.sum(
        (clearances > CRITICAL_RADIUS_KM)
        & (clearances <= HIGH_RADIUS_KM)
    )

    medium = np.sum(
        (clearances > HIGH_RADIUS_KM)
        & (clearances <= MEDIUM_RADIUS_KM)
    )

    # Continuous exposure score.
    #
    # Smaller clearance means larger risk.
    # The +1 prevents division by zero.
    exposure = np.sum(
        1.0 / (clearances + 1.0)
    )

    # Weighted hazard score.
    hazard_score = (
        critical * 10
        + high * 5
        + medium * 2
        + exposure
    )

    return {
        "route": name,
        "distance_km": route_length,
        "minimum_clearance_km": clearances.min(),
        "critical_encounters": int(critical),
        "high_encounters": int(high),
        "medium_encounters": int(medium),
        "exposure_score": exposure,
        "hazard_score": hazard_score,
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=== Antarctic Automatic Route Optimizer ===")
    print()

    print(f"Loading: {INPUT_FILE}")

    icebergs = pd.read_csv(INPUT_FILE)

    print(
        f"Loaded {len(icebergs)} predicted iceberg positions."
    )

    print()
    print(
        f"Start: {START}"
    )

    print(
        f"Destination: {END}"
    )

    # -----------------------------------------------------
    # Generate routes
    # -----------------------------------------------------

    routes = generate_candidate_routes(
        START,
        END
    )

    print()
    print(
        f"Generated {len(routes)} candidate routes."
    )

    # -----------------------------------------------------
    # Evaluate routes
    # -----------------------------------------------------

    results = []

    for name, route in routes.items():

        result = evaluate_route(
            name,
            route,
            icebergs
        )

        results.append(result)

    results = pd.DataFrame(results)

    # -----------------------------------------------------
    # Normalize distance.
    #
    # We don't want a route that is extremely long to win
    # merely because it avoids everything.
    # -----------------------------------------------------

    shortest_distance = (
        results["distance_km"].min()
    )

    results["distance_penalty"] = (
        results["distance_km"]
        / shortest_distance
    )

    # -----------------------------------------------------
    # Combined objective.
    #
    # Hazard is strongly preferred, but excessive distance
    # is penalized.
    # -----------------------------------------------------

    results["total_score"] = (
        results["hazard_score"]
        + 2.0 * (
            results["distance_penalty"] - 1.0
        )
    )

    results = results.sort_values(
        "total_score"
    ).reset_index(drop=True)

    results["recommended"] = False

    results.loc[
        results.index[0],
        "recommended"
    ] = True

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
    print("=== Candidate Route Results ===")

    print(
        results[
            [
                "route",
                "distance_km",
                "minimum_clearance_km",
                "critical_encounters",
                "high_encounters",
                "medium_encounters",
                "hazard_score",
                "total_score",
                "recommended"
            ]
        ].to_string(index=False)
    )

    print()

    recommended = results.iloc[0]

    print("=== Recommendation ===")

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
        f"Hazard score: "
        f"{recommended['hazard_score']:.2f}"
    )

    print()
    print("Saved results to:")
    print(OUTPUT_FILE)

    print()
    print("=== Optimization Complete ===")


if __name__ == "__main__":
    main()