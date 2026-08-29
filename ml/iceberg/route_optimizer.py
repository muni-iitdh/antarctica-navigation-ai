from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# PROJECT PATHS
# =========================================================

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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "iceberg"
    / "processed"
    / "route_optimizer_results.csv"
)


# =========================================================
# MODEL CONFIGURATION
# =========================================================

EARTH_RADIUS_KM = 6371.0

# Existing iceberg risk thresholds.
CRITICAL_RADIUS_KM = 2.0
HIGH_RADIUS_KM = 5.0
MEDIUM_RADIUS_KM = 10.0

# Existing environmental weighting.
ICEBERG_WEIGHT = 0.70
SEA_ICE_WEIGHT = 0.30

# Route sampling resolution.
# One observation approximately every 25 km.
ROUTE_SAMPLE_SPACING_KM = 25.0

# NOAA grid lookup tolerance.
# If the nearest NOAA grid cell is farther away than this,
# we do not pretend that the route has observed sea-ice data.
MAX_GRID_DISTANCE_KM = None

# Distance contribution to final route score.
# This is intentionally small compared with environmental hazard.
DISTANCE_WEIGHT = 2.0


# =========================================================
# HAVERSINE
# =========================================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):
    """
    Calculate great-circle distance between two
    latitude/longitude points in kilometres.
    """

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


# =========================================================
# POINT TO ROUTE SEGMENT
# =========================================================

def point_to_segment_distance(
    point_lat,
    point_lon,
    start_lat,
    start_lon,
    end_lat,
    end_lon
):
    """
    Approximate point-to-line-segment distance in km.

    The local equirectangular approximation is appropriate
    for these relatively short route segments.
    """

    mean_lat = np.radians(
        (
            start_lat
            + end_lat
            + point_lat
        ) / 3.0
    )

    km_per_lat = 111.32
    km_per_lon = (
        111.32
        * np.cos(mean_lat)
    )

    px = point_lon * km_per_lon
    py = point_lat * km_per_lat

    x1 = start_lon * km_per_lon
    y1 = start_lat * km_per_lat

    x2 = end_lon * km_per_lon
    y2 = end_lat * km_per_lat

    dx = x2 - x1
    dy = y2 - y1

    length_squared = (
        dx * dx
        + dy * dy
    )

    if length_squared == 0:

        return np.sqrt(
            (px - x1) ** 2
            + (py - y1) ** 2
        )

    t = (
        (px - x1) * dx
        + (py - y1) * dy
    ) / length_squared

    t = max(
        0.0,
        min(1.0, t)
    )

    closest_x = (
        x1
        + t * dx
    )

    closest_y = (
        y1
        + t * dy
    )

    return np.sqrt(
        (px - closest_x) ** 2
        + (py - closest_y) ** 2
    )


# =========================================================
# ROUTE DISTANCE
# =========================================================

def route_distance(route):
    """
    Total great-circle distance along a route.
    """

    total = 0.0

    for i in range(
        len(route) - 1
    ):

        lat1, lon1 = route[i]
        lat2, lon2 = route[i + 1]

        total += haversine_distance(
            lat1,
            lon1,
            lat2,
            lon2
        )

    return total


# =========================================================
# INTERPOLATE ROUTE
# =========================================================

def interpolate_segment(
    start,
    end,
    spacing_km
):
    """
    Generate approximately evenly spaced points
    along a route segment.
    """

    start_lat, start_lon = start
    end_lat, end_lon = end

    distance = haversine_distance(
        start_lat,
        start_lon,
        end_lat,
        end_lon
    )

    number_of_segments = max(
        1,
        int(
            np.ceil(
                distance
                / spacing_km
            )
        )
    )

    points = []

    for i in range(
        number_of_segments
        + 1
    ):

        fraction = (
            i
            / number_of_segments
        )

        lat = (
            start_lat
            + fraction
            * (
                end_lat
                - start_lat
            )
        )

        lon = (
            start_lon
            + fraction
            * (
                end_lon
                - start_lon
            )
        )

        points.append(
            (lat, lon)
        )

    return points


def sample_route(
    route,
    spacing_km=ROUTE_SAMPLE_SPACING_KM
):
    """
    Convert a multi-segment route into approximately
    equally spaced sample points.
    """

    samples = []

    for i in range(
        len(route) - 1
    ):

        segment_samples = (
            interpolate_segment(
                route[i],
                route[i + 1],
                spacing_km
            )
        )

        # Avoid duplicating waypoint points.
        if i > 0:
            segment_samples = (
                segment_samples[1:]
            )

        samples.extend(
            segment_samples
        )

    return samples


# =========================================================
# CANDIDATE ROUTES
# =========================================================

def generate_candidate_routes(
    start,
    end
):
    """
    Generate the same seven prototype candidate routes
    used previously.
    """

    start_lat, start_lon = start
    end_lat, end_lon = end

    midpoint_lat = (
        start_lat
        + end_lat
    ) / 2.0

    midpoint_lon = (
        start_lon
        + end_lon
    ) / 2.0

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

            name = (
                f"North +{offset:.1f}deg"
            )

            waypoint = (
                midpoint_lat
                + offset,
                midpoint_lon
            )

            routes[name] = [
                start,
                waypoint,
                end
            ]

        else:

            name = (
                f"South {offset:.1f}deg"
            )

            waypoint = (
                midpoint_lat
                + offset,
                midpoint_lon
            )

            routes[name] = [
                start,
                waypoint,
                end
            ]

    return routes


# =========================================================
# ICEBERG HAZARD
# =========================================================

def calculate_iceberg_hazard(
    route,
    icebergs
):
    """
    Evaluate predicted iceberg positions against the route.

    This retains the existing distance thresholds.
    """

    clearances = []

    for _, iceberg in icebergs.iterrows():

        minimum_distance = float(
            "inf"
        )

        for i in range(
            len(route) - 1
        ):

            distance = (
                point_to_segment_distance(
                    iceberg[
                        "predicted_latitude"
                    ],
                    iceberg[
                        "predicted_longitude"
                    ],
                    route[i][0],
                    route[i][1],
                    route[i + 1][0],
                    route[i + 1][1]
                )
            )

            minimum_distance = min(
                minimum_distance,
                distance
            )

        clearances.append(
            minimum_distance
        )

    clearances = np.asarray(
        clearances,
        dtype=float
    )

    critical = np.sum(
        clearances
        <= CRITICAL_RADIUS_KM
    )

    high = np.sum(
        (
            clearances
            > CRITICAL_RADIUS_KM
        )
        & (
            clearances
            <= HIGH_RADIUS_KM
        )
    )

    medium = np.sum(
        (
            clearances
            > HIGH_RADIUS_KM
        )
        & (
            clearances
            <= MEDIUM_RADIUS_KM
        )
    )

    # Continuous exposure component.
    exposure = np.sum(
        1.0
        / (
            clearances
            + 1.0
        )
    )

    iceberg_hazard_score = (
        critical * 10.0
        + high * 5.0
        + medium * 2.0
        + exposure
    )

    return {
        "minimum_clearance_km": float(
            np.min(clearances)
        ),
        "critical_encounters": int(
            critical
        ),
        "high_encounters": int(
            high
        ),
        "medium_encounters": int(
            medium
        ),
        "exposure_score": float(
            exposure
        ),
        "iceberg_hazard_score": float(
            iceberg_hazard_score
        ),
    }


# =========================================================
# SEA-ICE LOOKUP
# =========================================================

def prepare_sea_ice(
    sea_ice
):
    """
    Prepare NOAA sea-ice observations for nearest-grid
    lookup.
    """

    required_columns = {
        "date",
        "latitude",
        "longitude",
        "ice_concentration",
    }

    missing = (
        required_columns
        - set(sea_ice.columns)
    )

    if missing:

        raise ValueError(
            "Sea-ice dataset is missing "
            f"required columns: {sorted(missing)}"
        )

    sea_ice = sea_ice.copy()

    sea_ice["date"] = pd.to_datetime(
        sea_ice["date"]
    )

    sea_ice["latitude"] = pd.to_numeric(
        sea_ice["latitude"],
        errors="coerce"
    )

    sea_ice["longitude"] = pd.to_numeric(
        sea_ice["longitude"],
        errors="coerce"
    )

    sea_ice["ice_concentration"] = pd.to_numeric(
        sea_ice["ice_concentration"],
        errors="coerce"
    )

    sea_ice = sea_ice.dropna(
        subset=[
            "date",
            "latitude",
            "longitude",
            "ice_concentration",
        ]
    )

    # Concentration must be physically represented
    # between 0 and 1.
    sea_ice = sea_ice[
        (
            sea_ice[
                "ice_concentration"
            ] >= 0
        )
        & (
            sea_ice[
                "ice_concentration"
            ] <= 1
        )
    ].copy()

    return sea_ice


def select_sea_ice_date(
    sea_ice,
    analysis_date
):
    """
    Select the exact NOAA observation date used
    by the route calculation.
    """

    date = pd.Timestamp(
        analysis_date
    ).normalize()

    selected = sea_ice[
        sea_ice["date"].dt.normalize()
        == date
    ].copy()

    return selected


def nearest_sea_ice_observation(
    latitude,
    longitude,
    sea_ice
):
    """
    Find the nearest NOAA grid observation.

    Longitude is treated cyclically because the NOAA grid
    spans the full 0-360 degree longitude range.
    """

    if sea_ice.empty:
        return None

    lat_difference = (
        sea_ice["latitude"].to_numpy()
        - latitude
    )

    # Longitude distance must account for
    # the 0/360 boundary.
    lon_difference = np.abs(
        sea_ice["longitude"].to_numpy()
        - longitude
    )

    lon_difference = np.minimum(
        lon_difference,
        360.0 - lon_difference
    )

    mean_lat = np.radians(
        latitude
    )

    km_lat = (
        lat_difference
        * 111.32
    )

    km_lon = (
        lon_difference
        * 111.32
        * np.cos(mean_lat)
    )

    distances = np.sqrt(
        km_lat ** 2
        + km_lon ** 2
    )

    index = int(
        np.argmin(distances)
    )
   
    return {
        "distance_km": float(
            distances[index]
        ),
        "ice_concentration": float(
            sea_ice.iloc[
                index
            ][
                "ice_concentration"
            ]
        ),
    }


# =========================================================
# SEA-ICE ROUTE HAZARD
# =========================================================

def calculate_sea_ice_hazard(
    route,
    sea_ice
):
    """
    Sample the actual NOAA sea-ice field along the route.

    Environmental risk is based on the mean observed
    sea-ice concentration along route samples.

    Concentration:
        0.0 = no ice
        1.0 = full concentration

    Risk score:
        concentration * 100
    """

    samples = sample_route(
        route
    )

    concentrations = []
    grid_distances = []

    for latitude, longitude in samples:

        observation = (
            nearest_sea_ice_observation(
                latitude,
                longitude,
                sea_ice
            )
        )

        if observation is None:
            continue

                # Use the nearest valid NOAA grid observation.
        # The processed dataset is a spatial grid, so the
        # grid-cell center does not need to be within an
        # arbitrary distance threshold.
        concentrations.append(
            observation[
                "ice_concentration"
            ]
        )

        grid_distances.append(
            observation[
                "distance_km"
            ]
        )

    total_samples = len(
        samples
    )

    observed_samples = len(
        concentrations
    )

    if observed_samples == 0:

        return {
            "sea_ice_status": "UNKNOWN",
            "sea_ice_samples": 0,
            "sea_ice_coverage_percent": 0.0,
            "mean_ice_concentration": np.nan,
            "maximum_ice_concentration": np.nan,
            "sea_ice_hazard_score": np.nan,
            "nearest_grid_distance_km": np.nan,
        }

    concentrations = np.asarray(
        concentrations,
        dtype=float
    )

    coverage = (
        observed_samples
        / total_samples
        * 100.0
    )

    mean_concentration = float(
        np.mean(concentrations)
    )

    maximum_concentration = float(
        np.max(concentrations)
    )

    # Transparent prototype mapping:
    # concentration 0-1 -> risk score 0-100.
    sea_ice_hazard_score = (
        mean_concentration
        * 100.0
    )

    if coverage >= 90.0:

        status = "OBSERVED"

    else:

        status = "PARTIAL"

    return {
        "sea_ice_status": status,
        "sea_ice_samples": int(
            observed_samples
        ),
        "sea_ice_coverage_percent": float(
            coverage
        ),
        "mean_ice_concentration": (
            mean_concentration
        ),
        "maximum_ice_concentration": (
            maximum_concentration
        ),
        "sea_ice_hazard_score": (
            sea_ice_hazard_score
        ),
        "nearest_grid_distance_km": float(
            np.min(grid_distances)
        ),
    }


# =========================================================
# ROUTE EVALUATION
# =========================================================

def evaluate_route(
    name,
    route,
    icebergs,
    sea_ice
):

    distance = route_distance(
        route
    )

    iceberg_result = (
        calculate_iceberg_hazard(
            route,
            icebergs
        )
    )

    sea_ice_result = (
        calculate_sea_ice_hazard(
            route,
            sea_ice
        )
    )

    # -----------------------------------------------------
    # Combined environmental hazard
    # -----------------------------------------------------

    if (
        sea_ice_result[
            "sea_ice_status"
        ]
        == "OBSERVED"
    ):

        total_hazard = (
            ICEBERG_WEIGHT
            * iceberg_result[
                "iceberg_hazard_score"
            ]
            + SEA_ICE_WEIGHT
            * sea_ice_result[
                "sea_ice_hazard_score"
            ]
        )

        score_status = (
            "FULL_ENVIRONMENTAL_DATA"
        )

    elif (
        sea_ice_result[
            "sea_ice_status"
        ]
        == "PARTIAL"
    ):

        # We do not invent a sea-ice value.
        total_hazard = np.nan

        score_status = (
            "PARTIAL_ENVIRONMENTAL_DATA"
        )

    else:

        total_hazard = np.nan

        score_status = (
            "NO_ENVIRONMENTAL_DATA"
        )

    return {
        "route": name,
        "distance_km": distance,

        "minimum_clearance_km":
            iceberg_result[
                "minimum_clearance_km"
            ],

        "critical_encounters":
            iceberg_result[
                "critical_encounters"
            ],

        "high_encounters":
            iceberg_result[
                "high_encounters"
            ],

        "medium_encounters":
            iceberg_result[
                "medium_encounters"
            ],

        "iceberg_hazard_score":
            iceberg_result[
                "iceberg_hazard_score"
            ],

        "sea_ice_status":
            sea_ice_result[
                "sea_ice_status"
            ],

        "sea_ice_samples":
            sea_ice_result[
                "sea_ice_samples"
            ],

        "sea_ice_coverage_percent":
            sea_ice_result[
                "sea_ice_coverage_percent"
            ],

        "mean_ice_concentration":
            sea_ice_result[
                "mean_ice_concentration"
            ],

        "maximum_ice_concentration":
            sea_ice_result[
                "maximum_ice_concentration"
            ],

        "sea_ice_hazard_score":
            sea_ice_result[
                "sea_ice_hazard_score"
            ],

        "nearest_grid_distance_km":
            sea_ice_result[
                "nearest_grid_distance_km"
            ],

        "total_hazard_score":
            total_hazard,

        "score_status":
            score_status,
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "=== Antarctic Real-Data Route Optimizer ==="
    )

    # -----------------------------------------------------
    # LOAD ICEBERGS
    # -----------------------------------------------------

    print()
    print(
        f"Loading iceberg predictions: "
        f"{ICEBERG_FILE}"
    )

    icebergs_all = pd.read_csv(
        ICEBERG_FILE
    )

    icebergs_all[
        "Last Update"
    ] = pd.to_datetime(
        icebergs_all[
            "Last Update"
        ]
    )

    print(
        f"Loaded {len(icebergs_all)} "
        "predicted iceberg observations."
    )

    # -----------------------------------------------------
    # LOAD REAL NOAA SEA ICE
    # -----------------------------------------------------

    print()
    print(
        f"Loading NOAA sea-ice data: "
        f"{SEA_ICE_FILE}"
    )

    sea_ice_all = pd.read_csv(
        SEA_ICE_FILE
    )

    sea_ice_all = prepare_sea_ice(
        sea_ice_all
    )

    print(
        f"Loaded {len(sea_ice_all)} "
        "sea-ice grid observations."
    )

    # -----------------------------------------------------
    # FIND COMMON DATE
    # -----------------------------------------------------

    iceberg_dates = set(
        icebergs_all[
            "Last Update"
        ].dt.normalize()
    )

    sea_ice_dates = set(
        sea_ice_all[
            "date"
        ].dt.normalize()
    )

    common_dates = sorted(
        iceberg_dates
        .intersection(
            sea_ice_dates
        )
    )

    if not common_dates:

        raise RuntimeError(
            "No common observation date exists "
            "between iceberg predictions and "
            "NOAA sea-ice data."
        )

    analysis_date = common_dates[-1]

    # -----------------------------------------------------
    # CURRENT SNAPSHOTS
    # -----------------------------------------------------

    icebergs = icebergs_all[
        icebergs_all[
            "Last Update"
        ].dt.normalize()
        == analysis_date
    ].copy()

    sea_ice = select_sea_ice_date(
        sea_ice_all,
        analysis_date
    )

    print()
    print(
        "=== Time Alignment ==="
    )

    print(
        f"Analysis date: "
        f"{analysis_date.date()}"
    )

    print(
        f"Iceberg positions: "
        f"{len(icebergs)}"
    )

    print(
        f"NOAA sea-ice grid observations: "
        f"{len(sea_ice)}"
    )

    # -----------------------------------------------------
    # DEMO ROUTE
    # -----------------------------------------------------

    start = (
        -60.0,
        -60.0
    )

    destination = (
    -65.0,
    -45.0
)

    print()
    print(
        f"Start: {start}"
    )

    print(
        f"Destination: {destination}"
    )

    # -----------------------------------------------------
    # GENERATE ROUTES
    # -----------------------------------------------------

    routes = (
        generate_candidate_routes(
            start,
            destination
        )
    )

    print()
    print(
        f"Generated {len(routes)} "
        "candidate routes."
    )

    # -----------------------------------------------------
    # EVALUATE
    # -----------------------------------------------------

    results = []

    for name, route in routes.items():

        result = evaluate_route(
            name,
            route,
            icebergs,
            sea_ice
        )

        results.append(
            result
        )

    results = pd.DataFrame(
        results
    )

    # -----------------------------------------------------
    # DISTANCE PENALTY
    # -----------------------------------------------------

    shortest_distance = (
        results[
            "distance_km"
        ].min()
    )
    if (
    not np.isfinite(shortest_distance)
    or shortest_distance <= 0.0
):
        raise ValueError(
        "Start and destination must be different; "
        "route distance must be greater than zero."
    )

    results[
        "distance_penalty"
    ] = (
        results[
            "distance_km"
        ]
        / shortest_distance
        - 1.0
    )

    # -----------------------------------------------------
    # FINAL SCORE
    # -----------------------------------------------------

    results[
        "total_score"
    ] = np.nan

    valid = (
        results[
            "score_status"
        ]
        == "FULL_ENVIRONMENTAL_DATA"
    )

    results.loc[
        valid,
        "total_score"
    ] = (
        results.loc[
            valid,
            "total_hazard_score"
        ]
        + DISTANCE_WEIGHT
        * results.loc[
            valid,
            "distance_penalty"
        ]
    )

    # -----------------------------------------------------
    # RECOMMENDATION
    # -----------------------------------------------------

    results[
        "recommended"
    ] = False

    if valid.any():

        valid_results = (
            results[
                valid
            ]
            .sort_values(
                "total_score"
            )
        )

        recommended_route = (
            valid_results.iloc[
                0
            ]["route"]
        )

        results.loc[
            results["route"]
            == recommended_route,
            "recommended"
        ] = True

    # -----------------------------------------------------
    # SORT
    # -----------------------------------------------------

    results[
        "_sort_score"
    ] = results[
        "total_score"
    ].fillna(
        np.inf
    )

    results = (
        results
        .sort_values(
            "_sort_score"
        )
        .drop(
            columns=[
                "_sort_score"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # -----------------------------------------------------
    # DISPLAY
    # -----------------------------------------------------

    print()
    print(
        "=== Candidate Route Results ==="
    )

    display_columns = [
        "route",
        "distance_km",
        "minimum_clearance_km",
        "critical_encounters",
        "high_encounters",
        "medium_encounters",
        "iceberg_hazard_score",
        "sea_ice_status",
        "sea_ice_samples",
        "sea_ice_coverage_percent",
        "mean_ice_concentration",
        "maximum_ice_concentration",
        "sea_ice_hazard_score",
        "total_hazard_score",
        "total_score",
        "recommended",
    ]

    print(
        results[
            display_columns
        ].to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # RECOMMENDATION
    # -----------------------------------------------------

    print()
    print(
        "=== Recommendation ==="
    )

    if valid.any():

        recommended = (
            results[
                results[
                    "recommended"
                ]
            ].iloc[0]
        )

        print(
            f"Recommended route: "
            f"{recommended['route']}"
        )

        print(
            f"Analysis date: "
            f"{analysis_date.date()}"
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
            f"Critical encounters: "
            f"{int(recommended['critical_encounters'])}"
        )

        print(
            f"High encounters: "
            f"{int(recommended['high_encounters'])}"
        )

        print(
            f"Medium encounters: "
            f"{int(recommended['medium_encounters'])}"
        )

        print(
            f"Mean sea-ice concentration: "
            f"{recommended['mean_ice_concentration']:.3f}"
        )

        print(
            f"Maximum sea-ice concentration: "
            f"{recommended['maximum_ice_concentration']:.3f}"
        )

        print(
            f"Sea-ice route coverage: "
            f"{recommended['sea_ice_coverage_percent']:.1f}%"
        )

        print(
            f"Sea-ice hazard: "
            f"{recommended['sea_ice_hazard_score']:.2f}"
        )

        print(
            f"Combined hazard: "
            f"{recommended['total_hazard_score']:.2f}"
        )

        print(
            f"Final route score: "
            f"{recommended['total_score']:.2f}"
        )

        print(
            "Environmental data status: "
            "FULLY OBSERVED"
        )

    else:

        print(
            "NO ROUTE RECOMMENDED"
        )

        print(
            "Reason: no candidate route has "
            "complete NOAA sea-ice coverage."
        )

    # -----------------------------------------------------
    # OUTPUT
    # -----------------------------------------------------

    print()
    print(
        "Saved results to:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "=== Real-Data Route Optimization Complete ==="
    )


if __name__ == "__main__":
    main()