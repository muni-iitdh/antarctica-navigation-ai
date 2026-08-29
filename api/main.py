from pathlib import Path
import math

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ICEBERG_FILE = (
    PROJECT_ROOT
    / "data"
    / "iceberg"
    / "processed"
    / "iceberg_baseline_predictions.csv"
)

COMBINED_RISK_FILE = (
    PROJECT_ROOT
    / "data"
    / "iceberg"
    / "processed"
    / "combined_environmental_risk.csv"
)

SEA_ICE_FILE = (
    PROJECT_ROOT
    / "data"
    / "sea_ice"
    / "processed"
    / "sea_ice_concentration.csv"
)

ROUTE_RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "iceberg"
    / "processed"
    / "route_optimizer_results.csv"
)


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Antarctic Navigation Decision Support API",
    description=(
        "Prototype API for iceberg trajectory, "
        "sea-ice risk and route optimization."
    ),
    version="1.0.0",
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# HELPERS
# =========================================================

def clean_value(value):

    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if hasattr(value, "isoformat"):
        return value.isoformat()

    return value


def dataframe_to_records(df):

    records = []

    for record in df.to_dict(
        orient="records"
    ):

        cleaned = {
            key: clean_value(value)
            for key, value in record.items()
        }

        records.append(cleaned)

    return records


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/")
def root():

    return {
        "system": "Antarctic Navigation Decision Support System",
        "status": "running",
        "endpoints": [
            "/icebergs",
            "/sea-ice",
            "/route/optimize",
        ],
    }


# =========================================================
# ICEBERGS
# =========================================================

@app.get("/icebergs")
def get_icebergs():

    if not ICEBERG_FILE.exists():

        raise HTTPException(
            status_code=500,
            detail=(
                "Iceberg prediction file not found: "
                f"{ICEBERG_FILE}"
            ),
        )

    df = pd.read_csv(
        ICEBERG_FILE
    )

    df["Last Update"] = pd.to_datetime(
        df["Last Update"]
    )

    # Use the latest available iceberg snapshot.
    latest_date = df[
        "Last Update"
    ].max()

    df = df[
        df["Last Update"]
        == latest_date
    ].copy()

    # Add the existing iceberg risk classification
    # if it is available in the processed risk file.
    if COMBINED_RISK_FILE.exists():

        risk_df = pd.read_csv(
            COMBINED_RISK_FILE
        )

        risk_df["Last Update"] = pd.to_datetime(
            risk_df["Last Update"]
        )

        risk_df = risk_df[
            risk_df["Last Update"]
            == latest_date
        ].copy()

        risk_columns = [
            "Iceberg",
            "iceberg_risk_score",
            "risk_level",
        ]

        available_columns = [
            column
            for column in risk_columns
            if column in risk_df.columns
        ]

        if "Iceberg" in available_columns:

            df = df.merge(
                risk_df[
                    available_columns
                ],
                on="Iceberg",
                how="left",
            )

    return {
        "analysis_date": latest_date.date().isoformat(),
        "count": len(df),
        "icebergs": dataframe_to_records(df),
    }


# =========================================================
# SEA ICE
# =========================================================

@app.get("/sea-ice")
def get_sea_ice():

    if not SEA_ICE_FILE.exists():

        raise HTTPException(
            status_code=500,
            detail=(
                "Sea-ice file not found: "
                f"{SEA_ICE_FILE}"
            ),
        )

    df = pd.read_csv(
        SEA_ICE_FILE
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    latest_date = df[
        "date"
    ].max()

    df = df[
        df["date"]
        == latest_date
    ].copy()

    # Downsample the approximately 1-degree grid.
    #
    # Every 4th latitude and longitude point is retained.
    # This dramatically reduces browser workload while
    # preserving the large-scale Antarctic ice pattern.
    df = df.iloc[
        ::4,
        :
    ].copy()

    return {
        "analysis_date": latest_date.date().isoformat(),
        "count": len(df),
        "downsampling": "every 4th grid point",
        "sea_ice": dataframe_to_records(df),
    }


# =========================================================
# ROUTE OPTIMIZATION
# =========================================================

@app.get("/route/optimize")
def optimize_route(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
):

    # -----------------------------------------------------
    # BASIC COORDINATE VALIDATION
    # -----------------------------------------------------

    if not (
        -90 <= start_lat <= -50
        and -90 <= end_lat <= -50
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Latitude must be between -90 and -50 "
                "for the Antarctic operating region."
            ),
        )

    if not (
        -180 <= start_lon <= 180
        and -180 <= end_lon <= 180
    ):
        raise HTTPException(
            status_code=400,
            detail="Longitude must be between -180 and 180.",
        )

    # -----------------------------------------------------
    # REQUIRED DATA FILES
    # -----------------------------------------------------

    if not ICEBERG_FILE.exists():
        raise HTTPException(
            status_code=500,
            detail="Iceberg prediction data not found.",
        )

    if not SEA_ICE_FILE.exists():
        raise HTTPException(
            status_code=500,
            detail="Sea-ice data not found.",
        )

    # -----------------------------------------------------
    # IMPORT ROUTE OPTIMIZER FUNCTIONS
    # -----------------------------------------------------

    from ml.iceberg.route_optimizer import (
        generate_candidate_routes,
        evaluate_route,
        prepare_sea_ice,
        select_sea_ice_date,
        is_navigable_water,
        route_is_over_water,
    )

    # -----------------------------------------------------
    # LAND / WATER VALIDATION
    # -----------------------------------------------------

    if not is_navigable_water(
        start_lat,
        start_lon
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Start location is on land. "
                "Please select a navigable water location."
            ),
        )

    if not is_navigable_water(
        end_lat,
        end_lon
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Destination is on land. "
                "Please select a navigable water location."
            ),
        )

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

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

    sea_ice_all = pd.read_csv(
        SEA_ICE_FILE
    )

    sea_ice_all = prepare_sea_ice(
        sea_ice_all
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
        iceberg_dates.intersection(
            sea_ice_dates
        )
    )

    if not common_dates:
        raise HTTPException(
            status_code=500,
            detail=(
                "No common date exists between "
                "iceberg and sea-ice datasets."
            ),
        )

    analysis_date = common_dates[-1]

    # -----------------------------------------------------
    # CURRENT DATA SNAPSHOT
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

    # -----------------------------------------------------
    # START / DESTINATION
    # -----------------------------------------------------

    start = (
        start_lat,
        start_lon,
    )

    destination = (
        end_lat,
        end_lon,
    )

    # -----------------------------------------------------
    # GENERATE CANDIDATE ROUTES
    # -----------------------------------------------------

    routes = generate_candidate_routes(
        start,
        destination
    )

    # -----------------------------------------------------
    # REMOVE ROUTES THAT CROSS LAND
    # -----------------------------------------------------

    routes = {
        name: route
        for name, route in routes.items()
        if route_is_over_water(route)
    }

    if not routes:
        raise HTTPException(
            status_code=400,
            detail=(
                "No feasible water-only route exists "
                "between the selected locations."
            ),
        )

    # -----------------------------------------------------
    # EVALUATE ROUTES
    # -----------------------------------------------------

    results = []

    route_geometries = {}

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

        route_geometries[name] = [
            {
                "lat": point[0],
                "lon": point[1],
            }
            for point in route
        ]

    results_df = pd.DataFrame(
        results
    )

    # -----------------------------------------------------
    # FINAL ROUTE SCORE
    # -----------------------------------------------------

    shortest_distance = (
        results_df[
            "distance_km"
        ].min()
    )

    if (
        pd.isna(shortest_distance)
        or shortest_distance <= 0
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Start and destination must be different; "
                "route distance must be greater than zero."
            ),
        )

    results_df[
        "distance_penalty"
    ] = (
        results_df[
            "distance_km"
        ]
        / shortest_distance
        - 1.0
    )

    results_df[
        "total_score"
    ] = float("nan")

    valid = (
        results_df[
            "score_status"
        ]
        == "FULL_ENVIRONMENTAL_DATA"
    )

    results_df.loc[
        valid,
        "total_score"
    ] = (
        results_df.loc[
            valid,
            "total_hazard_score"
        ]
        + 2.0
        * results_df.loc[
            valid,
            "distance_penalty"
        ]
    )

    results_df[
        "recommended"
    ] = False

    if valid.any():

        best_index = (
            results_df.loc[
                valid,
                "total_score"
            ].idxmin()
        )

        results_df.loc[
            best_index,
            "recommended"
        ] = True

    # -----------------------------------------------------
    # BUILD RESPONSE
    # -----------------------------------------------------

    candidates = []

    for _, row in results_df.iterrows():

        route_name = row[
            "route"
        ]

        route_result = {
            key: clean_value(value)
            for key, value
            in row.to_dict().items()
        }

        route_result[
            "geometry"
        ] = route_geometries[
            route_name
        ]

        candidates.append(
            route_result
        )

    recommended_rows = results_df[
        results_df[
            "recommended"
        ]
    ]

    if len(recommended_rows) > 0:

        recommended_name = (
            recommended_rows.iloc[
                0
            ]["route"]
        )

    else:

        recommended_name = None

    return {
        "analysis_date": (
            analysis_date
            .date()
            .isoformat()
        ),
        "start": {
            "lat": start_lat,
            "lon": start_lon,
        },
        "destination": {
            "lat": end_lat,
            "lon": end_lon,
        },
        "recommended_route": (
            recommended_name
        ),
        "candidate_routes": candidates,
    }