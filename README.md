ANTARCTIC NAVIGATION AI

AI/ML-Assisted Antarctic Navigation Decision Support System

A prototype decision-support platform for Antarctic research-vessel route planning.

The system combines iceberg trajectory predictions and NOAA sea-ice observations to evaluate alternative navigation routes using environmental hazard and route-distance factors.

OVERVIEW

Antarctic navigation is challenging because vessel routes can be affected by drifting icebergs, sea-ice concentration, and geographic constraints.

This prototype provides an interactive environment-aware route planning workflow:

Iceberg Predictions
+
NOAA Sea-Ice Observations
|
Environmental Risk Analysis
|
Candidate Route Generation
|
Land / Water Validation
|
Route Evaluation
|
+-------------------+
|                   |
Iceberg Hazard   Sea-Ice Hazard
|                   |
+---------+---------+
|
Distance + Risk Scoring
|
Recommended Route
|
Interactive Web Map

KEY FEATURES

* AI/ML-assisted iceberg trajectory prediction integration
* NOAA sea-ice concentration integration
* Date alignment between environmental datasets
* Multiple candidate navigation routes
* Iceberg proximity and hazard assessment
* Sea-ice hazard assessment
* Minimum iceberg-clearance calculation
* Land/water validation
* Rejection of non-navigable land-based start/destination locations
* Route distance and environmental-risk scoring
* Recommended route selection
* Interactive Antarctic map
* Manual latitude/longitude input
* Map-based start and destination selection
* Environmental data coverage/status reporting

TECHNOLOGY STACK

Backend:

* Python
* FastAPI
* Uvicorn
* Pandas
* NumPy

Frontend:

* HTML
* CSS
* JavaScript
* Leaflet.js

Data:

* Iceberg prediction data
* NOAA sea-ice concentration observations

INSTALLATION

Requirements:

* Python 3.12+ recommended
* Git
* Internet connection for map tiles

Python 3.12 is recommended for maximum compatibility with the prototype dependencies.

1. CLONE THE REPOSITORY

git clone [https://github.com/muni-iitdh/antarctica-navigation-ai.git](https://github.com/muni-iitdh/antarctica-navigation-ai.git)

cd antarctica-navigation-ai

2. CREATE A VIRTUAL ENVIRONMENT

Windows:

py -3.12 -m venv .venv

.venv\Scripts\activate

If "py" is unavailable but Python 3.12 is installed:

python -m venv .venv

.venv\Scripts\activate

macOS / Linux:

python3 -m venv .venv

source .venv/bin/activate

After activation, the terminal should show:

(.venv)

3. INSTALL DEPENDENCIES

With the virtual environment activated:

pip install -r requirements.txt

RUNNING THE APPLICATION

The application uses:

1. A FastAPI backend
2. A local HTTP server for the frontend

Both need to be running.

TERMINAL 1 — START FASTAPI

From the project root:

Windows:

python -m uvicorn api.main:app --reload

If using Python 3.12 explicitly:

py -3.12 -m uvicorn api.main:app --reload

macOS / Linux:

python3 -m uvicorn api.main:app --reload

The backend should start at:

[http://127.0.0.1:8000](http://127.0.0.1:8000)

You should see:

Uvicorn running on [http://127.0.0.1:8000](http://127.0.0.1:8000)
Application startup complete.

Keep this terminal running.

TERMINAL 2 — START THE FRONTEND

Open a second terminal.

Navigate to the project directory:

cd antarctica-navigation-ai

If the virtual environment is not active, activate it first.

Then run:

python -m http.server 5500

On macOS/Linux, this can also be:

python3 -m http.server 5500

You should see something similar to:

Serving HTTP on 0.0.0.0 port 5500

Keep this terminal running.

OPEN THE APPLICATION

Open a browser and go to:

[http://localhost:5500/index.html](http://localhost:5500/index.html)

The Antarctic Navigation dashboard should load.

USING THE PROTOTYPE

OPTION 1 — ENTER COORDINATES

Enter:

Start:

* Latitude
* Longitude

Destination:

* Latitude
* Longitude

Then click:

Optimize Route

OPTION 2 — SELECT LOCATIONS ON THE MAP

The prototype also supports map-based location selection.

Select Start:

Click:

Click Map to Select Start

Then click the desired location on the map.

The latitude and longitude fields will be populated automatically.

Select Destination:

Click:

Click Map to Select Destination

Then click another location on the map.

The destination coordinates will be populated automatically.

Finally click:

Optimize Route

ROUTE VALIDATION

The system validates whether selected locations are navigable water locations.

Land-based start or destination coordinates are rejected.

For invalid requests, the system returns an error instead of generating a vessel route across land.

The frontend also clears any previously displayed route after a failed route request.

ROUTE OPTIMIZATION

For a valid start and destination, the system generates multiple candidate routes.

The prototype evaluates routes using:

ICEBERG RISK

* Iceberg proximity
* Minimum route clearance
* Critical encounters
* High-risk encounters
* Medium-risk encounters

SEA-ICE RISK

* Sea-ice concentration
* Mean concentration along the route
* Maximum concentration
* Environmental data coverage

ROUTE EFFICIENCY

* Route distance

Distance is currently used as a first-order proxy for route efficiency.

The final route score combines environmental hazard and route distance to select the recommended feasible route.

ENVIRONMENTAL DATA

The prototype uses processed environmental datasets included in the repository.

The route optimizer aligns iceberg prediction data and NOAA sea-ice observations by date before evaluating candidate routes.

The system reports environmental-data availability and coverage rather than silently assuming that environmental observations exist everywhere.

AI / ML COMPONENT

The AI/ML component of the prototype is focused on iceberg trajectory prediction.

Predicted iceberg positions are supplied to the route evaluation pipeline.

These predictions are combined with sea-ice observations to estimate environmental risk along candidate navigation routes.

The route-selection layer then evaluates multiple feasible routes using environmental hazard and route distance.

EXAMPLE WORKFLOW

1. User selects start location
2. User selects destination
3. Validate navigable water
4. Load aligned environmental data
5. Generate candidate routes
6. Evaluate iceberg hazards
7. Evaluate sea-ice hazards
8. Calculate route scores
9. Compare feasible routes
10. Recommend route

API ENDPOINTS

The FastAPI backend provides the following main endpoints:

GET /

Backend status endpoint.

GET /icebergs

Provides iceberg prediction information used by the frontend.

GET /sea-ice

Provides sea-ice information used by the frontend.

GET /route/optimize

Evaluates candidate routes between a specified start and destination.

Example:

[http://127.0.0.1:8000/route/optimize?start_lat=-60&start_lon=-60&end_lat=-65&end_lon=-45](http://127.0.0.1:8000/route/optimize?start_lat=-60&start_lon=-60&end_lat=-65&end_lon=-45)

PROJECT STRUCTURE

antarctica-navigation-ai/
|
├── api/
│   └── main.py
|
├── data/
│   ├── iceberg/
│   └── sea_ice/
|
├── ml/
│   └── iceberg/
│       └── route_optimizer.py
|
├── index.html
├── requirements.txt
├── README.md
└── .gitignore

IMPORTANT NOTE

This project is a hackathon/research prototype for environmental route decision support.

It is not intended for direct operational vessel navigation.

The current prototype uses a limited set of environmental variables.

Future development can incorporate:

* Sea-ice forecasting
* Satellite time-series data
* Meteorological forecasts
* Ocean currents
* Wind
* Waves
* Vessel-specific fuel-consumption models
* Vessel speed and operational constraints
* More advanced dynamic path-planning algorithms

FUTURE SCOPE

The current prototype can be extended into a more comprehensive Antarctic navigation platform.

Current Prototype:

Iceberg Prediction
+
NOAA Sea Ice
+
Route Risk Analysis

Future System:

* Sea-Ice Forecasting
* Weather Forecasts
* Ocean Currents
* Wind and Waves
* Vessel-Specific Fuel Model
* Dynamic Route Replanning
* Advanced Global Path Optimization

PROJECT REPOSITORY

GitHub:

[https://github.com/muni-iitdh/antarctica-navigation-ai](https://github.com/muni-iitdh/antarctica-navigation-ai)

QUICK START — MACOS / LINUX

git clone [https://github.com/muni-iitdh/antarctica-navigation-ai.git](https://github.com/muni-iitdh/antarctica-navigation-ai.git)

cd antarctica-navigation-ai

python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

Start the backend in Terminal 1:

python3 -m uvicorn api.main:app --reload

Start the frontend in Terminal 2:

python3 -m http.server 5500

Open:

[http://localhost:5500/index.html](http://localhost:5500/index.html)

QUICK START — WINDOWS

git clone [https://github.com/muni-iitdh/antarctica-navigation-ai.git](https://github.com/muni-iitdh/antarctica-navigation-ai.git)

cd antarctica-navigation-ai

py -3.12 -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt

Start the backend:

py -3.12 -m uvicorn api.main:app --reload

Start the frontend in a second terminal:

python -m http.server 5500

Open:

[http://localhost:5500/index.html](http://localhost:5500/index.html)

PROTOTYPE STATUS

Working prototype.

Core implemented functionality:

* Environmental data integration
* Iceberg prediction integration
* Sea-ice hazard analysis
* Candidate route generation
* Route scoring
* Water/land validation
* Interactive map
* Coordinate-based routing
* Map-based location selection
* FastAPI backend
* Local frontend serving
