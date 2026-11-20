# Route Ranker — Updated Streamlit Dashboard

This version integrates the Optimization notebook's route-merging and route-splitting screening logic into the dashboard.

## Dashboard pages
- Overview
- Network Analytics
- Route Optimization
  - Route merging: overlap categories and recommendations
  - Route splitting: 90th-percentile route-length + median unique-stop screening
  - Optional stop-level midpoint split points
- Model Insights
  - K-Means
  - DBSCAN
  - XGBoost

## Run on a supported local environment
Use Python 3.12. Create a virtual environment, then install:
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Important Windows Application Control note
The user's Windows CodeIntegrity Event 3077 explicitly blocked NumPy's compiled extension `_multiarray_umath.cp312-win_amd64.pyd` under policy ID `0283ac0f-fff1-49ae-ada1-8a933130cad6`.

Streamlit itself depends on NumPy and Pandas. Therefore, this dashboard cannot guarantee a local Windows run while that policy continues to block NumPy. Do not try to bypass or disable Windows Application Control. Use one of these supported paths:
1. Have Windows/security policy reviewed and have the legitimate Python/NumPy package approved by the policy owner; or
2. Deploy the app to Streamlit Community Cloud, where it runs on a hosted Linux environment rather than your Windows laptop.

The app source does not manually import NumPy in its optimization module, but the Streamlit/scikit-learn/XGBoost stack requires NumPy.

## Route splitting data requirement
The Optimization notebook identifies split candidates from `route_features.csv` using:
- 90th percentile of `avg_route_length_km`
- Median of `total_unique_stops`
- Candidate route must meet both conditions

To calculate the actual candidate midpoint stop, the notebook additionally requires original GTFS `trips.txt` and `stop_times.txt`. These were not among the processed CSV files previously supplied. If available, place them in:
`data/gtfs/trips.txt`
`data/gtfs/stop_times.txt`

Without these two files, the dashboard still displays and exports split-screening candidates, but correctly does not invent stop-level split points.

## Data files
The five previously supplied processed CSVs are included in `data/`.
