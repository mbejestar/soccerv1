
# ⚽ Global Football Match Predictor — Pro Pack

This pack lets you build an **international 1X2 model** (PSL + EPL + other top leagues) and deploy it with Streamlit.

## Contents
- `app.py` — Streamlit UI (works with or without `model.pkl`)
- `model_utils.py` — helper functions
- `train_model.py` — trains on **your CSV** (single file)
- `fetch_and_merge.py` — **auto-downloads** public league CSVs (EPL, LaLiga, Serie A, Bundesliga, Ligue 1) from Football-Data.co.uk, merges with your PSL CSVs, and saves `merged_matches.csv`
- `requirements.txt` — Python deps
- `make_international_model.sh` — one-liner to fetch → merge → train

## Quick start
```bash
pip install -r requirements.txt

# Option A: Use your own combined CSV
python train_model.py --csv merged_matches.csv --out model.pkl

# Option B: Auto-fetch major leagues + merge with your PSL files
# 1) Put your PSL CSV(s) here (e.g., psl_2024_25_results_rounds1_10.csv)
# 2) Run:
python fetch_and_merge.py --include_top5 --add_local "psl_2024_25_results_rounds1_10.csv"
python train_model.py --csv merged_matches.csv --out model.pkl

# Launch app
streamlit run app.py
```

## Expected columns in CSV
Required:
```
Date, HomeTeam, AwayTeam, HomeGoals, AwayGoals
```
Optional (improves accuracy):
```
HomeOdds, DrawOdds, AwayOdds  # decimal odds
```

## Notes
- The training split is **time-based** (earliest 85% train → latest 15% test).
- Rolling features are **past-only** (no data leakage).
- Team names are one-hot encoded; the model is **league-agnostic**.
