
"""Fetch EPL + other top leagues from Football-Data.co.uk and merge with your local CSVs."""
import argparse, os, io, pandas as pd, numpy as np, requests

# Football-data CSVs follow a pattern per season (e.g., 2023/24 EPL = E0.csv inside ZIPs or direct CSV season files).
# We’ll use their recent season CSV endpoints when available.
LEAGUE_CODES = {
    "EPL": ["E0"],
    "LaLiga": ["SP1"],
    "SerieA": ["I1"],
    "Bundesliga": ["D1"],
    "Ligue1": ["F1"]
}

# Known season years to attempt
SEASONS = ["2020-21","2021-22","2022-23","2023-24","2024-25"]

def try_download(league_code, season):
    # Newer football-data seasons often use e.g. https://www.football-data.co.uk/mmz4281/2425/E0.csv (YY format)
    season_key = {
        "2020-21": "2021",
        "2021-22": "2122",
        "2022-23": "2223",
        "2023-24": "2324",
        "2024-25": "2425"
    }[season]
    url = f"https://www.football-data.co.uk/mmz4281/{season_key}/{league_code}.csv"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200 and len(r.text) > 1000:
            return r.text
    except Exception:
        return None
    return None

def normalize(df):
    # Map standard columns
    colmap = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ["date"]: colmap[c] = "Date"
        if cl in ["hometeam","home team"]: colmap[c] = "HomeTeam"
        if cl in ["awayteam","away team"]: colmap[c] = "AwayTeam"
        if cl in ["fthg","homegoals","home goals"]: colmap[c] = "HomeGoals"
        if cl in ["ftag","awaygoals","away goals"]: colmap[c] = "AwayGoals"
        if cl in ["b365h","ph","whh","bbah","psh","maxh","avg h","avgh","homeodds"]: colmap[c] = "HomeOdds"
        if cl in ["b365d","pd","whd","bbad","psd","maxd","avg d","avgd","drawodds"]: colmap[c] = "DrawOdds"
        if cl in ["b365a","pa","wha","bbaa","psa","maxa","avg a","avga","awayodds"]: colmap[c] = "AwayOdds"
    df = df.rename(columns=colmap)
    keep = ['Date','HomeTeam','AwayTeam','HomeGoals','AwayGoals','HomeOdds','DrawOdds','AwayOdds']
    for k in keep:
        if k not in df.columns:
            df[k] = np.nan
    # Parse date
    try:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    except Exception:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date','HomeTeam','AwayTeam','HomeGoals','AwayGoals'])
    return df[keep]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include_top5", action="store_true", help="Fetch EPL, LaLiga, Serie A, Bundesliga, Ligue 1")
    ap.add_argument("--add_local", nargs="*", default=[], help="Paths to local CSVs to include (e.g., PSL/Nedbank)")
    ap.add_argument("--out", default="merged_matches.csv")
    args = ap.parse_args()

    frames = []
    if args.include_top5:
        for lg, codes in LEAGUE_CODES.items():
            for code in codes:
                for season in SEASONS:
                    txt = try_download(code, season)
                    if txt:
                        df = pd.read_csv(io.StringIO(txt))
                        df = normalize(df)
                        frames.append(df)

    for p in args.add_local:
        if os.path.exists(p):
            df = pd.read_csv(p)
            # Try to normalize
            # Map alternative names
            alt = df.rename(columns={
                'FTHG':'HomeGoals', 'FTAG':'AwayGoals',
                'Home':'HomeTeam', 'Away':'AwayTeam',
                'Score':'_drop_'
            })
            if 'Result' in alt.columns: alt = alt.drop(columns=['Result'])
            if '_drop_' in alt.columns: alt = alt.drop(columns=['_drop_'])
            # Ensure required columns exist
            need = ['Date','HomeTeam','AwayTeam','HomeGoals','AwayGoals']
            for k in need:
                if k not in alt.columns:
                    # Attempt another common mapping
                    if k=='Date' and 'date' in alt.columns: alt = alt.rename(columns={'date':'Date'})
            alt['Date'] = pd.to_datetime(alt['Date'], errors='coerce')
            alt = alt.dropna(subset=['Date','HomeTeam','AwayTeam','HomeGoals','AwayGoals'])
            frames.append(alt[['Date','HomeTeam','AwayTeam','HomeGoals','AwayGoals'] + [c for c in ['HomeOdds','DrawOdds','AwayOdds'] if c in alt.columns]])

    if not frames:
        raise SystemExit("No data fetched. Provide --include_top5 or --add_local files.")

    big = pd.concat(frames, ignore_index=True)
    big = big.sort_values('Date').reset_index(drop=True)
    big.to_csv(args.out, index=False)
    print("Wrote", args.out, "with rows:", len(big))

if __name__ == "__main__":
    main()
