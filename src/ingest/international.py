"""
Pull international match and player data.

StatsBomb Open Data (event-level xG, shots, passes, pressures):
  - FIFA World Cup 2022  (competition_id=43,  season_id=106)
  - FIFA World Cup 2018  (competition_id=43,  season_id=3)
  - UEFA Euro 2024       (competition_id=55,  season_id=282)
  - UEFA Euro 2020       (competition_id=55,  season_id=43)   ← 360 data
  - Copa América 2024    (competition_id=223, season_id=282)
  - AFCON 2023           (competition_id=1267,season_id=107)

UEFA Nations League (results + goals only — not in StatsBomb free tier):
  - Pulled from martinbel/football-data GitHub repo (CSV, CC BY 4.0)
  - Editions: 2022/23, 2024/25
  - No xG available; contributes to team form via results only
  - Config weight: 0.55 (lower than tournaments because of data quality)

Events come back as a single flat DataFrame (not split by type).
"""

import warnings
import pandas as pd
import requests
from io import StringIO
from pathlib import Path
from statsbombpy import sb
import yaml

warnings.filterwarnings("ignore")  # suppress NoAuthWarning spam

CFG = yaml.safe_load(open(Path(__file__).parents[2] / "configs/config.yaml"))
RAW_DIR = Path(__file__).parents[2] / CFG["paths"]["raw_international"]
WEIGHTS = CFG["tournament_weights"]

COMPETITIONS = [
    {"name": "wc_2022",    "competition_id": 43,   "season_id": 106, "type": "world_cup"},
    {"name": "wc_2018",    "competition_id": 43,   "season_id": 3,   "type": "world_cup"},
    {"name": "euro_2024",  "competition_id": 55,   "season_id": 282, "type": "continental_tournament"},
    {"name": "euro_2020",  "competition_id": 55,   "season_id": 43,  "type": "continental_tournament"},
    {"name": "copa_2024",  "competition_id": 223,  "season_id": 282, "type": "continental_tournament"},
    {"name": "afcon_2023", "competition_id": 1267, "season_id": 107, "type": "continental_tournament"},
]


def _pull_match_level(comp: dict) -> pd.DataFrame:
    matches = sb.matches(competition_id=comp["competition_id"],
                         season_id=comp["season_id"])
    if matches.empty:
        return pd.DataFrame()
    matches["tournament_name"] = comp["name"]
    matches["tournament_type"] = comp["type"]
    matches["tournament_weight"] = WEIGHTS.get(comp["type"], 0.5)
    return matches


def _summarise_match_events(match_id: int, tournament_name: str,
                             tournament_weight: float) -> pd.DataFrame:
    """
    Pull all events for one match and aggregate to per-player summary rows.
    Returns one row per player with xG, goals, shots, key passes, pressures.
    """
    try:
        events = sb.events(match_id=match_id)
    except Exception as e:
        print(f"  [WARN] match {match_id}: {e}")
        return pd.DataFrame()

    if events.empty:
        return pd.DataFrame()

    rows = {}

    def get_player(r):
        return r.get("player", None)

    def ensure(player, team):
        if player not in rows:
            rows[player] = {
                "match_id": match_id,
                "player": player,
                "team": team,
                "shots": 0, "goals": 0, "xg": 0.0,
                "key_passes": 0, "assists": 0, "pressures": 0,
                "tournament_name": tournament_name,
                "tournament_weight": tournament_weight,
            }

    # Shots → xG
    shots = events[events["type"] == "Shot"]
    for _, r in shots.iterrows():
        p = r.get("player")
        if not p or pd.isna(p):
            continue
        ensure(p, r.get("team"))
        rows[p]["shots"] += 1
        rows[p]["xg"] += float(r.get("shot_statsbomb_xg", 0) or 0)
        if r.get("shot_outcome") == "Goal" or str(r.get("shot_outcome_id")) == "98":
            rows[p]["goals"] += 1

    # Passes → key passes & assists
    passes = events[events["type"] == "Pass"]
    for _, r in passes.iterrows():
        p = r.get("player")
        if not p or pd.isna(p):
            continue
        ensure(p, r.get("team"))
        if r.get("pass_shot_assist") is True or r.get("pass_key_pass") is True:
            rows[p]["key_passes"] += 1
        if r.get("pass_goal_assist") is True:
            rows[p]["assists"] += 1

    # Pressures
    pressures = events[events["type"] == "Pressure"]
    for _, r in pressures.iterrows():
        p = r.get("player")
        if not p or pd.isna(p):
            continue
        ensure(p, r.get("team"))
        rows[p]["pressures"] += 1

    return pd.DataFrame(list(rows.values()))


def pull_competition(comp: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\nPulling {comp['name']}...")

    match_path = RAW_DIR / f"{comp['name']}_matches.parquet"
    events_path = RAW_DIR / f"{comp['name']}_player_events.parquet"

    if match_path.exists() and events_path.exists():
        print(f"  [SKIP] {comp['name']} already cached")
        return pd.read_parquet(match_path), pd.read_parquet(events_path)

    matches = _pull_match_level(comp)
    if matches.empty:
        print(f"  [WARN] No matches for {comp['name']}")
        return pd.DataFrame(), pd.DataFrame()

    print(f"  {len(matches)} matches found. Fetching events...")
    event_frames = []
    for i, match_id in enumerate(matches["match_id"].tolist(), 1):
        print(f"    [{i}/{len(matches)}] match {match_id}", end="\r")
        df = _summarise_match_events(match_id, comp["name"],
                                     WEIGHTS.get(comp["type"], 0.5))
        if not df.empty:
            event_frames.append(df)

    print()
    player_events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    if not player_events.empty:
        player_events["tournament_type"] = comp["type"]

    matches.to_parquet(match_path, index=False)
    player_events.to_parquet(events_path, index=False)
    print(f"  Saved {len(matches)} matches, {len(player_events)} player-event rows.")

    return matches, player_events


# UEFA Nations League — match results from martinbel/football-data (no xG)
# Each CSV has: Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR columns
NATIONS_LEAGUE_URLS = {
    "nations_league_2022_23": "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
}

# Fallback: direct CSV from openfootball or similar
NATIONS_LEAGUE_FALLBACK = [
    {
        "name": "nations_league_2022_23",
        "url": "https://raw.githubusercontent.com/jokecamp/FootballData/master/other/UEFANationsLeague/UEFANationsLeague2022-23.csv",
    },
    {
        "name": "nations_league_2024_25",
        "url": "https://raw.githubusercontent.com/jokecamp/FootballData/master/other/UEFANationsLeague/UEFANationsLeague2024-25.csv",
    },
]


def pull_nations_league() -> pd.DataFrame:
    """
    Pull UEFA Nations League match results (results + goals, no xG).
    Uses martj42/international_results which covers all international
    fixtures including Nations League from 1872 onward.

    We filter to Nations League matches from 2022 onward.
    """
    out_path = RAW_DIR / "nations_league_results.parquet"
    if out_path.exists():
        print("  [SKIP] Nations League already cached")
        return pd.read_parquet(out_path)

    print("\nPulling Nations League results (martj42/international_results)...")
    url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [WARN] Could not fetch Nations League data: {e}")
        return pd.DataFrame()

    df = pd.read_csv(StringIO(resp.text))
    df.columns = [c.lower().strip() for c in df.columns]

    # Filter to Nations League matches from 2022 onward
    # martj42 dataset has a 'tournament' column
    if "tournament" in df.columns:
        nl = df[
            df["tournament"].str.contains("Nations League", case=False, na=False) &
            (pd.to_datetime(df["date"], errors="coerce") >= "2022-01-01")
        ].copy()
    else:
        print("  [WARN] No 'tournament' column found — cannot filter Nations League")
        return pd.DataFrame()

    nl["tournament_type"] = "nations_league"
    nl["tournament_weight"] = WEIGHTS.get("nations_league", 0.55)
    nl["tournament_name"] = nl["date"].str[:4].apply(
        lambda y: f"nations_league_{y}"
    )

    # Standardise columns to match StatsBomb match output
    nl = nl.rename(columns={
        "home_team": "home_team_name",
        "away_team": "away_team_name",
        "home_score": "home_score",
        "away_score": "away_score",
    })

    nl.to_parquet(out_path, index=False)
    print(f"  Saved {len(nl)} Nations League matches")
    return nl


def pull_all_international() -> tuple[pd.DataFrame, pd.DataFrame]:
    all_matches, all_events = [], []

    # StatsBomb competitions (event-level xG)
    for comp in COMPETITIONS:
        matches, events = pull_competition(comp)
        if not matches.empty:
            all_matches.append(matches)
        if not events.empty:
            all_events.append(events)

    # Nations League (results only, no xG — different source)
    nl = pull_nations_league()
    if not nl.empty:
        all_matches.append(nl)
        # No player-level events available for Nations League

    matches_combined = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()
    events_combined  = pd.concat(all_events,  ignore_index=True) if all_events  else pd.DataFrame()

    matches_combined.to_parquet(RAW_DIR / "all_international_matches.parquet", index=False)
    events_combined.to_parquet(RAW_DIR / "all_international_player_events.parquet", index=False)

    print(f"\nDone. {len(matches_combined)} total international matches "
          f"({len(events_combined)} with player-level event data).")
    return matches_combined, events_combined


if __name__ == "__main__":
    pull_all_international()
