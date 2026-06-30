"""
Pull player-level club season stats from FBref (2021/22 - 2025/26).

Uses the soccerdata library which runs a headless Chrome browser to bypass
FBref's bot detection (plain requests get 403'd since 2025).

FBref lost its Opta licence in January 2026, so 2025/26 advanced stats
(xG, progressive passes) are unreliable. Standard stats remain available.
Cached in ~/soccerdata/data/FBref/ — subsequent runs are instant.
"""

import warnings
import pandas as pd
import numpy as np
from pathlib import Path
import yaml

warnings.filterwarnings("ignore")

CFG = yaml.safe_load(open(Path(__file__).parents[2] / "configs/config.yaml"))
RAW_DIR = Path(__file__).parents[2] / CFG["paths"]["raw_club_seasons"]
RAW_DIR.mkdir(parents=True, exist_ok=True)

# soccerdata league name format: "ENG-Premier League", "ESP-La Liga", etc.
LEAGUE_MAP = {
    # Big 5
    "premier_league": "ENG-Premier League",
    "la_liga":        "ESP-La Liga",
    "bundesliga":     "GER-Bundesliga",
    "serie_a":        "ITA-Serie A",
    "ligue_1":        "FRA-Ligue 1",
    # Second tier Europe
    "eredivisie":     "NED-Eredivisie",
    "primeira_liga":  "POR-Primeira Liga",
    "super_lig":      "TUR-Süper Lig",
    "belgian_pro":    "BEL-First Division A",
    "scottish_prem":  "SCO-Scottish Premiership",
    # Americas
    "brasileirao":    "BRA-Brasileirao",
    "mls":            "USA-MLS",
    "liga_mx":        "MEX-Liga MX",
    "arg_primera":    "ARG-Primera División",
    "colombian_liga": "COL-Liga BetPlay Dimayor",
    # Asia / Others
    "j1_league":      "JPN-J League",
    "k_league":       "KOR-K League 1",
    "saudi_pro":      "SAU-Saudi Pro League",
}

# soccerdata uses end-year of season as the season identifier (2022 = 2021-22)
SEASON_MAP = {
    "2021-2022": 2022,
    "2022-2023": 2023,
    "2023-2024": 2024,
    "2024-2025": 2025,
    "2025-2026": 2026,
}

# Advanced stats unavailable after FBref licence loss (Jan 2026)
ADVANCED_COLS = {"xg", "xag", "npxg", "progressive_passes", "progressive_carries",
                 "progressive_receptions", "pressure_regains"}

FBREF_LICENCE_LOSS_SEASON = "2025-2026"

# Stat types supported by soccerdata FBref wrapper
# (passing/defense/possession are NOT available — FBref removed them from the API)
STAT_TYPES = ["standard", "shooting", "playing_time", "misc"]


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns from soccerdata into simple lowercase names."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(c).strip().lower().replace(" ", "_")
                     for c in col if str(c).strip() and str(c) != "")
            .strip("_")
            for col in df.columns
        ]
    else:
        df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    return df


def _normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common soccerdata column variants to our standard names."""
    renames = {
        # Standard stats
        "performance_gls": "goals",
        "performance_ast": "assists",
        "expected_xg":     "xg",
        "expected_xag":    "xag",
        "expected_npxg":   "npxg",
        "playing_time_min": "min",
        "playing_time_mp":  "mp",
        # Shooting
        "standard_sh":     "shots",
        "standard_sot":    "shots_on_target",
        # Passing
        "kp":              "key_passes",
        # Defense
        "tackles_tkl":     "tackles",
        "tackles_tklw":    "tackles_won",
        "pressures_press": "pressures",
        "pressures_succ":  "pressure_regains",
        # Possession
        "touches_att_pen": "touches_att_pen_area",
        "carries_prогr":   "progressive_carries",
        "receiving_prогр": "progressive_receptions",
    }
    return df.rename(columns={k: v for k, v in renames.items() if k in df.columns})


def pull_league_season(league_name: str, sd_league: str,
                        season_str: str, sd_season: int) -> pd.DataFrame:
    """
    Pull all stat types for one league+season using soccerdata FBref.
    Returns merged per-player DataFrame.
    """
    import soccerdata as sd

    out_path = RAW_DIR / f"{league_name}_{season_str.replace('/', '_')}.parquet"
    if out_path.exists():
        print(f"  [SKIP] {league_name} {season_str} already cached")
        return pd.read_parquet(out_path)

    print(f"Pulling {league_name} {season_str}...")

    try:
        fbref = sd.FBref(leagues=[sd_league], seasons=[sd_season], no_cache=False)
    except Exception as e:
        print(f"  [WARN] Could not init FBref for {league_name}: {e}")
        return pd.DataFrame()

    frames = {}
    for stat in STAT_TYPES:
        try:
            df = fbref.read_player_season_stats(stat_type=stat)
            if df is None or df.empty:
                continue
            # Reset index to get player/team as regular columns
            df = df.reset_index()
            df = _flatten_columns(df)
            df = _normalise_cols(df)
            frames[stat] = df
            print(f"  {stat}: {len(df)} rows")
        except Exception as e:
            print(f"  [WARN] {stat} failed: {e}")

    if not frames:
        print(f"  [WARN] No data for {league_name} {season_str}")
        return pd.DataFrame()

    # Merge all stat frames on player + team
    base = frames.get("standard", next(iter(frames.values()))).copy()

    for stat, df in frames.items():
        if stat == "standard":
            continue
        merge_on = [c for c in ["player", "team"] if c in df.columns and c in base.columns]
        if not merge_on:
            continue
        overlap = [c for c in df.columns if c in base.columns and c not in merge_on]
        base = base.merge(df.drop(columns=overlap, errors="ignore"),
                          on=merge_on, how="left", suffixes=("", f"_{stat}"))

    base["league_name"] = league_name
    base["season"] = season_str

    # Flag advanced cols as unreliable post-licence-loss
    for col in ADVANCED_COLS:
        if col in base.columns:
            if season_str == FBREF_LICENCE_LOSS_SEASON:
                base[f"{col}_reliable"] = False
            else:
                base[f"{col}_reliable"] = True

    base.to_parquet(out_path, index=False)
    print(f"  Saved {len(base)} player rows → {out_path.name}")
    return base


def pull_all_club_seasons(leagues: dict | None = None,
                           seasons: list | None = None) -> pd.DataFrame:
    """
    Pull club season data for all configured leagues and seasons.
    Returns combined DataFrame; saves all_club_seasons.parquet.
    """
    if leagues is None:
        leagues = {k: LEAGUE_MAP[k] for k in LEAGUE_MAP
                   if k in CFG["leagues"]["fbref_ids"]}
    if seasons is None:
        start = CFG["seasons"]["start_year"]
        n = len(CFG["seasons"]["club"])
        seasons = [f"{y}-{y+1}" for y in range(start, start + n)]

    all_frames = []
    total = len(leagues) * len(seasons)
    done = 0
    for league_name, sd_league in leagues.items():
        for season_str in seasons:
            sd_season = SEASON_MAP.get(season_str)
            if sd_season is None:
                print(f"  [SKIP] Unknown season format: {season_str}")
                continue
            done += 1
            print(f"\n[{done}/{total}] {league_name} {season_str}")
            df = pull_league_season(league_name, sd_league, season_str, sd_season)
            if not df.empty:
                all_frames.append(df)

    if not all_frames:
        print("[WARN] No club data retrieved.")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_parquet(RAW_DIR / "all_club_seasons.parquet", index=False)
    print(f"\nDone. {len(combined)} total player-season rows saved.")
    return combined


if __name__ == "__main__":
    pull_all_club_seasons()
