"""
Pull WC2026 squad data: all 48 squads, 1,363 players with per-90 stats.

Primary source: github.com/risingtransfers/world-cup-2026-data (CC BY 4.0)
  - squads.csv: player_id, player_name, country, position, club, age, rt_value_estimate_eur
  - per90_stats.csv: player_id, player_name, season, minutes, goals_per90, assists_per90, etc.
  - Join key: player_id (shared across both files)
"""

import pandas as pd
import requests
from pathlib import Path
import yaml

CFG = yaml.safe_load(open(Path(__file__).parents[2] / "configs/config.yaml"))
RAW_DIR = Path(__file__).parents[2] / CFG["paths"]["raw_squads"]

SQUAD_URL = (
    "https://raw.githubusercontent.com/risingtransfers/"
    "world-cup-2026-data/main/data/squads.csv"
)
PER90_URL = (
    "https://raw.githubusercontent.com/risingtransfers/"
    "world-cup-2026-data/main/data/per90_stats.csv"
)


def pull_wc2026_squads() -> pd.DataFrame:
    """Download WC2026 squad list with player metadata."""
    out_path = RAW_DIR / "wc2026_squads.parquet"
    if out_path.exists():
        print("[SKIP] WC2026 squads already cached")
        return pd.read_parquet(out_path)

    print("Fetching WC2026 squad list...")
    try:
        resp = requests.get(SQUAD_URL, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[ERROR] Could not fetch squad data: {e}")
        return pd.DataFrame()

    df = pd.read_csv(pd.io.common.StringIO(resp.text))
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    # Normalise to consistent internal column names
    df = df.rename(columns={"player_name": "player", "country": "team"})

    print(f"  {len(df)} players across {df['team'].nunique()} teams")
    df.to_parquet(out_path, index=False)
    return df


def pull_wc2026_player_stats() -> pd.DataFrame:
    """
    Download per-90 stats for WC2026 squad players.
    Only covers players with sufficient league minutes in 2025/26.
    """
    out_path = RAW_DIR / "wc2026_player_stats.parquet"
    if out_path.exists():
        print("[SKIP] WC2026 player stats already cached")
        return pd.read_parquet(out_path)

    print("Fetching WC2026 player per-90 stats...")
    try:
        resp = requests.get(PER90_URL, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[ERROR] Could not fetch player stats: {e}")
        return pd.DataFrame()

    df = pd.read_csv(pd.io.common.StringIO(resp.text))
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"player_name": "player"})

    min_threshold = CFG["min_club_minutes_per_season"]
    if "minutes" in df.columns:
        df["sufficient_minutes"] = df["minutes"] >= min_threshold
        below = (~df["sufficient_minutes"]).sum()
        if below > 0:
            print(f"  [NOTE] {below} players below {min_threshold} min threshold")

    df.to_parquet(out_path, index=False)
    print(f"  {len(df)} player records saved")
    return df


def build_squad_master() -> pd.DataFrame:
    """
    Merge squad list + per-90 stats into a single master table.
    Join key: player_id (reliable; avoids name encoding issues).
    """
    out_path = RAW_DIR / "squad_master.parquet"
    if out_path.exists():
        print("[SKIP] squad_master already built")
        return pd.read_parquet(out_path)

    squads = pull_wc2026_squads()
    stats = pull_wc2026_player_stats()

    if squads.empty:
        return pd.DataFrame()

    if stats.empty:
        print("[WARN] No per-90 stats — returning squad list only")
        squads.to_parquet(out_path, index=False)
        return squads

    # player_id is the reliable join key across both files
    join_key = "player_id" if "player_id" in squads.columns and "player_id" in stats.columns \
               else "player"

    # Keep only the most recent season per player from per90 stats
    if "season" in stats.columns:
        stats = stats.sort_values("season").groupby(join_key).last().reset_index()

    master = squads.merge(stats, on=join_key, how="left",
                          suffixes=("", "_stats"))

    # Drop duplicate player name column if it came through twice
    dup_cols = [c for c in master.columns if c.endswith("_stats")]
    master = master.drop(columns=dup_cols, errors="ignore")

    master.to_parquet(out_path, index=False)
    print(f"\nSquad master built: {len(master)} players, {master['team'].nunique()} teams")
    return master


if __name__ == "__main__":
    build_squad_master()
