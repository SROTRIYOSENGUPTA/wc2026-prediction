"""
Pull FBref stats for all WC2026-relevant non-Big-5 leagues locally (macOS + Chrome).
Run with: python3 pull_global_leagues_local.py

Saves to data/raw/club_seasons/ alongside existing Big-5 files.
Then rebuilds all_club_seasons.parquet.
"""

import sys
import pathlib
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from src.ingest.club_seasons import (
    pull_league_season, LEAGUE_MAP, SEASON_MAP, RAW_DIR
)

# Pull these leagues for the 2 most recent full seasons
# 2025-26 FBref lost Opta licence — standard stats only, skip xG
TARGET_LEAGUES = [
    # Missing from local cache (Big-5 gaps)
    "serie_a",
    # Europe second-tier
    "eredivisie",
    "primeira_liga",
    "super_lig",
    "belgian_pro",
    "scottish_prem",
    # Americas
    "mls",           # Messi (Inter Miami)
    "brasileirao",   # Vinicius, Endrick, Rodrygo club peers
    "liga_mx",       # Mexico squad
    "arg_primera",   # Argentina squad domestic players
    "colombian_liga",
    # Asia / Other
    "j1_league",     # Japan squad
    "k_league",      # South Korea squad
    "saudi_pro",     # Ronaldo, Neymar etc.
]
RECENT_SEASONS = ["2023-2024", "2024-2025"]

print("=" * 60)
print("  Pulling global league data for WC2026")
print(f"  Leagues: {len(TARGET_LEAGUES)}, Seasons: {RECENT_SEASONS}")
print("=" * 60)

frames = []
errors = []

for league_key in TARGET_LEAGUES:
    sd_league = LEAGUE_MAP.get(league_key)
    if not sd_league:
        print(f"  [SKIP] No mapping for {league_key}")
        continue
    for season_str in RECENT_SEASONS:
        sd_season = SEASON_MAP.get(season_str)
        if sd_season is None:
            continue
        try:
            df = pull_league_season(league_key, sd_league, season_str, sd_season)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            msg = f"{league_key} {season_str}: {e}"
            print(f"  [ERROR] {msg}")
            errors.append(msg)

print(f"\nNew data: {sum(len(f) for f in frames)} rows across {len(frames)} files")
if errors:
    print(f"Errors ({len(errors)}):")
    for e in errors:
        print(f"  {e}")

# Rebuild all_club_seasons.parquet from ALL cached files
print("\nRebuilding all_club_seasons.parquet...")
all_files = sorted(RAW_DIR.glob("*_20[0-9][0-9]-20[0-9][0-9].parquet"))
all_frames = []
for f in all_files:
    try:
        df = pd.read_parquet(f)
        all_frames.append(df)
    except Exception as e:
        print(f"  [WARN] Could not read {f.name}: {e}")

if all_frames:
    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_parquet(RAW_DIR / "all_club_seasons.parquet", index=False)
    print(f"Saved {len(combined)} total rows → all_club_seasons.parquet")
    vc = combined["league_name"].value_counts()
    print("Leagues included:")
    for league, count in vc.items():
        print(f"  {league}: {count}")
else:
    print("[ERROR] No data to combine")
