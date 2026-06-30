#!/bin/bash
#SBATCH --partition=gpu-redhat
#SBATCH --job-name=wc2026_pull_leagues
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm.%N.%j.out
#SBATCH --error=slurm.%N.%j.err

echo "Job started: $(date)"
WORKDIR=/scratch/$USER/world_cup_2026
cd $WORKDIR
source ~/miniconda3/bin/activate wc2026

python - <<'PYEOF'
"""
Pull FBref data for all non-Big-5 leagues (MLS, Brasileirao, Liga MX, etc.)
using the 2 most recent seasons only (fast; historic seasons already cached or not critical).
Then rebuild all_club_seasons.parquet to include all leagues.
"""
import sys, pathlib, pandas as pd
sys.path.insert(0, '/scratch/ss3414/world_cup_2026')

from src.ingest.club_seasons import (
    pull_league_season, LEAGUE_MAP, SEASON_MAP, RAW_DIR
)

# Only pull these for the most recent 2 seasons — historic Big 5 already cached
TARGET_LEAGUES = [
    "mls", "brasileirao", "liga_mx", "arg_primera",
    "super_lig", "eredivisie", "primeira_liga",
    "j1_league", "k_league", "saudi_pro",
    "belgian_pro", "scottish_prem", "colombian_liga",
]
RECENT_SEASONS = ["2023-2024", "2024-2025"]

print("=== Pulling non-Big-5 league data ===")
frames = []
for league_key in TARGET_LEAGUES:
    sd_league = LEAGUE_MAP.get(league_key)
    if not sd_league:
        print(f"  [SKIP] No soccerdata name for {league_key}")
        continue
    for season_str in RECENT_SEASONS:
        sd_season = SEASON_MAP.get(season_str)
        if sd_season is None:
            continue
        print(f"\nPulling {league_key} {season_str}...")
        try:
            df = pull_league_season(league_key, sd_league, season_str, sd_season)
            if not df.empty:
                frames.append(df)
                print(f"  → {len(df)} rows")
        except Exception as e:
            print(f"  [ERROR] {league_key} {season_str}: {e}")

print(f"\nNew data: {sum(len(f) for f in frames)} rows across {len(frames)} files")

# Rebuild all_club_seasons.parquet from all cached parquet files
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
    print("Leagues included:", combined["league_name"].value_counts().to_dict())
else:
    print("[ERROR] No data to combine")

PYEOF

echo "Pull finished: $(date)"
echo ""
echo "Running features + live simulation..."
python run_pipeline.py --features-only --live --gpu --n 500000

echo "Job finished: $(date)"
