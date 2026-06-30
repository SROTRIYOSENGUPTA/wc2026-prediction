#!/bin/bash
#SBATCH --partition=gpu-redhat
#SBATCH --job-name=wc2026_v5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm.%N.%j.out
#SBATCH --error=slurm.%N.%j.err

echo "Job started: $(date)"
WORKDIR=/scratch/$USER/world_cup_2026
cd $WORKDIR
source ~/miniconda3/bin/activate wc2026

echo ""
echo "=== Step 1: Rebuild features (with squad_attack_quality + RT fallback) ==="
python - <<'PYEOF'
import sys
sys.path.insert(0, '/scratch/ss3414/world_cup_2026')
from src.features import build_all_features
pf, tf = build_all_features()
print(f"player_features: {len(pf)} rows, {len(pf.columns)} cols")
print(f"team_features:   {len(tf)} rows, {len(tf.columns)} cols")
if 'squad_attack_quality' in tf.columns:
    print("squad_attack_quality: OK")
    print(tf.sort_values('squad_attack_quality', ascending=False)[
        ['team','squad_attack_quality','squad_avg_club_xg_p90']
    ].head(10).to_string())
else:
    print("[WARN] squad_attack_quality missing from team features!")
PYEOF

echo ""
echo "=== Step 2: Retrain XGBoost model with new features ==="
python src/model.py --gpu

echo ""
echo "=== Step 3: Run 500k live simulation from current WC2026 state ==="
python run_pipeline.py --features-only --live --gpu --n 500000

echo ""
echo "=== Step 4: Show top 20 results ==="
python - <<'PYEOF'
import pathlib, pandas as pd

out_dir = pathlib.Path('/scratch/ss3414/world_cup_2026/outputs')
csv_files = sorted(out_dir.glob('live_simulation_*.csv'), key=lambda f: f.stat().st_mtime)
if csv_files:
    latest = csv_files[-1]
    print(f"Latest results: {latest.name}")
    df = pd.read_csv(latest)
    print(df.head(25).to_string(index=False))
PYEOF

echo ""
echo "Job finished: $(date)"
