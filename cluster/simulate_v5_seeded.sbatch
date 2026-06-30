#!/bin/bash
#SBATCH --partition=gpu-redhat
#SBATCH --job-name=wc2026_v5s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=slurm.%N.%j.out
#SBATCH --error=slurm.%N.%j.err

echo "Job started: $(date)"
WORKDIR=/scratch/$USER/world_cup_2026
cd $WORKDIR
source ~/miniconda3/bin/activate wc2026

echo "=== Running 500k LIVE simulation with seeded ELO + squad_attack_quality ==="
python run_pipeline.py --features-only --live --gpu --n 500000

echo ""
echo "=== Top 20 results ==="
python - <<'PYEOF'
import pathlib, pandas as pd

out_dir = pathlib.Path('/scratch/ss3414/world_cup_2026/outputs')
csv_files = sorted(out_dir.glob('live_simulation_*.csv'), key=lambda f: f.stat().st_mtime)
if csv_files:
    latest = csv_files[-1]
    print(f"File: {latest.name}")
    df = pd.read_csv(latest)
    print(df.head(25).to_string(index=False))
PYEOF

echo "Job finished: $(date)"
