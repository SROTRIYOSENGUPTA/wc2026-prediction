#!/bin/bash
#SBATCH --partition=gpu               # GPU partition
#SBATCH --requeue
#SBATCH --job-name=wc2026_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8             # More CPUs for 5-fold CV
#SBATCH --mem=64000                   # 64 GB RAM for XGBoost
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00               # 2 hours for training + calibration
#SBATCH --output=slurm.%N.%j.out
#SBATCH --error=slurm.%N.%j.err

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"

WORKDIR=/scratch/$USER/world_cup_2026
cd $WORKDIR

source ~/miniconda3/bin/activate wc2026

# Verify CUDA is available for XGBoost
python -c "import xgboost as xgb; print('XGBoost', xgb.__version__)"

echo "Training model..."
python run_pipeline.py --features-only --train --gpu

echo "Job finished: $(date)"
