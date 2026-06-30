#!/bin/bash
#SBATCH --partition=gpu-redhat          # GPU partition on Amarel
#SBATCH --requeue                       # Requeue if preempted
#SBATCH --job-name=wc2026_live          # Job name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000                     # 32 GB RAM
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm.%N.%j.out
#SBATCH --error=slurm.%N.%j.err

# -----------------------------------------------------------------------
# WC2026 live forward simulation
# Simulates the remainder of WC2026 from current group-stage state.
# Completed match results are fixed; remaining games are simulated.
# -----------------------------------------------------------------------

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

WORKDIR=/scratch/$USER/world_cup_2026
cd $WORKDIR

source ~/miniconda3/bin/activate wc2026

python -c "import cupy; print('CuPy version:', cupy.__version__); cupy.cuda.Device(0).use(); print('GPU OK')"

echo ""
echo "Starting live simulation..."
python run_pipeline.py --features-only --live --gpu --n 500000

echo ""
echo "Job finished: $(date)"
