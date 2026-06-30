#!/bin/bash
#SBATCH --partition=gpu               # GPU partition
#SBATCH --requeue                     # Requeue if preempted
#SBATCH --job-name=wc2026_sim         # Job name
#SBATCH --nodes=1                     # 1 node
#SBATCH --ntasks=1                    # 1 task
#SBATCH --cpus-per-task=4             # 4 CPU cores for data loading
#SBATCH --mem=32000                   # 32 GB RAM
#SBATCH --gres=gpu:1                  # 1 GPU
#SBATCH --time=01:00:00               # 1 hour time limit
#SBATCH --output=slurm.%N.%j.out      # STDOUT log
#SBATCH --error=slurm.%N.%j.err       # STDERR log

# -----------------------------------------------------------------------
# WC2026 Monte Carlo simulation — GPU job
# Runs 500k tournament simulations using CuPy on an Amarel GPU node.
# Expected runtime: ~5 min with CuPy vs ~45 min CPU-only.
# -----------------------------------------------------------------------

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Work from scratch (best I/O performance)
WORKDIR=/scratch/$USER/world_cup_2026
cd $WORKDIR

# Load modules
source ~/miniconda3/bin/activate wc2026

# Verify GPU is visible
python -c "import cupy; print('CuPy version:', cupy.__version__); cupy.cuda.Device(0).use(); print('GPU OK')"

# Run 500k simulations on GPU
echo ""
echo "Starting simulation..."
python run_pipeline.py --simulate --gpu --n 500000

echo ""
echo "Job finished: $(date)"
