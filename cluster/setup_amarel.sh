#!/bin/bash
# -----------------------------------------------------------------------
# One-time setup script — run this on Amarel AFTER transferring the project.
#
# Run from the Amarel login node:
#   bash ~/world_cup_2026/cluster/setup_amarel.sh
# -----------------------------------------------------------------------

set -e

echo "=== WC2026 Amarel Setup ==="

# 1. Create scratch working directory
WORKDIR=/scratch/$USER/world_cup_2026
echo "[1] Creating scratch directory: $WORKDIR"
mkdir -p $WORKDIR

# 2. Copy project from home to scratch (scratch has better I/O for jobs)
echo "[2] Copying project to scratch..."
cp -r ~/world_cup_2026/. $WORKDIR/

echo "    Project copied to $WORKDIR"

# 3. Load Python module
echo "[3] Loading Python module..."
module purge
module load python/3.11.4

# Check if this version exists, otherwise list available
if ! module load python/3.11.4 2>/dev/null; then
    echo "    python/3.11.4 not found. Available Python versions:"
    module avail python 2>&1 | grep python
    echo "    Edit this script to use the correct version."
    exit 1
fi

# 4. Create virtual environment in scratch
echo "[4] Creating virtual environment..."
cd $WORKDIR
python -m venv venv
source venv/bin/activate

# 5. Install packages (no apt/yum — pip to user venv only)
echo "[5] Installing Python packages..."
pip install --upgrade pip

# Core packages
pip install \
    statsbombpy>=1.0.3 \
    xgboost>=2.0.0 \
    lightgbm>=4.0.0 \
    scikit-learn>=1.3.0 \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    pyarrow>=12.0.0 \
    requests>=2.31.0 \
    beautifulsoup4>=4.12.0 \
    lxml>=4.9.0 \
    aiohttp>=3.8.0 \
    pyyaml>=6.0.0 \
    tqdm>=4.65.0

# 6. Install CuPy for GPU acceleration
# cupy-cuda12x matches CUDA 12.x (most Amarel GPU nodes)
# If you get a CUDA version error, check: nvidia-smi
echo "[6] Installing CuPy for GPU..."
pip install cupy-cuda12x

# Verify CuPy installed
python -c "import cupy; print('CuPy installed:', cupy.__version__)"

# 7. Verify XGBoost GPU support
python -c "
import xgboost as xgb
print('XGBoost:', xgb.__version__)
# Quick GPU test
import numpy as np
X = np.random.rand(100, 5)
y = np.random.randint(0, 3, 100)
clf = xgb.XGBClassifier(device='cuda', n_estimators=5, num_class=3, objective='multi:softprob')
clf.fit(X, y)
print('XGBoost GPU test: OK')
"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Working directory: $WORKDIR"
echo ""
echo "Next steps:"
echo "  1. Copy your cached data from home to scratch:"
echo "     cp -r ~/world_cup_2026/data $WORKDIR/"
echo ""
echo "  2. Submit the training job:"
echo "     sbatch $WORKDIR/cluster/gpu_train.sbatch"
echo ""
echo "  3. Once trained, submit the simulation job:"
echo "     sbatch $WORKDIR/cluster/gpu_simulate.sbatch"
echo ""
echo "  4. Monitor jobs:"
echo "     squeue -u \$USER"
echo "     tail -f slurm.*.out"
