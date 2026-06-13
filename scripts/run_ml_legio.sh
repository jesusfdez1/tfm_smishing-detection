#!/bin/bash
#SBATCH --job-name=tfm_ml
#SBATCH --output=%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=computo
#SBATCH --exclusive
cd $HOME/tfm_smishing-detection

echo "=================================================="
echo "Starting Machine Learning pipeline at: $PWD"
echo "=================================================="

# Create necessary output directories
mkdir -p output/ml

# Enable bash aliases so module commands work (uv, python)
shopt -s expand_aliases

# Load the official HPC containers module
export MY_ENV="tfm_smishing"
module load containers/cuda-12.4-uv

# Environment variables
export PYTHONUNBUFFERED=1

echo "=================================================="
echo "Forcing Python 3.11 to avoid gensim C-API compilation errors"
echo "=================================================="

# The SLURM bash script runs on the host node, NOT inside the container!
# Commands like 'uv' are aliases that execute inside the container.
# Therefore, absolute paths like /opt/venv_ml/bin/python only exist inside the container.
# We must use 'uv run' to execute our python script INSIDE the container using the venv we just created!
echo "=================================================="
echo "Executing ML pipeline inside container via uv run"
echo "=================================================="

export UV_PYTHON_INSTALL_DIR=/opt/uv_pythons
mkdir -p /opt/uv_pythons

# Download Python 3.11 explicitly into the overlay
uv python install 3.11

# Create the virtual environment in the overlay (WITHOUT --clear so it reuses the 80-min installation)
echo ">> Creating isolated environment in /opt/venv_ml"
uv venv /opt/venv_ml --python 3.11

# Install requirements explicitly into the overlay venv using uv (will skip if already installed)
uv pip install --python /opt/venv_ml scikit-learn pandas numpy xgboost lightgbm catboost gensim fasttext-wheel tqdm imbalanced-learn

echo ">> [PHASE 1/4] Training NORM variant (Baseline)"
uv run --python /opt/venv_ml python -m src.experiments.ml.main --data_root data/processed --out_dir output/ml/norm --feature_type norm

echo ">> [PHASE 2/4] Training RAW variant (Privacy vs Overfitting)"
uv run --python /opt/venv_ml python -m src.experiments.ml.main --data_root data/processed --out_dir output/ml/raw --feature_type raw

echo ">> [PHASE 3/4] Training ANONYMIZED variant"
uv run --python /opt/venv_ml python -m src.experiments.ml.main --data_root data/processed --out_dir output/ml/anonymized --feature_type anonymized

echo ">> [PHASE 4/4] Training INJECTED variant (Urgency + Theme)"
uv run --python /opt/venv_ml python -m src.experiments.ml.main --data_root data/processed --out_dir output/ml/injected --feature_type injected

echo "=================================================="
echo "Process finished. Job completed successfully."
echo "=================================================="
