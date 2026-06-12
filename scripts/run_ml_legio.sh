#!/bin/bash
#SBATCH --job-name=ml_pipeline
#SBATCH --output=%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=computo
#SBATCH --exclusive
cd $HOME/tfm_smishing-detection

echo "=================================================="
echo "Starting ml_pipeline at: $PWD"
echo "=================================================="

# Create necessary output directories
mkdir -p output/ml

# Enable bash aliases so module commands work (uv, python)
shopt -s expand_aliases

# Load the official container module
export MY_ENV="tfm_smishing_ml"
module load containers/cuda-12.4-uv

export PYTHONUNBUFFERED=1

echo "=================================================="
echo "Running Python 3.11 for gensim compatibility"
echo "=================================================="

# Use the 'uv' tool from the module to download Python 3.11
uv python install 3.11

# Create an isolated virtual environment with the stable version
uv venv .venv_ml --python 3.11

# Activate it
source .venv_ml/bin/activate

# Install requirements extremely fast with uv
uv pip install -r requirements.txt

# Run the Machine Learning grid search for ALL feature variants

echo ">> [1/4] Training NORM variant (Baseline)"
python -m src.experiments.ml.main --data_root data/processed --out_dir output/ml/norm --feature_type norm

echo ">> [2/4] Training RAW variant (Privacy vs Overfitting)"
python -m src.experiments.ml.main --data_root data/processed --out_dir output/ml/raw --feature_type raw

echo ">> [3/4] Training ANONYMIZED variant"
python -m src.experiments.ml.main --data_root data/processed --out_dir output/ml/anonymized --feature_type anonymized

echo ">> [4/4] Training INJECTED variant (Urgency + Theme)"
python -m src.experiments.ml.main --data_root data/processed --out_dir output/ml/injected --feature_type injected

echo "=================================================="
echo "Process finished. Job completed successfully."
echo "=================================================="
