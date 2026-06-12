#!/bin/bash
#SBATCH --job-name=dl_evals
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%j.out

cd $HOME/tfm_smishing-detection

echo "=================================================="
echo "Starting Deep Learning Evaluations (Surgeon Mode)"
echo "=================================================="

# Enable bash aliases so module commands work
shopt -s expand_aliases

# Base environment variables
export PYTHONUNBUFFERED=1
export PATH="/root/.local/bin:$PATH"

# Load the official container module
export MY_ENV="tfm_smishing"
module load containers/cuda-12.4-uv

echo "=================================================="
echo "Running Python from HPC image"
echo "=================================================="

# Ensure we have the minimum dependencies
python -m pip install scikit-learn transformers datasets accelerate torch pandas sentencepiece tiktoken huggingface_hub

echo ">> [1/2] Running Multilingual Benchmark (Exp 10)..."
python -m src.experiments.dl.eval_multilingual \
    --data_root data/processed \
    --out_dir output/dl/multilingual \
    --checkpoints_dir output/dl/checkpoints
    
echo ">> [2/2] Running Obfuscation Benchmark (Exp 7)..."
python -m src.experiments.dl.eval_obfuscation \
    --data_root data/processed \
    --out_dir output/dl/obfuscation \
    --checkpoints_dir output/dl/checkpoints

echo "=================================================="
echo "Evaluations Completed."
echo "Results in: output/dl/multilingual and output/dl/obfuscation"
echo "=================================================="
