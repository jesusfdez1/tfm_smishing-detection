#!/bin/bash
#SBATCH --job-name=tfm_dl
#SBATCH --output=%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=computo
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
cd $HOME/tfm_smishing-detection

echo "=================================================="
echo "Starting Deep Learning pipeline at: $PWD"
echo "=================================================="

# Create necessary output directories
mkdir -p output/dl/checkpoints

# Enable bash aliases so module commands work (uv, python)
shopt -s expand_aliases

# Environment variables
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="/root/.local/bin:$PATH"
export HF_HOME=$HOME/.cache/huggingface
# Load environment variables from .env file
if [ -f .env ]; then
  export $(grep -E '^HF_TOKEN' .env | xargs)
fi

mkdir -p $HF_HOME

# Clean corrupt RoBERTa-es cache just in case
rm -rf ~/.cache/huggingface/hub/models--PlanTL-GOB-ES--roberta-base-bne

# Load the official HPC containers module
export MY_ENV="tfm_smishing"
module load containers/cuda-12.4-uv

export PYTHONUNBUFFERED=1

echo "=================================================="
echo "Running Python from HPC image"
echo "=================================================="

# Install strictly necessary DL requirements using the cluster's superfast alias
# Added sentencepiece and tiktoken needed for roberta-es tokenizer
uv pip install --python /opt/venv scikit-learn transformers datasets accelerate torch pandas sentencepiece tiktoken huggingface_hub

# Authenticate HF robustly using python directly (avoids CLI binary path issues)
python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"

echo ">> [PHASE 1] Base Training and Evaluation (Normal Mode)"
python -m src.experiments.dl.main --data_root data/processed --out_dir output/dl --batch_size 16 --epochs 3

echo "=================================================="
echo "Starting Stress Evaluations (Surgeon Mode)"
echo "=================================================="

echo ">> [PHASE 2] Running Multilingual Benchmark (Exp 10)..."
python -m src.experiments.dl.eval_multilingual \
    --data_root data/processed \
    --out_dir output/dl/multilingual \
    --checkpoints_dir output/dl/checkpoints

echo ">> [PHASE 3] Running Obfuscation Benchmark (Exp 7)..."
python -m src.experiments.dl.eval_obfuscation \
    --data_root data/processed \
    --out_dir output/dl/obfuscation \
    --checkpoints_dir output/dl/checkpoints

echo "=================================================="
echo "Process finished. Job completed successfully."
echo "=================================================="
