#!/bin/bash
#SBATCH --job-name=tfm_llm
#SBATCH --output=%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=computo
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

cd $HOME/tfm_smishing-detection

echo "=================================================="
echo "Starting LLM pipeline at: $PWD"
echo "=================================================="

# Create necessary output directories
mkdir -p output/llm

# Enable bash aliases so module commands work
shopt -s expand_aliases

# Environment variables
export PYTHONUNBUFFERED=1
export HF_HOME=$HOME/.cache/huggingface
# Load environment variables from .env file
if [ -f .env ]; then
  export $(grep -E '^HF_TOKEN' .env | xargs)
fi
mkdir -p $HF_HOME

# Load the official HPC containers module with a dedicated LLM overlay
export MY_ENV="tfm_smishing"
module load containers/cuda-12.4-uv

echo "=================================================="
echo "Initializing Environment..."
echo "=================================================="

# Initialize virtual environment in the apptainer overlay
uv-venv

# Install requirements
uv pip install --python /opt/venv transformers accelerate torch sentencepiece huggingface_hub jinja2

# Authenticate HF
python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"

echo ">> [PHASE 3] LLM Evaluation (Edge vs Server)"
python -m src.experiments.llm.main \
    --models phi4_mini gemma3n_4b qwen3_5_9b ministral_8b qwen3_6_27b gemma4_31b mistral_small_24b deepseek_v4_flash kimi_moonlight

echo "=================================================="
echo "Process finished. Job completed successfully."
echo "=================================================="
