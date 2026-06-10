#!/bin/bash
#SBATCH --job-name=metasms_enrich
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=computo

# 1. Cleanup of previous zombie processes
echo "Cleaning up potential zombie Python processes..."
pkill -9 -f "python" || true
pkill -9 -f "apptainer" || true
sleep 3

cd $HOME/tfm_smishing-detection

echo "=================================================="
echo "Starting sequential LLM enrichment process"
echo "=================================================="

# Environment configuration
shopt -s expand_aliases
export MY_ENV="tfm_smishing"
module load containers/cuda-12.4-uv

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="/root/.local/bin:$PATH"

# Ninja workaround for Apptainer path resolution
echo '#!/bin/bash' > ninja
echo 'python -m ninja "$@"' >> ninja
chmod +x ninja
export PATH="$PWD:$PATH"

export HF_HOME=$HOME/.cache/huggingface
mkdir -p $HF_HOME

echo "Ensuring required dependencies are installed..."
python -m pip install -r requirements.txt --quiet
python -m pip install accelerate "transformers>=4.48.0" torchvision pillow ninja --quiet

# vLLM environment variables
export VLLM_USE_FLASHINFER=0
export VLLM_USE_V1=0

INPUT_CSV="data/processed/metasms_v1_privacy.csv"
OUTPUT_CSV="data/processed/metasms_v1_enriched.csv"

echo "=================================================="
echo "Launching main enrichment script (1 GPU)"
echo "=================================================="

# Execute the enrichment script sequentially on a single GPU
CUDA_VISIBLE_DEVICES=0 python src/dataset/enrich_metadataset.py \
    --input $INPUT_CSV \
    --output $OUTPUT_CSV \
    --batch-size 256 \
    --resume

echo "=================================================="
echo "Process completed successfully."
echo "=================================================="
