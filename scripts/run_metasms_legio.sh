#!/bin/bash
#SBATCH --job-name=metasms_build
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=computo
#SBATCH --gres=gpu:1
#SBATCH --exclusive

# 1. Define directory variables
HOME_DIR="$HOME/tfm_smishing-detection"
SCRATCH_DIR="/local_scratch/$SLURM_JOB_ID/tfm_smishing-detection"

echo "=================================================="
echo "Preparing environment in local SSD (/local_scratch)"
echo "=================================================="

# 2. Configure automatic cleanup (even if the job fails or is cancelled)
trap 'echo "Cleaning $SCRATCH_DIR..."; rm -rf /local_scratch/$SLURM_JOB_ID' EXIT

# 3. Create temporary directory in local SSD
mkdir -p "$SCRATCH_DIR"

# 4. Copy code and input data to local SSD
# Exclude previous processed data to avoid saturating the copy
rsync -avq --exclude='data/processed' --exclude='logs' "$HOME_DIR/" "$SCRATCH_DIR/"

# 5. Move to local SSD so all I/O happens there
cd "$SCRATCH_DIR"

# Create necessary folders in SSD
mkdir -p logs data/processed

echo "=================================================="
echo "Starting build_metasms at: $PWD"
echo "=================================================="

# Enable bash aliases so module commands work (uv, python)
shopt -s expand_aliases

# Following the manual: Load the official module using your image
export MY_ENV="tfm_smishing"
module load containers/cuda-12.4-uv

export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=debug
export HF_HUB_VERBOSITY=debug

echo "=================================================="
echo "Executing Python from tfm_smishing.img image"
echo "=================================================="

# Install requirements in case any is missing in the image using configured python
python -m pip install -r requirements.txt

# Execute the script as indicated in the manual
python src/dataset/build_metadataset.py --privacy-filter --checkpoint-every 5000 --chunk-size 5000

echo "=================================================="
echo "Process finished. Synchronizing results to HOME..."
echo "=================================================="

# 6. Sync back processed data and logs to network storage (CEPH)
rsync -avq "$SCRATCH_DIR/data/processed/" "$HOME_DIR/data/processed/"
rsync -avq "$SCRATCH_DIR/logs/" "$HOME_DIR/logs/"

echo "=================================================="
echo "Synchronization complete. Job finished successfully."
echo "=================================================="
