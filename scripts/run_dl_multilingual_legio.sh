#!/bin/bash
#SBATCH -J eval_multi
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o eval_multi_%j.out

echo "=================================================="
echo "Iniciando Evaluacion Multilingue (Cross-lingual)"
echo "=================================================="

# Variables
export MY_ENV="tfm_smishing"
export UV_PYTHON="3.11"
export UV_PROJECT_ENVIRONMENT="/home/$USER/envs/$MY_ENV"

# Load modules
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0

# Ensure uv is installed
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Run inside Singularity
singularity exec --nv \
    --env UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT" \
    --env UV_PYTHON="$UV_PYTHON" \
    --bind /home/$USER/envs:/home/$USER/envs \
    --bind $PWD:$PWD \
    --pwd $PWD \
    /home/$USER/envs/tfm_smishing.img bash -c '
        export PATH="$HOME/.cargo/bin:$PATH"
        uv run python -m src.experiments.dl.eval_multilingual \
            --data_root data/processed \
            --out_dir output/dl/multilingual \
            --checkpoints_dir output/dl/checkpoints
    '

echo "=================================================="
echo "Evaluacion Completada. Resultados en output/dl/multilingual/results_multilingual.csv"
echo "=================================================="
