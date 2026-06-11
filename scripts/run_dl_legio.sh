#!/bin/bash
#SBATCH --job-name=tfm_dl
#SBATCH --output=logs/dl_evaluation_%j.log
#SBATCH --error=logs/dl_evaluation_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

module load Python/3.11
source venv/bin/activate

# Execute the deep learning evaluation
# Automatically evaluates all 7 predefined SLM models
python -m src.experiments.dl.main --data_root data/processed --out_dir output/dl --batch_size 16 --epochs 3
