#!/bin/bash
#SBATCH -J dl_evals
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o dl_evals_%j.out

cd $HOME/tfm_smishing-detection

echo "=================================================="
echo "Iniciando Evaluaciones de Deep Learning (Modo Cirujano)"
echo "=================================================="

# Habilitamos los alias de bash para que funcionen los comandos del módulo
shopt -s expand_aliases

# Variables de entorno base
export PYTHONUNBUFFERED=1
export PATH="/root/.local/bin:$PATH"

# Siguiendo la arquitectura del script base: Cargamos el módulo
export MY_ENV="tfm_smishing"
module load containers/cuda-12.4-uv

echo "=================================================="
echo "Ejecutando Python desde la imagen HPC"
echo "=================================================="

# Aseguramos que tenemos las dependencias mínimas
python -m pip install scikit-learn transformers datasets accelerate torch pandas sentencepiece tiktoken huggingface_hub

echo ">> [1/2] Ejecutando Benchmark Multilingue (Exp 10)..."
python -m src.experiments.dl.eval_multilingual \
    --data_root data/processed \
    --out_dir output/dl/multilingual \
    --checkpoints_dir output/dl/checkpoints
    
echo ">> [2/2] Ejecutando Benchmark de Ofuscacion (Exp 7)..."
python -m src.experiments.dl.eval_obfuscation \
    --data_root data/processed \
    --out_dir output/dl/obfuscation \
    --checkpoints_dir output/dl/checkpoints

echo "=================================================="
echo "Evaluaciones Completadas."
echo "Resultados en: output/dl/multilingual y output/dl/obfuscation"
echo "=================================================="
