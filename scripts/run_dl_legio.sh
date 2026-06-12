#!/bin/bash
#SBATCH --job-name=tfm_dl
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=computo
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
cd $HOME/tfm_smishing-detection

echo "=================================================="
echo "Iniciando pipeline de Deep Learning en: $PWD"
echo "=================================================="

# Creamos las carpetas necesarias
mkdir -p output/dl/checkpoints

# Habilitamos los alias de bash para que funcionen los comandos del módulo (uv, python)
shopt -s expand_aliases

# Siguiendo la tesis: Cargamos el módulo de contenedores HPC oficial
export MY_ENV="tfm_smishing"
module load containers/cuda-12.4-uv

export PYTHONUNBUFFERED=1

echo "=================================================="
echo "Ejecutando Python desde la imagen HPC"
echo "=================================================="

# Instalamos los requirements estrictamente necesarios para DL ignorando gensim/fasttext que fallan en compilación
python -m pip install scikit-learn transformers datasets accelerate torch pandas

# Ejecutamos el entrenamiento y evaluación de Modelos de Lenguaje Pequeños (SLMs)
python -m src.experiments.dl.main --data_root data/processed --out_dir output/dl --batch_size 16 --epochs 3

echo "=================================================="
echo "Proceso finalizado. Trabajo terminado exitosamente."
echo "=================================================="
