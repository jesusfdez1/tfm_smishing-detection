#!/bin/bash
#SBATCH --job-name=ml_pipeline
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=computo
#SBATCH --exclusive
cd $HOME/tfm_smishing-detection

echo "=================================================="
echo "Iniciando ml_pipeline en: $PWD"
echo "=================================================="

# Creamos las carpetas necesarias
mkdir -p output/ml

# Habilitamos los alias de bash para que funcionen los comandos del módulo (uv, python)
shopt -s expand_aliases

# Siguiendo el manual: Cargamos el módulo oficial
export MY_ENV="tfm_smishing"
module load containers/cuda-12.4-uv

export PYTHONUNBUFFERED=1

echo "=================================================="
echo "Ejecutando Python desde la imagen tfm_smishing.img"
echo "=================================================="

# Instalamos los requirements por si falta alguno
python -m pip install -r requirements.txt

# Ejecutamos el grid search de Machine Learning
python -m src.experiments.ml.main --data_root data/processed --out_dir output/ml

echo "=================================================="
echo "Proceso finalizado. Trabajo terminado exitosamente."
echo "=================================================="
