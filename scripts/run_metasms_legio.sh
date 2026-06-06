#!/bin/bash
#SBATCH --job-name=metasms_build
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=computo
#SBATCH --gres=gpu:1

# Nos movemos a la carpeta del proyecto en tu HOME
cd $HOME/tfm_smishing-detection

# Creamos las carpetas necesarias
mkdir -p logs data/processed

echo "=================================================="
echo "Iniciando build_metasms directamente en HOME"
echo "=================================================="

# Habilitamos los alias de bash para que funcionen los comandos del módulo (uv, python)
shopt -s expand_aliases

# Siguiendo el manual: Cargamos el módulo oficial usando tu imagen
export MY_ENV="tfm_smishing"
module load containers/cuda-12.4-uv

export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=debug
export HF_HUB_VERBOSITY=debug

echo "=================================================="
echo "Ejecutando Python desde la imagen tfm_smishing.img"
echo "=================================================="

# Instalamos los requirements por si falta alguno en la imagen usando el python configurado
python -m pip install -r requirements.txt

# Ejecutamos el script tal cual indica el manual
python src/dataset/build_metadataset.py --privacy-filter --checkpoint-every 5000 --chunk-size 5000

echo "=================================================="
echo "Proceso finalizado."
echo "=================================================="
