#!/bin/bash
#SBATCH --job-name=ml_pipeline
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=computo
#SBATCH --exclusive

# 1. Definir variables de directorios
HOME_DIR="$HOME/tfm_smishing-detection"
SCRATCH_DIR="/local_scratch/$SLURM_JOB_ID/tfm_smishing-detection"

echo "=================================================="
echo "Preparando entorno en SSD local (/local_scratch)"
echo "=================================================="

# 2. Configurar limpieza automática (incluso si el job falla o es cancelado)
trap 'echo "Limpiando $SCRATCH_DIR..."; rm -rf /local_scratch/$SLURM_JOB_ID' EXIT

# 3. Crear directorio temporal en el SSD local
mkdir -p "$SCRATCH_DIR"

# 4. Copiar el código y los datos de entrada al SSD local
# Se excluyen carpetas de salida previas
rsync -avq --exclude='output' --exclude='logs' "$HOME_DIR/" "$SCRATCH_DIR/"

# 5. Moverse al SSD local para que toda la E/S ocurra ahí
cd "$SCRATCH_DIR"

# Creamos las carpetas necesarias en el SSD
mkdir -p output/ml

echo "=================================================="
echo "Iniciando ml_pipeline en: $PWD"
echo "=================================================="

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
echo "Proceso finalizado. Sincronizando resultados a HOME..."
echo "=================================================="

# 6. Sincronizar de vuelta los resultados al almacenamiento en red (CEPH)
mkdir -p "$HOME_DIR/output/ml/"
rsync -avq "$SCRATCH_DIR/output/ml/" "$HOME_DIR/output/ml/"

echo "=================================================="
echo "Sincronización completa. Trabajo terminado exitosamente."
echo "=================================================="
