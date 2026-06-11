#!/bin/bash
#SBATCH --job-name=tfm_dl
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

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
rsync -avq --exclude='output' --exclude='logs' "$HOME_DIR/" "$SCRATCH_DIR/"

# 5. Moverse al SSD local para que toda la E/S ocurra ahí
cd "$SCRATCH_DIR"

# Creamos las carpetas necesarias en el SSD
mkdir -p output/dl/checkpoints

echo "=================================================="
echo "Iniciando pipeline de Deep Learning en: $PWD"
echo "=================================================="

# Habilitamos los alias de bash para que funcionen los comandos del módulo (uv, python)
shopt -s expand_aliases

# Siguiendo la tesis: Cargamos el módulo de contenedores HPC oficial
export MY_ENV="tfm_smishing"
module load containers/cuda-12.4-uv

export PYTHONUNBUFFERED=1

echo "=================================================="
echo "Ejecutando Python desde la imagen HPC"
echo "=================================================="

# Instalamos los requirements por si falta alguno (acelerado por uv)
python -m pip install -r requirements.txt

# Ejecutamos el entrenamiento y evaluación de Modelos de Lenguaje Pequeños (SLMs)
python -m src.experiments.dl.main --data_root data/processed --out_dir output/dl --batch_size 16 --epochs 3

echo "=================================================="
echo "Proceso finalizado. Sincronizando resultados a HOME..."
echo "=================================================="

# 6. Sincronizar de vuelta los resultados (y checkpoints) al almacenamiento en red (CEPH)
mkdir -p "$HOME_DIR/output/dl/"
rsync -avq "$SCRATCH_DIR/output/dl/" "$HOME_DIR/output/dl/"

echo "=================================================="
echo "Sincronización completa. Trabajo terminado exitosamente."
echo "=================================================="
