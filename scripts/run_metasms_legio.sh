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
# Se excluyen datos procesados previos para no saturar la copia
rsync -avq --exclude='data/processed' --exclude='logs' "$HOME_DIR/" "$SCRATCH_DIR/"

# 5. Moverse al SSD local para que toda la E/S ocurra ahí
cd "$SCRATCH_DIR"

# Creamos las carpetas necesarias en el SSD
mkdir -p logs data/processed

echo "=================================================="
echo "Iniciando build_metasms en: $PWD"
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
echo "Proceso finalizado. Sincronizando resultados a HOME..."
echo "=================================================="

# 6. Sincronizar de vuelta los datos procesados y logs al almacenamiento en red (CEPH)
rsync -avq "$SCRATCH_DIR/data/processed/" "$HOME_DIR/data/processed/"
rsync -avq "$SCRATCH_DIR/logs/" "$HOME_DIR/logs/"

echo "=================================================="
echo "Sincronización completa. Trabajo terminado exitosamente."
echo "=================================================="
