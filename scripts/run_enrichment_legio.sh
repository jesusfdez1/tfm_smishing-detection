#!/bin/bash
#SBATCH --job-name=metasms_enrich
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=computo

# Nos movemos a la carpeta del proyecto en tu HOME
cd $HOME/tfm_smishing-detection

echo "=================================================="
echo "Iniciando enriquecimiento (resume) en HOME"
echo "=================================================="

# Habilitamos los alias de bash
shopt -s expand_aliases
export MY_ENV="tfm_smishing"
module load containers/cuda-12.4-uv

export PYTHONUNBUFFERED=1

# Instalamos requirements
python -m pip install -r requirements.txt

# Ejecutamos el enriquecimiento con el flag --resume
# Asegúrate de modificar los nombres de los archivos input/output según lo que estabas usando
python src/dataset/enrich_metadataset.py \
    --input data/processed/metasms_unlabeled.csv \
    --output data/processed/metasms_enriched.csv \
    --resume

echo "=================================================="
echo "Proceso finalizado."
echo "=================================================="
