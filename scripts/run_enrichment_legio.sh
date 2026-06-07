#!/bin/bash
#SBATCH --job-name=metasms_enrich
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:3
#SBATCH --partition=computo

# Nos movemos a la carpeta del proyecto en tu HOME
cd $HOME/tfm_smishing-detection

echo "=================================================="
echo "Iniciando enriquecimiento PARALELO en HOME"
echo "=================================================="

# Habilitamos los alias de bash
shopt -s expand_aliases
export MY_ENV="tfm_smishing"
module load containers/cuda-12.4-uv

export PYTHONUNBUFFERED=1

# DOPAJE DE ALMACENAMIENTO: Caché ultrarrápida
export HF_HOME=/local_scratch/$USER/huggingface_cache
mkdir -p $HF_HOME

# Instalamos requirements esenciales y librerías clave
python -m pip install -r requirements.txt
python -m pip install accelerate transformers>=4.48.0 torchvision pillow python-whois
# Instalación forzada del "Fast Path" evitando el fallo del entorno aislado
python -m pip install ninja packaging
python -m pip install flash-attn causal-conv1d flash-linear-attention --no-build-isolation

# DOPAJE MULTI-GPU: Partición de datos
INPUT_CSV="data/processed/metasms_v1_privacy.csv"
echo "Dividiendo dataset en 3 partes..."

# Creamos directorio temporal
mkdir -p data/processed/tmp_split
HEADER=$(head -n 1 $INPUT_CSV)
tail -n +2 $INPUT_CSV > data/processed/tmp_split/body.csv
split -d -n l/3 data/processed/tmp_split/body.csv data/processed/tmp_split/part_

# Preparamos las 3 partes
for i in 00 01 02; do
    echo "$HEADER" > data/processed/tmp_split/input_$i.csv
    cat data/processed/tmp_split/part_$i >> data/processed/tmp_split/input_$i.csv
done

echo "Lanzando 3 instancias en paralelo..."

# Ejecutamos las 3 en paralelo (cada una anclada a una GPU)
CUDA_VISIBLE_DEVICES=0 python src/dataset/enrich_metadataset.py \
    --input data/processed/tmp_split/input_00.csv \
    --output data/processed/tmp_split/output_00.csv \
    --batch-size 64 --resume &

CUDA_VISIBLE_DEVICES=1 python src/dataset/enrich_metadataset.py \
    --input data/processed/tmp_split/input_01.csv \
    --output data/processed/tmp_split/output_01.csv \
    --batch-size 64 --resume &

CUDA_VISIBLE_DEVICES=2 python src/dataset/enrich_metadataset.py \
    --input data/processed/tmp_split/input_02.csv \
    --output data/processed/tmp_split/output_02.csv \
    --batch-size 64 --resume &

# Esperamos a que terminen las 3
wait

echo "Fusionando resultados..."
echo "$HEADER" > data/processed/metasms_v1_enriched.csv
# Fusionamos asegurándonos de que existen
[ -f data/processed/tmp_split/output_00.csv ] && tail -n +2 data/processed/tmp_split/output_00.csv >> data/processed/metasms_v1_enriched.csv
[ -f data/processed/tmp_split/output_01.csv ] && tail -n +2 data/processed/tmp_split/output_01.csv >> data/processed/metasms_v1_enriched.csv
[ -f data/processed/tmp_split/output_02.csv ] && tail -n +2 data/processed/tmp_split/output_02.csv >> data/processed/metasms_v1_enriched.csv

echo "Limpiando archivos temporales..."
rm -r data/processed/tmp_split

echo "=================================================="
echo "Proceso PARALELO finalizado."
echo "=================================================="
