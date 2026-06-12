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
export MY_ENV="tfm_smishing_ml"
module load containers/cuda-12.4-uv

export PYTHONUNBUFFERED=1

echo "=================================================="
echo "Ejecutando Python parcheado (3.11) para compatibilidad con gensim"
echo "=================================================="

# Usamos la herramienta 'uv' instalada en el módulo para descargar un Python 3.11 normal
uv python install 3.11

# Creamos un entorno virtual aislado con esa versión estable
uv venv .venv_ml --python 3.11

# Lo activamos
source .venv_ml/bin/activate

# Instalamos los requirements a la velocidad del rayo con uv
uv pip install -r requirements.txt

# Ejecutamos el grid search de Machine Learning con el Python estable para TODAS las variantes de características

echo ">> [1/4] Entrenando variante NORM (Base)"
python -m src.experiments.ml.main --data_root data/processed --out_dir output/ml/norm --feature_type norm

echo ">> [2/4] Entrenando variante RAW (Privacidad vs Overfitting)"
python -m src.experiments.ml.main --data_root data/processed --out_dir output/ml/raw --feature_type raw

echo ">> [3/4] Entrenando variante ANONYMIZED"
python -m src.experiments.ml.main --data_root data/processed --out_dir output/ml/anonymized --feature_type anonymized

echo ">> [4/4] Entrenando variante INJECTED (Urgencia + Tema)"
python -m src.experiments.ml.main --data_root data/processed --out_dir output/ml/injected --feature_type injected

echo "=================================================="
echo "Proceso finalizado. Trabajo terminado exitosamente."
echo "=================================================="
