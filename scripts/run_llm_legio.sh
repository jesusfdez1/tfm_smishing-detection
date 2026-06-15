#!/bin/bash
#SBATCH --job-name=tfm_llm
#SBATCH --output=%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=computo
#SBATCH --gres=gpu:2

cd $HOME/tfm_smishing-detection
mkdir -p output/llm

export PYTHONUNBUFFERED=1
export HF_HOME=$HOME/.cache/huggingface
if [ -f .env ]; then export $(grep -E '^HF_TOKEN' .env | xargs); fi
mkdir -p $HF_HOME

# ==============================================================================
# En lugar de usar el 'module load' que nos inyectaba comandos ocultos (--no-home, --overlay)
# llamamos directamente a Apptainer.
#
# Al crear el entorno en $HOME (CEPH) y tener la caché en $HOME (CEPH),
# uv PUEDE HACER HARDLINKS NATIVOS. 0 crasheos de concurrencia y se instala en 2 segundos.
# ==============================================================================
CONTAINER="/opt/ohpc/pub/containers/cuda-12.4-uv.sif"

echo "=================================================="
echo "1. Creando entorno virtual limpio en el directorio local..."
echo "=================================================="
rm -rf .venv_llm
# Ejecutamos uv desde dentro del contenedor pelado
apptainer exec --nv $CONTAINER uv venv .venv_llm --python 3.11

echo "=================================================="
echo "2. Instalando dependencias velozmente (Hardlinks en CEPH)..."
echo "=================================================="
# Esta vez no saldrá el "warning: Failed to hardlink" y no se colgará
apptainer exec --nv $CONTAINER uv pip install --python .venv_llm pandas scikit-learn datasets transformers accelerate torch sentencepiece huggingface_hub jinja2 tiktoken ninja vllm>=0.5.0

echo "=================================================="
echo "3. Autenticando HuggingFace..."
echo "=================================================="
apptainer exec --nv $CONTAINER .venv_llm/bin/python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"

echo "=================================================="
echo "4. >> [PHASE 3] LLM Evaluation (Edge vs Server)"
echo "=================================================="
# Solución al cuelgue de vLLM en Multi-GPU (NCCL Hang) dentro de Apptainer/SLURM
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

apptainer exec --nv $CONTAINER bash -c "export PATH=\$PWD/.venv_llm/bin:\$PATH && export NCCL_IB_DISABLE=1 && export NCCL_P2P_DISABLE=1 && .venv_llm/bin/python -m src.experiments.llm.main \
    --test_samples 10000 \
    --models phi4_mini gemma3_4b qwen35_4b ministral_8b falcon3_7b olmoe_1b_7b moonlight_16b gemma4_12b deepseek_r1_14b qwen25_14b mistral_nemo_12b qwen35_35b_moe"

echo "=================================================="
echo "Process finished. Job completed successfully."
echo "=================================================="
