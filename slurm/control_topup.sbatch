#!/bin/bash
#SBATCH --account=proj-606
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=control-topup
#SBATCH --output=/home/baralgb/proj606/logs/%x-%j.out
# Fills each control base to 11 usable models from the seeded reserves. Submit with --dependency=afterany on the Qwen3 extraction
# array, so the list of lost controls is final and total use stays at 3 GPUs. Resubmitting is safe.
R=/home/baralgb/proj606; PY=$R/.venv/bin/python; C=$R/code
export HF_HOME=$R/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false; cd $C
$PY topup_controls.py $R $R/data/topup_list.json 2>&1 | cut -c1-300
