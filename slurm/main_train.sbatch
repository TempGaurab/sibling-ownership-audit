#!/bin/bash
#SBATCH --account=proj-606
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/baralgb/proj606/logs/%x-%j.out
# Confirmatory training for one family on one GPU. Resubmitting is safe: finished runs are skipped.
# Usage: sbatch --job-name=train-FAM main_train.sbatch TASKS.txt
R=/home/baralgb/proj606
export HF_HOME=$R/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
cat "$1" | xargs -P 4 -I{} bash -c '{} 2>&1 | grep -E "^done|Traceback|Error" | cut -c1-160'
