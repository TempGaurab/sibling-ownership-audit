#!/bin/bash
#SBATCH --account=proj-606
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=pilot2
#SBATCH --output=/home/baralgb/proj606/logs/%x-%j.out
# Pilot family only. Training check after the float32 fix; never reported as results.
R=/home/baralgb/proj606
export HF_HOME=$R/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
cat $R/outputs/pilot2/tasks.txt | xargs -P 6 -I{} bash -c '{} 2>&1 | grep -E "^done|Traceback|Error" | cut -c1-160'
