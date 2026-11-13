#!/bin/bash
#SBATCH --account=proj-606
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/baralgb/proj606/logs/%x-%j.out
# Builds a family's declared copies on one GPU: PREP in order, then TASKS four at a time. Computes no score. Resubmitting is safe: finished runs are skipped.
# Usage: sbatch --job-name=copies-FAM main_copies.sbatch PREP.txt TASKS.txt
R=/home/baralgb/proj606
export HF_HOME=$R/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false; cd $R/code
while read -r line; do bash -c "$line" 2>&1 | grep -E "^done|exists|Traceback|Error" | cut -c1-200; done < "$1"
echo PREP_DONE
cat "$2" | xargs -P 4 -I{} bash -c '{} 2>&1 | grep -E "^done|Traceback|Error" | cut -c1-200'
echo COPIES_DONE
