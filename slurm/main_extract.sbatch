#!/bin/bash
#SBATCH --account=proj-606
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/baralgb/proj606/logs/%x-%A_%a.out
# Confirmatory feature extraction for one family, one shard per array task. These are scores: submit only after the new manifest is stamped.
# Usage: sbatch --job-name=extract-FAM --array=0-(N-1)%K main_extract.sbatch FAMILY N. Resubmitting is safe: finished models are skipped.
R=/home/baralgb/proj606; PY=$R/.venv/bin/python; C=$R/code; O=$R/outputs/main/features/$1; S=$SLURM_ARRAY_TASK_ID
export HF_HOME=$R/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false; cd $C
$PY extract_blackbox.py $O/manifest.json $R/data/prompts.json $R/data/reef.json $O/bb $S $2 2>&1 | grep -E '^FAIL|Traceback' | cut -c1-200
$PY extract_whitebox.py $O/manifest.json $O/claimants.json $O/wb $S $2 2>&1 | grep -E '^FAIL|Traceback' | cut -c1-200
echo EXTRACT_DONE
