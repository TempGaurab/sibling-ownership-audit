#!/bin/bash
#SBATCH --account=proj-606
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=hub-check
#SBATCH --output=/home/baralgb/proj606/logs/%x-%j.out
# Provenance weight check against the declared base only. Computes no score against any claimant. Resubmitting resumes.
R=/home/baralgb/proj606; PY=$R/.venv/bin/python; O=$R/outputs/hub; C=$R/code
export HF_HOME=$R/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
$PY $C/make_hub_manifest.py $R $O
$PY $C/check_weights.py $O/hub_manifest.json $O/weight_check.jsonl 0 1 2>&1 | grep -E 'CHECK_DONE|Traceback' | cut -c1-200
