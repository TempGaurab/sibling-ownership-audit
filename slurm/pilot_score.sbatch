#!/bin/bash
#SBATCH --account=proj-606
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=pilot-score
#SBATCH --output=/home/baralgb/proj606/logs/%x-%j.out
# Pipeline dry run on the pilot family only. Never reported as results.
R=/home/baralgb/proj606; PY=$R/.venv/bin/python; O=$R/outputs/pilot
export HF_HOME=$R/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
$PY $R/code/extract_blackbox.py $O/manifest_all.json $R/data/prompts.json $O/bb 0 1 2>&1 | grep -E '^(ok|FAIL)|Traceback|Error' | cut -c1-200
$PY $R/code/extract_whitebox.py $O/manifest_same.json $O/claimants.json $O/wb 0 1 2>&1 | grep -E '^(ok|FAIL)|Traceback|Error' | cut -c1-200
