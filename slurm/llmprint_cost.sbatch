#!/bin/bash
#SBATCH --account=proj-606
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=llmprint-cost
#SBATCH --output=/home/baralgb/proj606/logs/%x-%j.out
# Measures GCG cost per prompt on one claimant of each size. Builds a fingerprint only; no suspect is scored.
R=/home/baralgb/proj606; PY=$R/.venv/bin/python; O=$R/outputs/llmprint_cost; C=$R/code
export HF_HOME=$R/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false; mkdir -p $O; cd $C
for spec in "smollm2 pilot2 HuggingFaceTB/SmolLM2-135M" "qwen3 qwen3 Qwen/Qwen3-0.6B" "tinyllama tinyllama TinyLlama/TinyLlama-1.1B-Chat-v1.0"; do set -- $spec
  echo "cost $1"; $PY llmprint.py build --model $R/models/$2/claimant/c0_ultrachat --base $3 --revision $(cat $R/data/$1.rev) --out $O/cost_$1.jsonl --pairs 2 --iters 100 2>&1 | grep -E '^ok|Traceback|Error' | cut -c1-200
done
echo COST_DONE
