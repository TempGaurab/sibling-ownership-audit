#!/bin/bash
#SBATCH --account=proj-606
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --job-name=llmprint
#SBATCH --output=/home/baralgb/proj606/logs/%x-%A_%a.out
# Builds the full LLMPrint fingerprint on one claimant per array task (0 to 5), two processes on one GPU. No suspect is scored. Resubmitting resumes.
R=/home/baralgb/proj606; PY=$R/.venv/bin/python; C=$R/code; I=$SLURM_ARRAY_TASK_ID
SPECS=("qwen3 Qwen/Qwen3-0.6B c0_ultrachat" "qwen3 Qwen/Qwen3-0.6B c1_oasst1" "qwen3 Qwen/Qwen3-0.6B c2_gsm8k" "tinyllama TinyLlama/TinyLlama-1.1B-Chat-v1.0 c0_ultrachat" "tinyllama TinyLlama/TinyLlama-1.1B-Chat-v1.0 c1_oasst1" "tinyllama TinyLlama/TinyLlama-1.1B-Chat-v1.0 c2_gsm8k")
set -- ${SPECS[$I]}; FAM=$1; BASE=$2; CID=$3; O=$R/outputs/main/llmprint/$FAM/parts; mkdir -p $O
export HF_HOME=$R/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4; cd $C
for s in 0 1; do
  $PY llmprint.py build --model $R/models/$FAM/claimant/$CID --base $BASE --revision $(cat $R/data/$FAM.rev) --out $O/$CID.part$s.jsonl --pairs 300 --iters 1000 --shard $s --nshards 2 --autocast 2>&1 | grep -E 'Traceback|Error' | cut -c1-200 &
done
wait
echo "$FAM $CID pairs $(cat $O/$CID.part*.jsonl | wc -l)"; echo LLMPRINT_DONE
