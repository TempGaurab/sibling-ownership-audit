#!/bin/bash
#SBATCH --account=proj-606
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --job-name=llmprint-pack
#SBATCH --output=/home/baralgb/proj606/logs/%x-%j.out
# Measures how much GCG throughput improves when packing processes on one GPU. Builds fingerprints only; no suspect is scored.
R=/home/baralgb/proj606; PY=$R/.venv/bin/python; O=$R/outputs/llmprint_pack; C=$R/code
export HF_HOME=$R/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2; mkdir -p $O; cd $C
for spec in "smollm2 pilot2 HuggingFaceTB/SmolLM2-135M" "qwen3 qwen3 Qwen/Qwen3-0.6B" "tinyllama tinyllama TinyLlama/TinyLlama-1.1B-Chat-v1.0"; do set -- $spec
  for P in 1 2 4 6; do
    rm -f $O/$1_${P}_*.jsonl; t0=$(date +%s)
    for ((i=0;i<P;i++)); do
      $PY llmprint.py build --model $R/models/$2/claimant/c0_ultrachat --base $3 --revision $(cat $R/data/$1.rev) --out $O/$1_${P}_$i.jsonl --pairs $((2*P)) --iters 100 --shard $i --nshards $P > /dev/null 2>&1 &
    done
    wait; t1=$(date +%s)
    n=$(cat $O/$1_${P}_*.jsonl | wc -l); echo "$1 procs=$P pairs=$n wall=$((t1-t0))s"
  done
done
echo PACK_DONE
