#!/bin/bash
#SBATCH --account=proj-606
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --job-name=pilot-timing2
#SBATCH --output=/home/baralgb/proj606/logs/%x-%j.out
# Timing and training-depth checks only. No model here is scored; timing models are deleted after.
R=/home/baralgb/proj606; PY=$R/.venv/bin/python; T=$R/models/pilot_timing
export HF_HOME=$R/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
run() { $PY $R/code/train_model.py --base $1 --revision $2 --dataset ultrachat --pool sibling --recipe $3 --lr $4 --steps $5 --seed $6 --out $T/$7 2>&1 | grep -E '^done|Error|error' ; }
SM=$(cat $R/data/smollm2.rev); QW=$(cat $R/data/qwen3.rev); TL=$(cat $R/data/tinyllama.rev)
echo "== depth, SmolLM2"
run HuggingFaceTB/SmolLM2-135M $SM full 5e-5 2000 11 d_full5e5 & run HuggingFaceTB/SmolLM2-135M $SM full 2e-5 2000 12 d_full2e5 & run HuggingFaceTB/SmolLM2-135M $SM lora16 1e-4 2000 13 d_lora & wait
for spec in "qw Qwen/Qwen3-0.6B $QW" "tl TinyLlama/TinyLlama-1.1B-Chat-v1.0 $TL"; do set -- $spec
  for N in 1 4 6; do t=$(date +%s); for i in $(seq 1 $N); do run $2 $3 full 2e-5 200 $((N*10+i)) $1_n${N}_$i & done; wait; echo "$1 full N=$N wall=$(( $(date +%s)-t ))s mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"; done
  t=$(date +%s); for i in 1 2 3 4; do run $2 $3 lora16 1e-4 200 $((90+i)) $1_l_$i & done; wait; echo "$1 lora N=4 wall=$(( $(date +%s)-t ))s"
done
$PY - <<P
import glob,json
for f in sorted(glob.glob("$T/*/meta.json")):
    m=json.load(open(f)); print(f.split("/")[-2],m["recipe"],m["lr"],m["steps"],"train",round(m["train_seconds"]),"total",round(m["total_seconds"]),"gb",round(m["gpu_mem_gb"],1),"loss",round(m["loss_first"],2),round(m["loss_last"],2))
P
du -sh $T/*_n1_1 $T/*_l_1
