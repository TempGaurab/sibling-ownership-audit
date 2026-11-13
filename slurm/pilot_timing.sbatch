#!/bin/bash
#SBATCH --account=proj-606
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --job-name=pilot-timing
#SBATCH --output=/home/baralgb/proj606/logs/%x-%j.out
# Pilot family only. Timing and packing on one GPU; these models are never reported as results.
set -eu
R=/home/baralgb/proj606; PY=$R/.venv/bin/python
$PY -c "import torch; assert torch.cuda.is_available()"
set +e
export HF_HOME=$R/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
BASE=HuggingFaceTB/SmolLM2-135M; REV=$(cat $R/data/smollm2.rev); S=500
run() { $PY $R/code/train_model.py --base $BASE --revision $REV --dataset ultrachat --pool sibling --steps $S --recipe $1 --seed $2 --out $R/models/pilot/$3; }
nvidia-smi -L
echo "== single, full"; run full 1 p1_full
echo "== single, lora16"; run lora16 2 p1_lora
for N in 4 8; do
  echo "== packed N=$N, full"; t=$(date +%s)
  for i in $(seq 1 $N); do run full $((100*N+i)) n${N}_$i & done; wait
  echo "packed N=$N wall=$(( $(date +%s)-t ))s"; nvidia-smi --query-gpu=memory.used --format=csv,noheader
done
