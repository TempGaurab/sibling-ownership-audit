#!/bin/bash
#SBATCH --account=proj-606
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=llmprint-ac
#SBATCH --output=/home/baralgb/proj606/logs/%x-%j.out
# Compares float32 and bf16-autocast GCG search: time on all sizes, final search loss on the pilot family. Builds fingerprints only; no suspect is scored.
R=/home/baralgb/proj606; PY=$R/.venv/bin/python; O=$R/outputs/llmprint_ac; C=$R/code
export HF_HOME=$R/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false; rm -rf $O; mkdir -p $O; cd $C
for spec in "smollm2 pilot2 HuggingFaceTB/SmolLM2-135M 8 200" "qwen3 qwen3 Qwen/Qwen3-0.6B 2 100" "tinyllama tinyllama TinyLlama/TinyLlama-1.1B-Chat-v1.0 2 100"; do set -- $spec
  for mode in f32 bf16; do
    fl=""; [ $mode = bf16 ] && fl="--autocast"
    $PY llmprint.py build --model $R/models/$2/claimant/c0_ultrachat --base $3 --revision $(cat $R/data/$1.rev) --out $O/$1_$mode.jsonl --pairs $4 --iters $5 $fl 2>&1 | grep -E 'Traceback|Error' | cut -c1-200
    $PY -c "
import json,numpy as np
r=[json.loads(l) for l in open('$O/$1_$mode.jsonl')]
print('$1','$mode','pairs',len(r),'s/pair',round(np.mean([x['seconds'] for x in r]),1),'mean final loss',round(np.mean([x['loss'] for x in r]),4),flush=True)"
  done
done
echo AC_DONE
