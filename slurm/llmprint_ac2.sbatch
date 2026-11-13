#!/bin/bash
#SBATCH --account=proj-606
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --job-name=llmprint-ac2
#SBATCH --output=/home/baralgb/proj606/logs/%x-%j.out
# Packing under bf16 autocast, Qwen3 claimant c0. Throughput uses per-pair search seconds, so model loading is not counted. Builds fingerprints only; no suspect is scored.
R=/home/baralgb/proj606; PY=$R/.venv/bin/python; O=$R/outputs/llmprint_ac2; C=$R/code
export HF_HOME=$R/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2; rm -rf $O; mkdir -p $O; cd $C
for P in 1 2 3 4; do
  for ((i=0;i<P;i++)); do
    $PY llmprint.py build --model $R/models/qwen3/claimant/c0_ultrachat --base Qwen/Qwen3-0.6B --revision $(cat $R/data/qwen3.rev) --out $O/q_${P}_$i.jsonl --pairs $((3*P)) --iters 100 --shard $i --nshards $P --autocast > /dev/null 2>&1 &
  done
  wait
  $PY -c "
import glob,json,numpy as np
s=[json.loads(l)['seconds'] for f in glob.glob('$O/q_${P}_*.jsonl') for l in open(f)]
print('procs',$P,'pairs',len(s),'s/pair',round(np.mean(s),1),'pairs per 1000s',round(1000*$P/np.mean(s),1),flush=True)"
done
echo AC2_DONE
