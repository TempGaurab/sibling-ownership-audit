#!/bin/bash
#SBATCH --account=proj-606
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=pilot5
#SBATCH --output=/home/baralgb/proj606/logs/%x-%j.out
# Pilot family only, never reported. LLMPrint build and verify end to end at reduced size.
R=/home/baralgb/proj606; PY=$R/.venv/bin/python; O=$R/outputs/pilot5; C=$R/code; B=HuggingFaceTB/SmolLM2-135M; V=$(cat $R/data/smollm2.rev)
export HF_HOME=$R/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false; mkdir -p $O; cd $C
$PY llmprint.py build --model $R/models/pilot2/claimant/c0_ultrachat --base $B --revision $V --out $O/c0_ultrachat.jsonl --pairs 20 --iters 200 2>&1 | grep -E 'Traceback|Error' | cut -c1-200
$PY llmprint.py verify $R/outputs/pilot4/small.json $O/c0_ultrachat.jsonl $O/verify.jsonl 2>&1 | grep -E 'FAIL|Traceback|Error' | cut -c1-200
$PY - <<PY
import json
r={json.loads(l)["id"]:json.loads(l)["bits"]["c0_ultrachat"] for l in open("$O/verify.jsonl")}
ref=r["c0_ultrachat"]; print("claimant bits, share set",sum(b==1 for b in ref),"of",len(ref),"None",sum(b is None for b in ref))
for k,v in r.items(): print(k,"agree",round(sum(a==b for a,b in zip(v,ref))/len(ref),3))
PY
echo PILOT5_DONE
