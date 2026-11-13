#!/bin/bash
#SBATCH --account=proj-606
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=pilot4
#SBATCH --output=/home/baralgb/proj606/logs/%x-%j.out
# Pilot family only for scores; never reported. Quantization agreement, new extractors, LLMPrint cost on three model sizes.
R=/home/baralgb/proj606; PY=$R/.venv/bin/python; O=$R/outputs/pilot4; C=$R/code; B=HuggingFaceTB/SmolLM2-135M; V=$(cat $R/data/smollm2.rev)
export HF_HOME=$R/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
mkdir -p $O; cd $C
$PY - <<PY 2>&1 | grep -vE "Warning|warn" | cut -c1-200
import glob,json,os,sys,numpy as np
sys.argv=["x"]
from extract_blackbox import load,nexttokens
P=json.load(open("$R/data/prompts.json"))[:5000]; f={}
def e(i,p): return dict(id=i,path=p,base="$B",base_revision="$V",adapter=os.path.exists(p+"/adapter_config.json"))
for c in ["c0_ultrachat","c1_oasst1","c2_gsm8k"]:
    m,t=load(e(c,"$R/models/pilot2/claimant/"+c)); f[c]=nexttokens(m,t,P); del m
    for b in (8,4):
        q=f"$R/outputs/pilot3/qtest/m/{c}_q{b}"
        if os.path.exists(q+"/meta.json"):
            m,t=load(e(c+str(b),q)); print(c,b,"bit agreement with claimant",round(float((nexttokens(m,t,P)==f[c]).mean()),3),flush=True); del m
PY
$PY make_manifest.py $R pilot2 $B $V $O
$PY - <<PY
import json
m=json.load(open("$O/manifest.json")); keep=[e for e in m if e["kind"] in("claimant","base")]+[e for e in m if e["kind"]=="sib"][:4]+[e for e in m if e["kind"]=="copy"][:4]
json.dump(keep,open("$O/small.json","w")); print(len(keep))
PY
$PY extract_blackbox.py $O/small.json $R/data/prompts.json $R/data/reef.json $O/bb 0 1 2>&1 | grep -E '^(ok|FAIL)|Traceback|Error' | cut -c1-200
$PY extract_whitebox.py $O/small.json $O/claimants.json $O/wb 0 1 2>&1 | grep -E '^(ok|FAIL)|Traceback|Error' | cut -c1-200
for spec in "smollm2 $B $V" "qwen3 Qwen/Qwen3-0.6B $(cat $R/data/qwen3.rev)" "tinyllama TinyLlama/TinyLlama-1.1B-Chat-v1.0 $(cat $R/data/tinyllama.rev)"; do set -- $spec
  echo "cost $1"; $PY llmprint.py build --model $2 --base $2 --revision $3 --out $O/cost_$1.jsonl --pairs 2 --iters 100 2>&1 | grep -E '^ok|Traceback|Error' | cut -c1-200
done
$PY - <<PY 2>&1 | cut -c1-220
import glob,json,numpy as np
for f in sorted(glob.glob("$O/wb/*.json")):
    r=json.load(open(f)); print(r["id"],"wo_base",r["wo_base"],list(r["scores"].items())[:1])
for f in sorted(glob.glob("$O/bb/*/meta.json"))[:3]: print(json.load(open(f))["seconds"])
PY
echo PILOT4_DONE
