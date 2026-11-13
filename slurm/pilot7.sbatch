#!/bin/bash
#SBATCH --account=proj-606
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=pilot7
#SBATCH --output=/home/baralgb/proj606/logs/%x-%j.out
# Pilot family only, never reported. Fresh extraction with current code (Witness Overlap, REEF per paper, float32), then first real run of audit.py and analyze.py.
R=/home/baralgb/proj606; PY=$R/.venv/bin/python; O=$R/outputs/pilot7; C=$R/code; P=$R/outputs/pilot3; B=HuggingFaceTB/SmolLM2-135M; V=$(cat $R/data/smollm2.rev)
export HF_HOME=$R/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false; mkdir -p $O; cd $C
$PY make_manifest.py $R pilot2 $B $V $O
$PY - <<PY
import json
m=json.load(open("$O/manifest.json")); c=[e for e in json.load(open("$P/manifest.json")) if e.get("kind")=="control"]
json.dump(m+c,open("$O/manifest.json","w"),indent=0); print("controls",len(c),"total",len(m)+len(c))
PY
$PY extract_blackbox.py $O/manifest.json $R/data/prompts.json $R/data/reef.json $O/bb 0 1 2>&1 | grep -E '^FAIL|Traceback' | cut -c1-200
$PY extract_whitebox.py $O/manifest.json $O/claimants.json $O/wb 0 1 2>&1 | grep -E '^FAIL|Traceback' | cut -c1-200
echo "features: bb $(ls $O/bb | wc -l), wb $(ls $O/wb | wc -l)"
$PY make_meta.py $O/manifest.json $O/meta.json
$PY audit.py smollm2 $O $O/manifest.json $O/claimants.json none $O/meta.json $O/scores.csv $O/rules.jsonl 2>&1 | grep -E 'AUDIT_DONE|Traceback|Error' | cut -c1-300
$PY - <<PY
import pandas as pd
d=pd.read_csv("$O/scores.csv"); print(len(d),"rows"); print(d.groupby(["kind"]).suspect.nunique().to_string())
print(d[d.kind=="copy"].stratum.value_counts().to_string()); print(d[d.kind=="sib"].groupby(["stratum"]).size().to_string()); print(d[d.kind=="sib"][["dataset","recipe"]].drop_duplicates().to_string())
print("nan by score", d.groupby("score").value.apply(lambda v:int(v.isna().sum())).to_dict())
PY
$PY analyze.py $O/scores.csv $O/analysis.json 2>&1 | tail -8 | cut -c1-300; ls -la $O/analysis.json
echo PILOT7_DONE
