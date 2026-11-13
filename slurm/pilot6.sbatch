#!/bin/bash
#SBATCH --account=proj-606
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=pilot6
#SBATCH --output=/home/baralgb/proj606/logs/%x-%j.out
# Pilot family only, never reported. First real runs of the keyed fingerprint and of audit.py and analyze.py on real features.
R=/home/baralgb/proj606; PY=$R/.venv/bin/python; O=$R/outputs/pilot6; C=$R/code; P=$R/outputs/pilot3; M=$R/models/pilot2; K=$R/keys/pilot
export HF_HOME=$R/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false; mkdir -p $O $K; chmod 700 $R/keys; cd $C
echo "== audit"
$PY make_meta.py $P/manifest.json $O/meta.json
$PY audit.py smollm2 $P $P/manifest.json $P/claimants.json none $O/meta.json $O/scores.csv $O/rules.jsonl 2>&1 | grep -E 'AUDIT_DONE|Traceback|Error' | cut -c1-300
$PY - <<PY
import pandas as pd
d=pd.read_csv("$O/scores.csv"); print(len(d),"rows"); print(d.groupby(["kind"]).suspect.nunique().to_string())
print(d[d.kind=="copy"].stratum.value_counts().to_string()); print(d[d.kind=="sib"].groupby(["stratum"]).size().to_string()); print(d[d.kind=="sib"][["dataset","recipe"]].drop_duplicates().to_string())
print("nan values", int(d.value.isna().sum()), "by score", d.groupby("score").value.apply(lambda v:int(v.isna().sum())).to_dict())
PY
echo "== analyze"
$PY analyze.py $O/scores.csv $O/analysis.json 2>&1 | tail -8 | cut -c1-300; ls -la $O/analysis.json
echo "== keyed"
for spec in "c0_ultrachat ultrachat 2e-05 70000" "c1_oasst1 oasst1 2e-05 70001"; do set -- $spec
  $PY train_keyed.py --keys $K/$1.json --base $M/root/$1 --dataset $2 --pool all --recipe full --lr $3 --steps 2000 --seed $4 --out $M/keyed/$1 2>&1 | grep -E '^done|Traceback|Error' | cut -c1-200
done
$PY - <<PY
import json
m=json.load(open("$R/outputs/pilot4/small.json"))
m+=[dict(id="keyed_"+c,kind="keyed",path="$M/keyed/"+c,base="$M/root/"+c,base_revision=None,adapter=False) for c in ("c0_ultrachat","c1_oasst1")]
json.dump(m,open("$O/keyed_manifest.json","w")); print(len(m))
PY
$PY score_keyed.py $O/keyed_manifest.json $K $O/keyed.jsonl 2>&1 | grep -E 'FAIL|Traceback' | cut -c1-200
cut -c1-200 $O/keyed.jsonl
echo PILOT6_DONE
