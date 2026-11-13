#!/bin/bash
#SBATCH --account=proj-606
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --job-name=pilot3
#SBATCH --output=/home/baralgb/proj606/logs/%x-%j.out
# Pilot family only, never reported. Precision check, copy pipeline, then extraction on float32 weights.
R=/home/baralgb/proj606; PY=$R/.venv/bin/python; O=$R/outputs/pilot3; C=$R/code; F=pilot2
B=HuggingFaceTB/SmolLM2-135M; V=$(cat $R/data/smollm2.rev)
export HF_HOME=$R/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
mkdir -p $O
$PY $C/compare_precision.py $R 2>&1 | grep -E 'taskcos|Traceback|Error' | cut -c1-200
$PY $C/make_copies.py $F $B $V $R 1 4 $O/prep.txt $O/tasks.txt 2000
cat $O/prep.txt | while read -r l; do bash -c "$l" 2>&1 | grep -E '^done|Traceback|Error' | cut -c1-200; done
cat $O/tasks.txt | xargs -P 6 -I{} bash -c '{} 2>&1 | grep -E "^done|Traceback|Error" | cut -c1-200'
$PY $C/make_manifest.py $R $F $B $V $O
$PY - <<PYEOF
import json
m=json.load(open("$O/manifest.json")); seen={e["id"] for e in m}
for e in json.load(open("$R/outputs/pilot/manifest_all.json")):
    if not e["id"].startswith("pilot_") and e["id"] not in seen: m.append(dict(e, kind="control"))
json.dump(m,open("$O/manifest.json","w"),indent=0)
print("with controls",len(m))
PYEOF
$PY $C/extract_blackbox.py $O/manifest.json $R/data/prompts.json $O/bb 0 1 2>&1 | grep -E '^(ok|FAIL)|Traceback|Error' | cut -c1-200
$PY $C/extract_whitebox.py $O/manifest.json $O/claimants.json $O/wb 0 1 2>&1 | grep -E '^(ok|FAIL)|Traceback|Error' | cut -c1-200
$PY $C/pilot_check.py $O 2>&1 | cut -c1-220
echo PILOT3_DONE
