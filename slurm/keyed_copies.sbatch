#!/bin/bash
#SBATCH --account=proj-606
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/baralgb/proj606/logs/%x-%j.out
# Nov 12 deviation 6c40f1e0353f (post hoc): copies of the keyed claimants through the frozen copy grid, then keyed scoring of
# those copies with the unchanged score_keyed.py. One GPU per family. Resubmitting is safe: finished copies and scored models are skipped.
# Usage: sbatch --job-name=kcopies-FAM keyed_copies.sbatch FAM BASE REV   (PAR=N sets parallel copy builds, default 4)
R=/home/baralgb/proj606; PY=$R/.venv/bin/python; C=$R/code; FAM=$1; O=$R/outputs/keyed_copies; mkdir -p $O
export HF_HOME=$R/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false; cd $C
$PY make_keyed_copies.py $FAM $2 $3 $R 5 5 $O/prep_$FAM.txt $O/tasks_$FAM.txt || exit 1
while read -r line; do bash -c "$line" 2>&1 | grep -E "^done|exists|Traceback|Error" | cut -c1-200; done < $O/prep_$FAM.txt
echo PREP_DONE
cat $O/tasks_$FAM.txt | xargs -P ${PAR:-4} -I{} bash -c '{} 2>&1 | grep -E "^done|Traceback|Error" | cut -c1-200'
echo COPIES_DONE
$PY - $O/tasks_$FAM.txt $O/manifest_$FAM.json $2 $3 <<'PY'
import json, os, sys
outs = [l.split("--out ")[1].split()[0] for l in open(sys.argv[1]) if l.strip()]
ok = [p for p in outs if os.path.exists(f"{p}/meta.json")]
def entry(p):  # same rule as make_manifest.py: LoRA copies are adapters on the keyed root
    e = dict(id="k_" + os.path.basename(p), kind="kcopy", path=p, base=sys.argv[3], base_revision=sys.argv[4], adapter=False)
    if os.path.exists(f"{p}/adapter_config.json"):
        e["adapter"] = True; b = json.load(open(f"{p}/adapter_config.json"))["base_model_name_or_path"]
        if b != sys.argv[3]: e["base"], e["base_revision"] = b, None
    return e
json.dump([entry(p) for p in ok], open(sys.argv[2], "w"))
print("MANIFEST", len(ok), "of", len(outs))
PY
$PY score_keyed.py $O/manifest_$FAM.json $R/keys/$FAM $O/keyed_$FAM.jsonl 2>&1 | grep -E 'FAIL|Traceback' | cut -c1-200
echo "SCORED $(wc -l < $O/keyed_$FAM.jsonl)"
echo KCOPIES_DONE
