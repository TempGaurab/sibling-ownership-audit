#!/bin/bash
#SBATCH --account=proj-606
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/baralgb/proj606/logs/%x-%j.out
# Nov 11 deviation (DEVIATIONS.md, chat templates): rerun LLMPrint verification and keyed scoring for the one TinyLlama hub sibling
# whose chat template could not render a standard message. Both scripts are idempotent and append to the existing files.
# Exits non-zero unless the model now has exactly one record in each file.
R=/home/baralgb/proj606; PY=$R/.venv/bin/python; C=$R/code; FAM=tinyllama; O=$R/outputs/main/features/$FAM; L=$R/outputs/main/llmprint/$FAM
export HF_HOME=$R/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false; cd $C
ID=Bhooyas/tinyllama-dolly-15k; SUB=$O/manifest_template_nov11.json
$PY - "$O/manifest.json" "$SUB" "$ID" <<'PY'
import json, sys
m = [e for e in json.load(open(sys.argv[1])) if e["id"] == sys.argv[3]]
assert len(m) == 1, "model not in manifest"
json.dump(m, open(sys.argv[2], "w")); print("SUBSET", len(m))
PY
[ $? = 0 ] || exit 1
FP=$(ls $L/*.jsonl | tr '\n' ',' | sed 's/,$//')
$PY llmprint.py verify $SUB "$FP" $O/llmprint.jsonl 2>&1 | grep -E '^FAIL|Traceback' | cut -c1-200
$PY score_keyed.py $SUB $R/keys/$FAM $O/keyed.jsonl 2>&1 | grep -E 'FAIL|Traceback' | cut -c1-200
a=$(grep -c "\"id\": \"$ID\"" $O/llmprint.jsonl); b=$(grep -c "\"id\": \"$ID\"" $O/keyed.jsonl)
echo "RESCORE_COUNTS llmprint=$a keyed=$b"; [ "$a" = 1 ] && [ "$b" = 1 ] || exit 1
echo RESCORE_DONE
