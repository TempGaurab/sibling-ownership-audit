#!/bin/bash
#SBATCH --account=proj-606
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/baralgb/proj606/logs/%x-%j.out
# Nov 12 deviation (DEVIATIONS.md, keyed generation inputs): rerun keyed scoring for the one TinyLlama hub sibling whose tokenizer
# returns token_type_ids, which generate() rejects. score_keyed.py is idempotent and appends to the existing file.
# Then runs the post-scoring check on both families. Exits non-zero unless the model has exactly one keyed record and both
# checks print CHECK_OK, so an analysis job submitted with afterok on this job runs only on complete, checked outputs.
R=/home/baralgb/proj606; PY=$R/.venv/bin/python; C=$R/code; FAM=tinyllama; O=$R/outputs/main/features/$FAM
export HF_HOME=$R/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false; cd $C
ID=zrowt/results; SUB=$O/manifest_tokentype_nov12.json
$PY - "$O/manifest.json" "$SUB" "$ID" <<'PY'
import json, sys
m = [e for e in json.load(open(sys.argv[1])) if e["id"] == sys.argv[3]]
assert len(m) == 1, "model not in manifest"
json.dump(m, open(sys.argv[2], "w")); print("SUBSET", len(m))
PY
[ $? = 0 ] || exit 1
$PY score_keyed.py $SUB $R/keys/$FAM $O/keyed.jsonl 2>&1 | grep -E 'FAIL|Traceback' | cut -c1-200
b=$(grep -c "\"id\": \"$ID\"" $O/keyed.jsonl); echo "RESCORE_COUNTS keyed=$b"; [ "$b" = 1 ] || exit 1
for spec in qwen3:score-qwen3-3795325 tinyllama:score-tinyllama-3796032; do
  f=${spec%%:*}; r=$($PY -I post_score_check.py $R/outputs/main/features/$f $R/logs/${spec#*:}.out | tail -n 1)
  echo "CHECK $f $r"; [ "$r" = CHECK_OK ] || exit 1
done
echo RESCORE_DONE
