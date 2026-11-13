#!/bin/bash
#SBATCH --account=proj-606
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/baralgb/proj606/logs/%x-%j.out
# Nov 10 deviation (DEVIATIONS.md, tokenizer loading): re-extract black-box features for the four TinyLlama hub siblings
# whose SentencePiece tokenizers failed before sentencepiece/protobuf were installed. Extraction code is unchanged.
# The old error.json is kept as error_superseded_nov10.json. Exits non-zero unless all four now have meta.json.
R=/home/baralgb/proj606; PY=$R/.venv/bin/python; C=$R/code; O=$R/outputs/main/features/tinyllama
export HF_HOME=$R/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false; cd $C
IDS="alexredna/TinyLlama-1.1B-Chat-v1.0-reasoning-v2 alexredna/Tukan-1.1B-Chat-reasoning-sft-COLA-2.5epr-R8-A16 legionlm/orcamath-tinyllama-1.1B-v1-chat mstyslavity/5tiny"
SUB=$O/manifest_reextract_nov10.json
$PY - "$O/manifest.json" "$SUB" $IDS <<'PY'
import json, sys
man, out, ids = sys.argv[1], sys.argv[2], sys.argv[3:]
m = [e for e in json.load(open(man)) if e["id"] in ids]
assert sorted(e["id"] for e in m) == sorted(ids), "not all four ids are in the manifest"
json.dump(m, open(out, "w")); print("SUBSET", len(m))
PY
[ $? = 0 ] || exit 1
for i in $IDS; do d=$O/bb/${i//\//__}; [ -f $d/error.json ] && mv $d/error.json $d/error_superseded_nov10.json; done
$PY extract_blackbox.py $SUB $R/data/prompts.json $R/data/reef.json $O/bb 0 1 2>&1 | grep -E '^ok|^FAIL|Traceback' | cut -c1-200
n=0; for i in $IDS; do [ -f $O/bb/${i//\//__}/meta.json ] && n=$((n+1)); done
echo "REEXTRACT_OK $n/4"; [ $n = 4 ] || exit 1
echo REEXTRACT_DONE
