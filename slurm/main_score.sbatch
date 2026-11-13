#!/bin/bash
#SBATCH --account=proj-606
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/baralgb/proj606/logs/%x-%j.out
# Confirmatory scoring for one family after extraction: LLMPrint verification, keyed scoring, audit. Submit only after the new manifest is stamped.
# Usage: sbatch --job-name=score-FAM main_score.sbatch FAMILY. Idempotent where the scripts are.
R=/home/baralgb/proj606; PY=$R/.venv/bin/python; C=$R/code; FAM=$1; O=$R/outputs/main/features/$FAM; L=$R/outputs/main/llmprint/$FAM
export HF_HOME=$R/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false; cd $C
SAMPLE=$R/data/census_sample_$FAM.json
$PY make_meta.py $O/manifest.json $O/meta.json $SAMPLE
for c in c0_ultrachat c1_oasst1 c2_gsm8k; do cat $L/parts/$c.part*.jsonl | sort -t: -k2 -n > $L/$c.jsonl; [ "$(wc -l < $L/$c.jsonl)" = 300 ] || echo "WARNING $c has $(wc -l < $L/$c.jsonl) pairs"; done
FP=$(ls $L/*.jsonl | tr '\n' ',' | sed 's/,$//')
$PY llmprint.py verify $O/manifest.json "$FP" $O/llmprint.jsonl 2>&1 | grep -E '^FAIL|Traceback' | cut -c1-200
$PY - <<PY
import json,os
m=json.load(open("$O/manifest.json"))
m+=[dict(id="keyed_"+c,kind="keyed",path="$R/models/$FAM/keyed/"+c,base="$R/models/$FAM/root/"+c,base_revision=None,adapter=False) for c in ("c0_ultrachat","c1_oasst1","c2_gsm8k")]
json.dump(m,open("$O/keyed_manifest.json","w"))
PY
$PY score_keyed.py $O/keyed_manifest.json $R/keys/$FAM $O/keyed.jsonl 2>&1 | grep -E 'FAIL|Traceback' | cut -c1-200
$PY audit.py $FAM $O $O/manifest.json $O/claimants.json $R/data/strata.json $O/meta.json $O/scores.csv $O/rules.jsonl 2>&1 | grep -E 'AUDIT_DONE|Traceback|Error' | cut -c1-300
echo SCORE_DONE
