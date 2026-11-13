#!/bin/bash
#SBATCH --account=proj-606
#SBATCH --partition=gpu-h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/baralgb/proj606/logs/%x-%j.out
# Confirmatory analysis under the frozen rules, run once both families are scored and pass the post-scoring check:
# analyze.py on both families' scores concatenated, apply_rules.py (original rules, H1 and H5), and the exploratory ablations.
# CPU only: these scripts use numpy and pandas, so no GPU is requested. Outputs are aggregate tables only.
# Refuses to run unless each script matches its hash in the last stamped manifest (prereg/timestamps/manifest_2026-11-09b.txt).
# Usage: sbatch --job-name=analyze-main main_analyze.sbatch CODE_COMMIT
R=/home/baralgb/proj606; PY=$R/.venv/bin/python; C=$R/code; F=$R/outputs/main/features; A=$R/outputs/main/analysis; cd $C
sha256sum -c --quiet <<'SUMS' || { echo HASH_MISMATCH; exit 1; }
37dd69bc44446b79b93d3d91692384c452c38cba46765ba385cdd841ac6af84c  analyze.py
2cd352d2540c99aaac70d41d9b83ced0a0ddc4ccf2a7e1f03715127a8ff3e7f1  apply_rules.py
13828e37319aa3a29049bb0b5b44ca163d7e4d3c47a4c2beaccc4caf2741a0ba  ablations.py
6957d6f5b4013e06dabf6690009eeb5ed47a195604f5abb53ecb0b1406950997  rules.py
SUMS
for f in qwen3 tinyllama; do grep -q SCORE_DONE $R/logs/score-$f-*.out 2>/dev/null || { echo "NOT_SCORED $f"; exit 1; }; done
mkdir -p $A
{ echo "code_commit ${1:-unknown}"; date -Is; sha256sum analyze.py apply_rules.py ablations.py rules.py $F/*/scores.csv $F/*/rules.jsonl $F/*/llmprint.jsonl $F/*/keyed.jsonl; } > $A/provenance.txt
{ head -n 1 $F/qwen3/scores.csv; tail -q -n +2 $F/qwen3/scores.csv $F/tinyllama/scores.csv; } > $A/scores_all.csv
for step in "analyze.py $A/scores_all.csv $A/analysis.json" "apply_rules.py $A/rules.json qwen3:$F/qwen3 tinyllama:$F/tinyllama" "ablations.py $A/scores_all.csv $A/ablations.json"; do
  set -- $step; n=${1%.py}
  $PY "$@" > $A/$n.log 2>&1; s=$?
  tail -n 3 $A/$n.log | cut -c1-300
  [ $s = 0 ] || { echo "STEP_FAILED $n status=$s"; exit 1; }
done
sha256sum $A/analysis.json $A/rules.json $A/ablations.json >> $A/provenance.txt
echo ANALYZE_DONE
