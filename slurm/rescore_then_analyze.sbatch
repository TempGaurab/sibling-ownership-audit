#!/bin/bash
#SBATCH --account=proj-606
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/baralgb/proj606/logs/%x-%j.out
# Runs rescore_keyed.sbatch and then, only if it succeeds (one keyed record for the model, CHECK_OK for both families),
# main_analyze.sbatch in the same allocation. Combined on Nov 12 because the shared node had no allocatable CPUs: a separate
# CPU-only analysis job would have queued behind other jobs of the account. Both scripts are unchanged and run with bash.
# Usage: sbatch --job-name=rescore-analyze rescore_then_analyze.sbatch CODE_COMMIT
cd /home/baralgb/proj606/code
bash rescore_keyed.sbatch || { echo RESCORE_FAILED; exit 1; }
bash main_analyze.sbatch "$1"
