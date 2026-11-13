#!/bin/bash
#SBATCH --account=proj-606
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=test-rules
#SBATCH --output=/home/baralgb/proj606/logs/%x-%j.out
# Synthetic checks of the rule code, including the GPU-batched MPS. No model is involved.
R=/home/baralgb/proj606; cd $R; mkdir -p t/src; cp code/rules.py code/test_rules.py code/analyze.py code/test_analyze.py t/src/; cd t
$R/.venv/bin/python src/test_rules.py 2>&1 | tail -12 | cut -c1-200
$R/.venv/bin/python src/test_analyze.py 2>&1 | tail -6 | cut -c1-200
cd $R; rm -rf t; echo TEST_DONE
