#!/usr/bin/env python3
"""Append keyed-fingerprint training commands, one per claimant, to a family's copy TASKS file. Settings are those that ran on the
pilot family: continue from the claimant root on the owner's data plus 20 secret pairs, full-weight, lr 2e-5, 2,000 steps. Keys stay in R/keys/FAMILY, outside the repo.
Usage: make_keyed.py FAMILY R TASKS"""
import sys
fam, R, tasks = sys.argv[1:4]; M = f"{R}/models/{fam}"; L = []
for c, (cid, ds) in enumerate((("c0_ultrachat", "ultrachat"), ("c1_oasst1", "oasst1"), ("c2_gsm8k", "gsm8k"))):
    L.append(f"{R}/.venv/bin/python {R}/code/train_keyed.py --keys {R}/keys/{fam}/{cid}.json --base {M}/root/{cid} --dataset {ds} --pool all --recipe full --lr 2e-05 --steps 2000 --seed {70000 + c} --out {M}/keyed/{cid}")
open(tasks, "a").write("\n".join(L) + "\n"); print(len(L), "keyed tasks appended")
