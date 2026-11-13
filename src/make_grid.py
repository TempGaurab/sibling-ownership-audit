#!/usr/bin/env python3
"""Write one training command per line. Idempotent downstream: train_model skips finished runs.
Usage: make_grid.py FAMILY BASE REV R N_PER_CELL [--copies-only]
Each claimant has its own dataset. high: same dataset, sibling pool, same recipe family as the claimant.
mid: same dataset, a different recipe family. Low is reused from other claimants' siblings."""
import sys
fam, base, rev, R, n = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
copies_only = "--copies-only" in sys.argv
STEPS = 2000
CL = [("ultrachat", "full", 2e-5), ("oasst1", "lora16", 1e-4), ("gsm8k", "full", 5e-5)]
OTHER = {"full": [("lora8", 1e-4), ("lora16", 1e-4)], "lora16": [("full", 2e-5), ("full", 5e-5)]}
SAME = {"full": [("full", 2e-5), ("full", 5e-5)], "lora16": [("lora8", 1e-4), ("lora16", 1e-4)]}
py = f"{R}/.venv/bin/python {R}/code/train_model.py"
out = f"{R}/models/{fam}"
b = f"--base {base} --revision {rev}"
for c, (ds, rec, lr) in enumerate(CL):
    cid = f"c{c}_{ds}"
    if not copies_only:
        print(f"{py} {b} --dataset {ds} --pool claimant --recipe {rec} --lr {lr} --steps {STEPS} --seed {9000+c} --out {out}/claimant/{cid}")
        for i in range(n):
            r, l = SAME[rec][i % 2]
            print(f"{py} {b} --dataset {ds} --pool sibling --recipe {r} --lr {l} --steps {STEPS} --seed {10000+100*c+i} --out {out}/sib/{cid}_high_{i}")
            r, l = OTHER[rec][i % 2]
            print(f"{py} {b} --dataset {ds} --pool sibling --recipe {r} --lr {l} --steps {STEPS} --seed {20000+100*c+i} --out {out}/sib/{cid}_mid_{i}")
