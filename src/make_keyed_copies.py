#!/usr/bin/env python3
"""Copy-building commands for the keyed claimants of one family (Nov 12 deviation 6c40f1e0353f, post hoc). The grid, seeds,
attacker datasets, merge partners and generation seeds are those of src/make_copies.py, so each keyed copy differs from its
plain counterpart only in the starting model. Outputs go to models/FAM/{kroot,kgen,kcopy}; nothing existing is touched.
Usage: make_keyed_copies.py FAMILY BASE REV R N_PER_CELL N_SIB PREP TASKS [GEN_N]"""
import sys
fam, base, rev, R, n, nsib, prep, tasks = sys.argv[1:9]; n, nsib = int(n), int(nsib)
GEN = int(sys.argv[9]) if len(sys.argv) > 9 else 10000
CL = ["c0_ultrachat", "c1_oasst1", "c2_gsm8k"]
ATT = {"c0_ultrachat": "oasst1", "c1_oasst1": "ultrachat", "c2_gsm8k": "ultrachat"}
py = f"{R}/.venv/bin/python"; tr = f"{py} {R}/code/transform.py"; tm = f"{py} {R}/code/train_model.py"; ts = f"{py} {R}/code/train_student.py"
M = f"{R}/models/{fam}"; b = f"--base {base} --revision {rev}"; P, T = [], []
for c, cid in enumerate(CL):
    cl = f"{M}/keyed/{cid}"; root = f"{M}/kroot/{cid}"; ds = ATT[cid]; out = f"{M}/kcopy/{cid}"; other = CL[(c + 1) % 3]
    P.append(f"{tr} prep --claimant {cl} {b} --out {root}")
    P.append(f"{tr} generate --claimant {cl} {b} --dataset {ds} --n {GEN} --seed {40000 + c} --out {M}/kgen/{cid}")
    for steps in (20, 200, 2000):
        for rec, lr in (("full", 2e-5), ("lora16", 1e-4)):
            for i in range(n):
                T.append(f"{tm} --base {root} --dataset {ds} --pool all --recipe {rec} --lr {lr} --steps {steps} --seed {50000 + 1000 * c + steps + i + (500 if rec != 'full' else 0)} --out {out}_ft_{rec}_{steps}_{i}")
    for w in (0.9, 0.7, 0.5, 0.3):
        for i in range(n):
            T.append(f"{tr} merge --claimant {cl} --other {M}/sib/{other}_high_{i % nsib} --w {w} {b} --out {out}_merge_{int(w * 100)}_{i}")
    for k in (1000, 10000):
        for i in range(n):
            T.append(f"{ts} {b} --dataset {M}/kgen/{cid}/outputs.jsonl --pool all --recipe full --lr 2e-05 --steps 2000 --n-train {k} --seed {60000 + 1000 * c + k + i} --out {out}_student_{k}_{i}")
    for bits in (8, 4):
        T.append(f"{tr} quant --claimant {cl} --bits {bits} {b} --out {out}_quant_{bits}")
    for f in (0.1, 0.3, 0.5):
        T.append(f"{tr} prune --claimant {cl} --frac {f} {b} --out {out}_prune_{int(f * 100)}")
open(prep, "w").write("\n".join(P) + "\n"); open(tasks, "w").write("\n".join(T) + "\n")
print(len(P), "prep;", len(T), "keyed copies")
