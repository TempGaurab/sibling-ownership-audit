#!/usr/bin/env python3
"""Pilot family only, never reported: runs the audit rules end to end on extracted features, to check the pipeline.
Usage: pilot_check.py OUTDIR"""
import collections, glob, json, os, re, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rules import nikolic, mps, rank_p
O = sys.argv[1]; man = {e["id"].replace("/", "__"): e for e in json.load(open(f"{O}/manifest.json"))}
nt, gr = {}, {}
for k in man:
    f = f"{O}/bb/{k}/nexttoken.npy"
    if os.path.exists(f): nt[k], gr[k] = np.load(f), np.load(f"{O}/bb/{k}/gram.npy")
wb = {os.path.basename(f)[:-5]: json.load(open(f))["scores"] for f in glob.glob(f"{O}/wb/*.json")}
CL = sorted(k for k in man if man[k]["kind"] == "claimant"); CTRL = [k for k in nt if man[k]["kind"] == "control"]
print("features", len(nt), "of", len(man), "whitebox", len(wb), "controls", len(CTRL))
def cka(a, b): return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
def group(k, c):
    e = man[k]
    if e["kind"] == "sib":
        return ("high" if "_high_" in k else "mid") if k.startswith(c) else "low"
    if e["kind"] == "copy":
        return ("copy_" + k.split("_")[2]) if k.startswith(c) else None
    return e["kind"] if e["kind"] in ("control", "base") else None
for c in CL:
    g = collections.defaultdict(list)
    for k in nt:
        if k == c or group(k, c) is None: continue
        s = dict(agree=float((nt[k][:5000] == nt[c][:5000]).mean()), cka=cka(gr[k], gr[c]))
        if k in wb and c in wb[k]: s.update(wb[k][c])
        g[group(k, c)].append((k, s))
    print("==", c)
    for name, rows in sorted(g.items()):
        cols = sorted({x for _, s in rows for x in s})
        print(f"{name:14s} n={len(rows):3d} " + " ".join(f"{x}={np.nanmean([s[x] for _, s in rows if x in s]):.3f}" for x in cols))
    P1 = {k: nt[k] for k in [c] + CTRL}; P2 = dict(P1, base=nt["base"]) if "base" in nt else P1
    for name, rows in sorted(g.items()):
        if name in ("control", "base"): continue
        fn = [nikolic(nt[k], {x: v for x, v in P1.items() if x != k}) == c for k, _ in rows]
        f2 = [nikolic(nt[k], {x: v for x, v in P2.items() if x != k}) == c for k, _ in rows]
        m1 = [c in mps(nt[k], {x: v for x, v in P1.items() if x != k}, R=200, n_prompts=5000)[0] for k, _ in rows[:4]]
        ind = [s["agree"] for k, s in g.get("control", [])]
        q = [rank_p(ind, s["agree"]) <= 0.2 for _, s in rows] if ind else []
        print(f"  {name:14s} names claimant: nikolic P1 {np.mean(fn):.2f} P2 {np.mean(f2):.2f}; mps P1 {np.mean(m1):.2f} (first 4); agree above controls at coarse p<=0.2: {np.mean(q) if q else float('nan'):.2f}")
print("PILOT_CHECK_DONE")
