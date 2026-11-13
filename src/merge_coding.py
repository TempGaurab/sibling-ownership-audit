#!/usr/bin/env python3
"""Merge the two blind passes. Reports Cohen's kappa per field as LLM repeatability (not human or independent agreement).
Disagreements go to unknown. Expands to every sampled model. Writes data/strata/strata.json and agreement.txt. Reads no score."""
import collections, glob, json, numpy as np
def load(p):
    return {r["key"]: r for f in sorted(glob.glob(f"data/strata/pass{p}/*.jsonl")) for r in map(json.loads, open(f))}
a, b = load(1), load(2); keys = sorted(set(a) & set(b)); groups = json.load(open("data/strata/groups.json"))
assert set(a) == set(b) == set(groups), (len(a), len(b), len(groups))
def kappa(x, y):
    lab = sorted(set(x) | set(y)); x, y = np.array(x), np.array(y); po = (x == y).mean()
    pe = sum((x == l).mean() * (y == l).mean() for l in lab)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan"), po
lines = []; w = {g: [m for m in groups[g]] for g in groups}
def rep(name, f):
    xs, ys = [f(a[k]) for k in keys], [f(b[k]) for k in keys]
    k, po = kappa(xs, ys); lines.append(f"{name}: kappa {k:.3f}, raw agreement {po:.3f}, disagreements {sum(p != q for p, q in zip(xs, ys))} of {len(keys)}"); return xs, ys
for ds in ("ultrachat", "oasst1", "gsm8k"): rep(f"data_overlap[{ds}]", lambda r: r["overlap"][ds])
pooled = [(a[k]["overlap"][d], b[k]["overlap"][d]) for k in keys for d in ("ultrachat", "oasst1", "gsm8k")]
k, po = kappa(*map(list, zip(*pooled))); lines.append(f"data_overlap pooled: kappa {k:.3f}, raw agreement {po:.3f}")
stated = lambda r: "unknown" if set(r["overlap"].values()) == {"unknown"} else "stated"
rep("dataset stated at all", stated); rep("recipe_family", lambda r: r["recipe_family"]); rep("teacher_use", lambda r: r["teacher_use"])
out = {}; tally = collections.Counter(); ds_t = {d: collections.Counter() for d in ("ultrachat", "oasst1", "gsm8k")}
for k in keys:
    o = {d: (a[k]["overlap"][d] if a[k]["overlap"][d] == b[k]["overlap"][d] else "unknown") for d in ds_t}
    rf = a[k]["recipe_family"] if a[k]["recipe_family"] == b[k]["recipe_family"] else "unknown"
    tu = a[k]["teacher_use"] if a[k]["teacher_use"] == b[k]["teacher_use"] else "unknown"
    for m in groups[k]:
        out[m] = dict(overlap=o, recipe_family=rf, teacher_use=tu, evidence=a[k]["evidence"])
        for d in ds_t: ds_t[d][o[d]] += 1
        tally[rf] += 1
json.dump(out, open("data/strata/strata.json", "w"), indent=0)
lines += [f"final, models={len(out)}: {d} {dict(c)}" for d, c in ds_t.items()] + [f"recipe {dict(tally)}"]
open("data/strata/agreement.txt", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
