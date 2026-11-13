#!/usr/bin/env python3
"""Exploratory ablations of the sibling reference, labelled as such; no hypothesis depends on them. Written and tested on synthetic
data before any confirmatory score existed. Each changes only the reference, on the controlled siblings under the held-out recipe-cell
split (S2 of analyze.py): false-accusation rate per family and score, with the proposed test (same-stratum reference intersected with
independents) and with the sibling reference alone, and power on declared copies against the high-stratum reference.
Grid: reference size 49, 99, 199; claimant copies placed in the reference at 1, 5, 10%; stratum label noise 5, 10, 20% (reference
members swapped for other-stratum siblings); score-based removal of the top 5 or 10% of the reference (labelled sensitivity analysis,
never used in a primary result). 20 seeded repeats per setting. A size larger than the reference available is skipped, not padded.
Also reports effective reference size as publisher clusters. Usage: ablations.py SCORES.csv OUT.json"""
import json, math, sys, numpy as np, pandas as pd
from analyze import rank_p, summarize, ALPHA
SEED, REPEATS = 2027, 20
GRID = [("base", 0)] + [("size", n) for n in (49, 99, 199)] + [("contam", f) for f in (.01, .05, .10)] + [("noise", f) for f in (.05, .10, .20)] + [("filter", f) for f in (.05, .10)]
H4B = ("quant_8", "quant_4", "ft_full_20", "ft_lora16_20")

def perturb(r, cfg, rng, alt, cop):
    """Return the changed reference, or None when the setting cannot be applied."""
    kind, v = cfg; r = np.asarray(r, float)
    if kind == "base": return r
    if kind == "size": return None if len(r) < v else rng.choice(r, v, replace=False)
    if kind == "filter": return r[r <= np.quantile(r, 1 - v)]
    pool = cop if kind == "contam" else alt
    if pool is None or not len(pool): return None
    k = math.ceil(v * len(r)); out = r.copy(); out[rng.choice(len(r), k, replace=False)] = rng.choice(pool, k, replace=len(pool) < k)
    return out

def run(T, cfg):
    ind = T[T.kind == "control"]; S = T[T.kind == "sib"]; C = T[T.kind == "copy"]; rows = []; power = []
    rng = np.random.default_rng([SEED, sum(map(ord, cfg[0])), int(round(cfg[1] * 100))])
    for (fam, c, sc), d in S.groupby(["family", "claimant", "score"]):
        I = ind[(ind.family == fam) & (ind.claimant == c) & (ind.score == sc)].value.values
        cop = C[(C.family == fam) & (C.claimant == c) & (C.score == sc)]
        for st, ds in d.groupby("stratum"):
            for v, e in ds.groupby("recipe"):
                r = ds[ds.recipe != v].value.values; alt = d[(d.stratum != st) & (d.recipe != v)].value.values
                if not len(r): continue
                a = np.zeros(len(e)); b = np.zeros(len(e)); m = 0; sizes = []
                for _ in range(1 if cfg[0] == "base" else REPEATS):
                    q = perturb(r, cfg, rng, alt, cop.value.values)
                    if q is None or not len(q): continue
                    p = rank_p(q, e.value.values) <= ALPHA; b += p; a += p & (rank_p(I, e.value.values) <= ALPHA) if len(I) else p; m += 1; sizes.append(len(q))
                if m: rows.append(e.assign(x=a / m, x_sib=b / m, ref_n=int(min(sizes))))
        hi = d[d.stratum == "high"].value.values
        if not (len(hi) and len(cop)): continue
        for tr, g in cop.groupby("stratum"):
            hit = []
            for _ in range(1 if cfg[0] == "base" else REPEATS):
                placed = cop.iloc[rng.permutation(len(cop))[:len(cop) // 2]]  # copies placed in the reference are never evaluated
                q = perturb(hi, cfg, rng, d[d.stratum != "high"].value.values, placed.value.values)
                if q is None or not len(q): continue
                ev = g.value.values if cfg[0] != "contam" else g[~g.index.isin(placed.index)].value.values
                if not len(ev): continue
                hit.append(np.mean((rank_p(q, ev) <= ALPHA) & ((rank_p(I, ev) <= ALPHA) if len(I) else True)))
            if hit: power.append(dict(family=fam, claimant=c, score=sc, transform=tr, power=float(np.mean(hit))))
    R = pd.concat(rows) if rows else pd.DataFrame(); P = pd.DataFrame(power); out = {}
    if len(R):
        out["far"] = summarize(R); out["far_by_stratum"] = summarize(R, ("family", "score", "stratum")); out["far_sibling_only"] = summarize(R.assign(x=R.x_sib))
    if len(P):
        pw = lambda d: {"|".join(k): dict(mean=float(g.power.mean()), per_claimant={c: float(x.power.mean()) for c, x in g.groupby("claimant")}) for k, g in d.groupby(["family", "score"])}
        out["power_all_copies"] = pw(P); out["power_h4b_copies"] = pw(P[P["transform"].isin(H4B)]) if P["transform"].isin(H4B).any() else {}
    return out

def effective_n(T):
    """Reference sizes against independent publisher clusters, the effective sample size for hub populations."""
    out = {}
    for fam, d in T.groupby("family"):
        one = d[d.claimant == d.claimant.iloc[0]]; one = one[one.score == one.score.iloc[0]]; h, c = one[one.kind == "hub"], one[one.kind == "control"]
        out[fam] = dict(hub_models=int(len(h)), hub_publishers=int(h.publisher.nunique()), control_models=int(len(c)), control_publishers=int(c.publisher.nunique()))
    return out

def main():
    T = pd.read_csv(sys.argv[1], dtype={"publisher": str, "month": str, "recipe": str, "dataset": str})
    R = dict(label="exploratory ablations; no hypothesis depends on them", effective_n=effective_n(T), grid={f"{k}_{v:g}": run(T, (k, v)) for k, v in GRID})
    json.dump(R, open(sys.argv[2], "w"), indent=1); print("ablations written", list(R["grid"]))

if __name__ == "__main__":
    main()
