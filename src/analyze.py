#!/usr/bin/env python3
"""Confirmatory analysis under the frozen rules in prereg/hypotheses.md. Written and tested on synthetic data before any
confirmatory score existed. Input: a long table of scores, one row per (family, claimant, suspect, score), larger meaning
more like a copy. Columns: family claimant suspect score value kind stratum dataset recipe publisher month
kind: sib (controlled), hub, control, copy. stratum: high mid low (controlled); shared disjoint unknown (hub); for copies the
transformation and level, as in ft_full_20. recipe: for controlled siblings, the recipe cell, as in full@2e-05 or lora8.
Within-population leave-one-out is never used. Reference and evaluation sets never share a cluster.
Usage: analyze.py SCORES.csv OUT.json"""
import collections, json, sys, numpy as np, pandas as pd
ALPHA, B, SEED = 0.05, 2000, 2027

def rank_p(ref, t):
    """Vectorized conformal rank p-value; ties count against rejection."""
    ref = np.sort(np.asarray(ref, float)); t = np.asarray(t, float)
    return (1 + len(ref) - np.searchsorted(ref, t, side="left")) / (len(ref) + 1)

def boot(rows, stat=np.mean, b=B, seed=SEED):
    """Percentile interval, resampling clusters. rows: DataFrame with columns x and cluster. Returns estimate, lo, hi, clusters."""
    rows = rows.dropna(subset=["x"])
    if not len(rows): return dict(est=None, lo=None, hi=None, n=0, clusters=0)
    g = rows.groupby("cluster").x.agg(["sum", "count"]).values; rng = np.random.default_rng(seed); k = len(g)
    i = rng.integers(0, k, (b, k)); s, c = g[i, 0].sum(1), g[i, 1].sum(1); v = s / c
    return dict(est=float(rows.x.mean()), lo=float(np.quantile(v, .025)), hi=float(np.quantile(v, .975)), n=int(len(rows)), clusters=int(k), p_le_05=float((1 + (v <= ALPHA).sum()) / (b + 1)))

def holm(ps):
    """Holm adjusted p-values for a dict of name to p."""
    names = sorted(ps, key=ps.get); m = len(names); out = {}; run = 0.0
    for i, n in enumerate(names): run = max(run, min(1.0, (m - i) * ps[n])); out[n] = run
    return out

def clusters(d):
    return np.where(d.kind == "hub", d.publisher.astype(str), d.dataset.astype(str) + "|" + d.recipe.astype(str))

def eval_split(D, fold_col, ref_filter, eval_filter, indep=None):
    """For each fold value: references exclude that cluster, evaluation is only that cluster. Returns rows with x = rejected."""
    out = []
    for (fam, c, sc), d in D.groupby(["family", "claimant", "score"]):
        I = None if indep is None else indep[(indep.family == fam) & (indep.claimant == c) & (indep.score == sc)].value.values
        ev = eval_filter(d)
        for v, e in ev.groupby(fold_col):
            r = ref_filter(d, e); r = r[r[fold_col] != v]
            if len(r) == 0: continue
            rej = rank_p(r.value.values, e.value.values) <= ALPHA
            if I is not None and len(I): rej = rej & (rank_p(I, e.value.values) <= ALPHA)
            out.append(e.assign(x=rej.astype(float), ref_n=len(r)))
    return pd.concat(out) if out else pd.DataFrame(columns=list(D.columns) + ["x", "ref_n"])

def summarize(rows, by=("family", "score")):
    if not len(rows): return {}
    rows = rows.assign(cluster=clusters(rows)); res = {}
    for key, g in rows.groupby(list(by)):
        r = boot(g); r["per_claimant"] = {str(c): float(x.x.mean()) for c, x in g.groupby("claimant")}; r["min_ref_n"] = int(g.ref_n.min()); res["|".join(map(str, key))] = r
    return res

def splits(T):
    """The six preregistered held-out splits for the proposed test: same-stratum sibling reference intersected with independents."""
    ind = T[T.kind == "control"]; S = T[T.kind == "sib"]; H = T[T.kind == "hub"]; R = {}
    same = lambda d, e: d[(d.kind == e.kind.iloc[0]) & (d.stratum == e.stratum.iloc[0])]
    hub_eval = lambda d: d[d.kind == "hub"]
    # S1 leave-one-publisher-out, hub: pooled and disjoint references, each evaluated on its own population.
    R["S1_pooled"] = eval_split(H, "publisher", lambda d, e: d, hub_eval, ind)
    R["S1_disjoint"] = eval_split(H, "publisher", lambda d, e: d[d.stratum == "disjoint"], lambda d: d[d.stratum == "disjoint"], ind)
    # S2 leave-one-recipe-cell-out, controlled, same stratum. S3 leave-one-dataset-out, within mid and low.
    R["S2"] = pd.concat([eval_split(S[S.stratum == st], "recipe", lambda d, e: d, lambda d: d, ind) for st in ("high", "mid", "low")])
    R["S3"] = eval_split(S[S.stratum == "low"], "dataset", lambda d, e: d, lambda d: d, ind)
    # S4 time split: reference before the median upload month, suspects from that month on. Fixed in the preregistration.
    CUT = {"qwen3": "2026-03", "tinyllama": "2025-03"}; rows = []
    for (fam, c, sc), d in H.groupby(["family", "claimant", "score"]):
        cut = CUT[fam] if fam in CUT else sorted(d.month.astype(str))[len(d) // 2]; m = d.month.astype(str); r, e = d[m < cut], d[m >= cut]
        if not (len(r) and len(e)): continue
        I = ind[(ind.family == fam) & (ind.claimant == c) & (ind.score == sc)].value.values; rej = rank_p(r.value.values, e.value.values) <= ALPHA
        rows.append(e.assign(x=(rej & (rank_p(I, e.value.values) <= ALPHA)).astype(float) if len(I) else rej.astype(float), ref_n=len(r)))
    R["S4"] = pd.concat(rows) if rows else pd.DataFrame()
    # S5 hub reference to controlled suspects; S6 controlled reference to hub suspects. Different populations, so no shared cluster.
    for name, a, b in (("S5", H, S), ("S6", S, H)):
        rows = []
        for (fam, c, sc), e in b.groupby(["family", "claimant", "score"]):
            r = a[(a.family == fam) & (a.claimant == c) & (a.score == sc)].value.values
            if not len(r): continue
            I = ind[(ind.family == fam) & (ind.claimant == c) & (ind.score == sc)].value.values; rej = rank_p(r, e.value.values) <= ALPHA
            rows.append(e.assign(x=(rej & (rank_p(I, e.value.values) <= ALPHA)).astype(float) if len(I) else rej.astype(float), ref_n=len(r)))
        R[name] = pd.concat(rows) if rows else pd.DataFrame()
    return {k: summarize(v) for k, v in R.items()}, R

def h1(T):
    """Independent-reference rank test on siblings, per score and family, plus the existence rule with Holm."""
    ind = T[T.kind == "control"]; rows = []
    for (fam, c, sc), d in T[T.kind.isin(["sib", "hub"])].groupby(["family", "claimant", "score"]):
        I = ind[(ind.family == fam) & (ind.claimant == c) & (ind.score == sc)].value.values
        if len(I): rows.append(d.assign(x=(rank_p(I, d.value.values) <= ALPHA).astype(float), ref_n=len(I)))
    return pd.concat(rows) if rows else pd.DataFrame()

def h1_rule(res):
    adj = holm({k: v["p_le_05"] for k, v in res.items()})
    return {k: dict(v, holm_p=adj[k], over_accuses=bool(v["est"] > .10 and v["lo"] > .05 and adj[k] <= ALPHA)) for k, v in res.items()}

def h3(T):
    """Highest-sharing stratum on held-out recipe cells: pooled sibling reference against same-stratum reference."""
    S = T[T.kind == "sib"]; ind = T[T.kind == "control"]; hi = lambda d: d[d.stratum == "high"]
    pooled = eval_split(S, "recipe", lambda d, e: d, hi, ind); strat = eval_split(S[S.stratum == "high"], "recipe", lambda d, e: d, lambda d: d, ind)
    p, s = summarize(pooled), summarize(strat); out = {}
    for k in sorted(set(p) & set(s)): out[k] = dict(pooled=p[k], stratified=s[k], holds=bool(p[k]["est"] >= .10 and s[k]["est"] <= .08))
    return out

def h4(T):
    """Power on declared copies against each reference. (a) the loss from sibling calibration. (b) usefulness on the named copies."""
    C = T[T.kind == "copy"]; ind = T[T.kind == "control"]; S = T[T.kind == "sib"]; out = {}
    for (fam, sc, tr), g in C.groupby(["family", "score", "stratum"]):
        pi, ps = [], []
        for c, e in g.groupby("claimant"):
            I = ind[(ind.family == fam) & (ind.claimant == c) & (ind.score == sc)].value.values
            hi = S[(S.family == fam) & (S.claimant == c) & (S.score == sc) & (S.stratum == "high")].value.values
            if len(I): pi += list(rank_p(I, e.value.values) <= ALPHA)
            if len(hi): ps += list((rank_p(hi, e.value.values) <= ALPHA) & ((rank_p(I, e.value.values) <= ALPHA) if len(I) else True))
        out[f"{fam}|{sc}|{tr}"] = dict(power_independent=float(np.mean(pi)) if pi else None, power_sibling=float(np.mean(ps)) if ps else None, n=len(g))
    a = {k: bool(v["power_independent"] is not None and v["power_sibling"] is not None and v["power_independent"] - v["power_sibling"] > .05) for k, v in out.items()}
    useful = {}
    for fam in C.family.unique():
        for sc in C[C.family == fam].score.unique():
            need = [out.get(f"{fam}|{sc}|{t}", {}).get("power_sibling") for t in ("quant_8", "quant_4", "ft_full_20", "ft_lora16_20")]
            useful[f"{fam}|{sc}"] = bool(all(x is not None and x >= .80 for x in need))
    return dict(grid=out, a_any_score_loses_power=any(a.values()), a=a, b_useful=useful, b_holds=any(useful.values()))

def main():
    T = pd.read_csv(sys.argv[1], dtype={"publisher": str, "month": str, "recipe": str, "dataset": str}); R = {}
    res1 = h1(T); R["H1"] = h1_rule(summarize(res1)); R["H1_by_stratum"] = summarize(res1, ("family", "score", "kind", "stratum")) if len(res1) else {}
    sp, _ = splits(T); R["splits"] = sp
    R["H2"] = {s: {k: dict(v, holds=bool(v["est"] <= .08 and v["hi"] <= .10)) for k, v in sp[s].items()} for s in sp}
    R["H3"] = h3(T); R["H4"] = h4(T)
    R["H6"] = {k: dict(v, ages=bool(v["est"] > .08)) for k, v in sp.get("S4", {}).items()}
    json.dump(R, open(sys.argv[2], "w"), indent=1); print("hypothesis tables written", {k: len(v) for k, v in R.items()})

if __name__ == "__main__":
    main()
