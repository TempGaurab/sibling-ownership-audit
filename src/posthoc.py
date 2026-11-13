#!/usr/bin/env python3
"""Post hoc, exploratory analyses of existing scores, specified in prereg/DEVIATIONS.md (Nov 12, entry 0f417a48700b) before any
output was seen. No rule, threshold or verdict changes, no model is run and no score is recomputed. Aggregate output only: hub
models are never named in the output.
Usage:
  posthoc.py scores SCORES.csv DOWNLOADS.jsonl OUT.json     (cluster: distributions, grouped hub splits, size composition, ties)
  posthoc.py weights SAMPLE1.json [SAMPLE2.json ...] -- WEIGHT_CHECK.jsonl OUT.json   (local: weight-check pass/fail on metadata)"""
import collections, json, os, sys, numpy as np, pandas as pd

ALPHA = 0.05
QS = (0.1, 0.25, 0.5, 0.75, 0.9)


def quarter(m):
    m = str(m)
    return f"{m[:4]}Q{(int(m[5:7]) - 1) // 3 + 1}" if len(m) >= 7 and m[:4].isdigit() else "unknown"


def hub_adapter(downloads):
    """Adapter (LoRA or other PEFT) if the pinned snapshot ships adapter_config.json, else full weights."""
    out = {}
    for line in open(downloads):
        r = json.loads(line)
        if r.get("path") and r.get("status") == "ok":
            out[r["id"]] = "adapter" if os.path.exists(os.path.join(r["path"], "adapter_config.json")) else "full"
    return out


def quant(v):
    v = np.asarray(v, float)
    return dict(n=int(len(v)), **{f"q{int(q * 100)}": float(np.quantile(v, q)) for q in QS}) if len(v) else dict(n=0)


def holds(v): return v.get("est") is not None and v["est"] <= 0.08 and v["hi"] <= 0.10


def scores(path, downloads, out):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from analyze import rank_p, eval_split, summarize, splits
    T = pd.read_csv(path, dtype={"publisher": str, "month": str, "recipe": str, "dataset": str})
    ad = hub_adapter(downloads)
    T["atype"] = np.where(T.kind == "sib", np.where(T.recipe.astype(str).str.startswith("lora"), "adapter", "full"),
                          np.where(T.kind == "hub", T.suspect.map(ad).fillna("unknown"), ""))
    T["quarter"] = np.where(T.kind == "hub", T.month.map(quarter), "")
    R = dict(label="post hoc, exploratory; DEVIATIONS.md Nov 12 entry 0f417a48700b; no verdict depends on it")
    one = T[(T.kind == "hub")].drop_duplicates(["family", "suspect"])
    R["hub_counts"] = {f: dict(adapter_type=dict(collections.Counter(g.atype)), quarter=dict(sorted(collections.Counter(g.quarter).items())))
                       for f, g in one.groupby("family")}
    # 1. Score distributions by recipe cell and adapter type (pooled over claimants; per-claimant medians alongside).
    D = {}
    for (f, sc, rc), g in T[T.kind == "sib"].groupby(["family", "score", "recipe"]):
        D[f"{f}|{sc}|sib|{rc}"] = dict(quant(g.value), per_claimant_median={c: float(x.value.median()) for c, x in g.groupby("claimant")})
    for (f, sc, at), g in T[T.kind.isin(["sib", "hub"])].groupby(["family", "score", "atype"]):
        for kind, h in g.groupby("kind"):
            D[f"{f}|{sc}|{kind}|{at}"] = dict(quant(h.value), per_claimant_median={c: float(x.value.median()) for c, x in h.groupby("claimant")})
    for (f, sc), g in T[T.kind == "control"].groupby(["family", "score"]):
        D[f"{f}|{sc}|control|all"] = quant(g.value)
    R["distributions"] = D
    # S5 and S6 rejections of the proposed test, broken down by recipe cell and adapter type.
    _, rows = splits(T.drop(columns=["atype", "quarter"]))
    B = {}
    for name in ("S5", "S6"):
        x = rows[name].join(T[["atype"]], how="left")
        keys = ["family", "score", "recipe"] if name == "S5" else ["family", "score", "atype"]
        if name == "S5":
            for (f, sc, at), g in x.assign(atype=np.where(x.recipe.astype(str).str.startswith("lora"), "adapter", "full")).groupby(["family", "score", "atype"]):
                B[f"{name}|{f}|{sc}|{at}"] = dict(rate=float(g.x.mean()), n=int(len(g)), per_claimant={c: float(y.x.mean()) for c, y in g.groupby("claimant")})
        for k, g in x.groupby(keys):
            B[f"{name}|" + "|".join(map(str, k))] = dict(rate=float(g.x.mean()), n=int(len(g)), per_claimant={c: float(y.x.mean()) for c, y in g.groupby("claimant")})
    R["cross_by_recipe"] = B
    # 2. Grouped hub splits: S1-pooled test with upload quarter or adapter type held out.
    ind = T[T.kind == "control"].drop(columns=["atype", "quarter"]); H = T[T.kind == "hub"]
    G = {}
    for col in ("quarter", "atype"):
        h = H[H[col] != "unknown"]
        r = eval_split(h, col, lambda d, e: d, lambda d: d, ind)
        s = summarize(r)
        G[col] = {k: dict(v, holds=holds(v)) for k, v in s.items()}
        G[col + "_folds"] = {f: dict(collections.Counter(g.drop_duplicates("suspect")[col])) for f, g in h.groupby("family")}
        G[col + "_by_fold"] = {"|".join(map(str, k)): dict(rate=float(g.x.mean()), n=int(len(g)), min_ref_n=int(g.ref_n.min()))
                               for k, g in r.groupby(["family", "score", col])}
    R["grouped_hub_splits"] = G
    # 3. Composition of the reference available at each size in the Nov 9 ablation (S2 loop of ablations.py).
    C = {}
    S = T[T.kind == "sib"]
    for (f, c, sc), d in S.groupby(["family", "claimant", "score"]):
        if c != sorted(S.claimant.unique())[0] or sc != sorted(S.score.unique())[0]: continue  # composition is the same for every claimant and score
        for st, ds in d.groupby("stratum"):
            for v, e in ds.groupby("recipe"):
                n = int((ds.recipe != v).sum())
                for size in (0, 49, 99, 199):
                    key = f"{f}|size_{size}"; z = C.setdefault(key, dict(cells=0, suspects=0, by_stratum=collections.Counter(), ref_sizes=[]))
                    if size and n < size: continue
                    z["cells"] += 1; z["suspects"] += int(len(e)); z["by_stratum"][st] += int(len(e)); z["ref_sizes"].append(min(n, size) if size else n)
    for z in C.values(): z["by_stratum"] = dict(z["by_stratum"]); z["ref_sizes"] = sorted(set(z["ref_sizes"]))
    R["size_composition"] = C
    # 4. Ties: suspect values equal to some value in the independent reference, or in the same-stratum sibling reference.
    Z = {}
    for (f, sc), g in T.groupby(["family", "score"]):
        sus = g[g.kind.isin(["sib", "hub"])]; tied_i = tied_s = 0
        for (c, kind, st), e in sus.groupby(["claimant", "kind", "stratum"]):
            I = set(g[(g.kind == "control") & (g.claimant == c)].value.values)
            ref = g[(g.claimant == c) & (g.kind == kind) & (g.stratum == st)].value.values
            cnt = collections.Counter(ref)
            tied_i += int(np.isin(e.value.values, list(I)).sum()) if I else 0
            tied_s += int(sum(cnt[v] > 1 for v in e.value.values))
        Z[f"{f}|{sc}"] = dict(suspects=int(len(sus)), tied_with_independent=tied_i / len(sus), tied_within_sibling_reference=tied_s / len(sus),
                              distinct_values=int(g.value.nunique()), rows=int(len(g)))
    R["ties"] = Z
    json.dump(R, open(out, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("POSTHOC_DONE", out)


def weights(samples, wc, out):
    meta = {}
    for p in samples:
        fam = "qwen3" if "Qwen3" in p else "tinyllama" if "TinyLlama" in p else os.path.basename(p)
        for r in json.load(open(p)): meta[r["id"]] = dict(r, family=fam)
    pubsize = collections.Counter(r["author"] for r in meta.values())
    last = {}
    for line in open(wc):
        r = json.loads(line); last[r["id"]] = r
    rows = []
    for i, r in last.items():
        m = meta.get(i)
        if not m: continue
        rows.append(dict(family=m["family"], passed=bool(r.get("passed")), reason=r.get("reason") or "passed", format=m.get("format"), quarter=quarter(m.get("created", "")[:7]),
                         dataset_tag=bool(m.get("datasets")), multi_model_publisher=pubsize[m["author"]] > 1, licence=str(m.get("license"))))
    df = pd.DataFrame(rows); R = dict(label="post hoc; metadata only, no score", n=int(len(df)), passed=int(df.passed.sum()))
    for col in ("family", "format", "quarter", "dataset_tag", "multi_model_publisher", "licence"):
        R[col] = {str(k): dict(passed=int(g.passed.sum()), failed=int((~g.passed).sum())) for k, g in df.groupby(col)}
    R["fail_reasons"] = dict(collections.Counter(df[~df.passed].reason))
    json.dump(R, open(out, "w"), indent=1); print("WEIGHTS_DONE", R["n"], R["passed"])


if __name__ == "__main__":
    if sys.argv[1] == "scores": scores(*sys.argv[2:5])
    elif sys.argv[1] == "weights":
        k = sys.argv.index("--"); weights(sys.argv[2:k], sys.argv[k + 1], sys.argv[k + 2])
