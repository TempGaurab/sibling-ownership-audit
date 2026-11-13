"""Synthetic checks of the analysis code, with known answers. No model is involved. Numbers here are simulated and never reported."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "src")
from analyze import rank_p, boot, holm, splits, h1, h1_rule, h3, h4, summarize, ALPHA
assert rank_p([1, 2, 3], 4) == .25 and rank_p([1, 2, 3], 3) == .5 and rank_p([4, 4], 4) == 1.0
assert np.allclose(rank_p([1, 2, 3], [4, 3, 0]), [.25, .5, 1.0])
a = holm(dict(a=.01, b=.04, c=.03)); assert abs(a["a"] - .03) < 1e-9 and abs(a["c"] - .06) < 1e-9 and abs(a["b"] - .06) < 1e-9
rng = np.random.default_rng(0)
def make(sib_shift, high_shift, copy_shift=8.0, n=120):
    rows = []
    for c in ("c0", "c1", "c2"):
        def add(kind, st, v, ds, rec, pub=None, mon="2026-01"):
            rows.append(dict(family="qwen3", claimant=c, suspect=f"m{len(rows)}", score="s", value=v, kind=kind, stratum=st, dataset=ds, recipe=rec, publisher=pub, month=mon))
        for i in range(400): add("control", "ind", rng.normal(), "x", "x", f"p{i}")
        for st, sh in (("high", sib_shift + high_shift), ("mid", sib_shift), ("low", sib_shift)):
            for ds in ("d0", "d1", "d2"):
                for rec in ("full@2e-05", "full@5e-05", "lora8", "lora16"):
                    for i in range(n // 12): add("sib", st, rng.normal(sib_shift + (high_shift if st == "high" else 0)), ds, rec)
        for i in range(300): add("hub", "disjoint", rng.normal(sib_shift), "h", "h", f"h{i % 90}", "2026-01" if i % 2 else "2026-06")
        for t in ("quant_8", "quant_4", "ft_full_20", "ft_lora16_20"):
            for i in range(5): add("copy", t, rng.normal(copy_shift), "c", "c")
    return pd.DataFrame(rows)
# 1. Siblings exchangeable with each other and shifted from independents: independent rule over-accuses, sibling rule holds.
T = make(3.0, 0.0); r = h1_rule(summarize(h1(T))); e = r["qwen3|s"]; print("H1 FAR, shifted siblings", round(e["est"], 3), e["over_accuses"]); assert e["est"] > .5 and e["over_accuses"]
sp, _ = splits(T)
for k in ("S1_pooled", "S2", "S3", "S4", "S5", "S6"):
    v = sp[k]["qwen3|s"]; print(k, "FAR", round(v["est"], 3), "clusters", v["clusters"], "min ref", v["min_ref_n"]); assert v["est"] < .13, k
# 2. Siblings the same as independents: the independent rule stays calibrated, and no over-accusation is claimed.
T0 = make(0.0, 0.0); e0 = h1_rule(summarize(h1(T0)))["qwen3|s"]; print("H1 FAR, no sibling shift", round(e0["est"], 3), e0["over_accuses"]); assert not e0["over_accuses"] and e0["est"] < .09
# 3. A shifted high stratum: pooling fails inside it and stratifying holds.
T3 = make(2.0, 3.0); h = h3(T3)["qwen3|s"]; print("H3 pooled", round(h["pooled"]["est"], 3), "stratified", round(h["stratified"]["est"], 3), h["holds"]); assert h["holds"]
# 4. Power: strong copies are detected; weak copies are not, and the rule says so.
p = h4(make(3.0, 0.0, 12.0)); assert p["b_holds"]; w = h4(make(3.0, 0.0, 3.2)); print("H4 weak copies useful:", w["b_holds"], "loses power:", w["a_any_score_loses_power"]); assert not w["b_holds"] and w["a_any_score_loses_power"]
# 5. Held-out folds never share a cluster: a leak would make S1 vacuous at exactly floor(alpha N)/N, so check on a dependent population.
T5 = make(3.0, 0.0); T5.loc[(T5.kind == "hub") & (T5.publisher == "h0"), "value"] += 50
v = splits(T5)[0]["S1_pooled"]["qwen3|s"]; print("S1 with one outlying publisher", round(v["est"], 3)); assert v["est"] > 0
print("all analysis checks passed")
