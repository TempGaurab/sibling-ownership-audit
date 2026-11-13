"""Synthetic checks of the ablation code, with known directions. No model is involved. Numbers here are simulated and never reported."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "src")
from ablations import perturb, run, effective_n, GRID
rng = np.random.default_rng(0)
r = np.arange(100.0)
assert len(perturb(r, ("size", 49), rng, None, None)) == 49 and perturb(r, ("size", 199), rng, None, None) is None
f = perturb(r, ("filter", .10), rng, None, None); assert f.max() < 90 and len(f) >= 89
c = perturb(r, ("contam", .10), rng, None, np.array([1000.0])); assert (c == 1000).sum() == 10 and len(c) == 100
n = perturb(r, ("noise", .05), rng, np.array([-5.0]), None); assert (n == -5).sum() == 5 and len(n) == 100
assert perturb(r, ("contam", .05), rng, None, np.array([])) is None
def make(high_shift=3.0):
    rows = []; g = np.random.default_rng(1)
    for c in ("c0", "c1", "c2"):
        def add(kind, st, v, ds, rec, pub=None):
            rows.append(dict(family="qwen3", claimant=c, suspect=f"m{len(rows)}", score="s", value=v, kind=kind, stratum=st, dataset=ds, recipe=rec, publisher=pub, month="2026-01"))
        for i in range(500): add("control", "ind", g.normal(-20), "x", "x", f"p{i}")
        for st, sh in (("high", high_shift), ("mid", 0.0), ("low", 0.0)):
            for ds in ("d0", "d1", "d2"):
                for rec in ("full@2e-05", "full@5e-05", "lora8", "lora16"):
                    for i in range(40): add("sib", st, g.normal(sh), ds, rec)
        for i in range(120): add("hub", "disjoint", g.normal(), "h", "h", f"h{i % 30}")
        for t in ("quant_8", "quant_4", "ft_full_20", "ft_lora16_20", "merge_50"):
            for i in range(20): add("copy", t, g.normal(high_shift + 2.6), "c", "c")
    return pd.DataFrame(rows)
T = make(); k = "qwen3|s"; res = {f"{a}_{b:g}": run(T, (a, b)) for a, b in GRID}
far = lambda n: res[n]["far"][k]["est"]; hi = lambda n: res[n]["far_by_stratum"]["qwen3|s|high"]["est"]; pw = lambda n: res[n]["power_all_copies"][k]["mean"]
for n in res: print(f"{n:12s} FAR {far(n):.3f} high {hi(n):.3f} power {pw(n):.3f}")
assert abs(far("base_0") - .05) < .02, "held-out base near level"
assert abs(far("size_49") - .05) < .025 and abs(far("size_199") - .05) < .025, "reference size keeps the level"
assert far("filter_0.1") > far("base_0") + .03, "score-based removal inflates false accusation"
assert pw("contam_0.1") < pw("base_0"), "contamination is conservative for power"
assert far("contam_0.1") < far("base_0"), "contamination is conservative for false accusation"
assert abs(hi("noise_0.2") - .0625) < .015 and hi("noise_0.2") > hi("base_0"), "label noise: with 20% of references from N(0,1), the 95th percentile of the mixture gives 1 - Phi(1.534) = 0.0625 in the high stratum"
assert run(T, ("size", 99999)) == {} or "far" not in run(T, ("size", 99999)), "impossible size is skipped"
assert abs(res["base_0"]["far_sibling_only"][k]["est"] - .05) < .02
a, b = run(T, ("noise", .1)), run(T, ("noise", .1)); assert a["far"][k]["est"] == b["far"][k]["est"], "seeded"
e = effective_n(T)["qwen3"]; assert e == dict(hub_models=120, hub_publishers=30, control_models=500, control_publishers=500)
print("all ablation checks passed")
