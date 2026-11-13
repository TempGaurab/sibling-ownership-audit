#!/usr/bin/env python3
"""Agreement between one human coder and the LLM stratum codes (Nov 12 deviation 6c40f1e0353f, post hoc). The LLM codes stay the
codes used in every analysis; this only measures them. Reads no score. Aggregate output only (no repository named).
Per field: Cohen's kappa and raw agreement of the human against the merged LLM code (disagreements between passes -> unknown, as in
merge_coding.py) and against each pass, on the sample; raw agreement weighted back to all distinct cards by inverse sampling
fraction per group; and counts of cards the human codes same or partial that the merged code does not, and the reverse.
Usage: human_agreement.py HUMAN_CODES.csv OUT.json"""
import csv, glob, json, sys, numpy as np

D = ("ultrachat", "oasst1", "gsm8k")
L = lambda p: {r["key"]: r for f in sorted(glob.glob(f"data/strata/pass{p}/*.jsonl")) for r in map(json.loads, open(f))}
a, b = L(1), L(2)
merged = {k: dict(overlap={d: a[k]["overlap"][d] if a[k]["overlap"][d] == b[k]["overlap"][d] else "unknown" for d in D},
                  **{f: a[k][f] if a[k][f] == b[k][f] else "unknown" for f in ("recipe_family", "teacher_use")}) for k in a}
S = json.load(open("data/strata/human_sample.json"))["groups"]
weight = {k: g["population"] / len(g["sampled"]) for g in S.values() for k in g["sampled"]}
H = {r["key"]: r for r in csv.DictReader(open(sys.argv[1]))}
assert set(H) == set(weight), f"CSV has {len(H)} keys, sample has {len(weight)}; missing {len(set(weight) - set(H))}"
blank = [k for k, r in H.items() if any(not r[f] for f in D + ("recipe_family", "teacher_use"))]
assert not blank, f"{len(blank)} cards incomplete"
keys = sorted(H)
human = {k: dict(overlap={d: H[k][d] for d in D}, recipe_family=H[k]["recipe_family"], teacher_use=H[k]["teacher_use"]) for k in keys}


def kappa(x, y, w=None):
    x, y = np.array(x), np.array(y); w = np.ones(len(x)) if w is None else np.asarray(w, float); w = w / w.sum()
    po = float((w * (x == y)).sum()); pe = sum(float((w * (x == l)).sum() * (w * (y == l)).sum()) for l in set(x) | set(y))
    return dict(kappa=(po - pe) / (1 - pe) if pe < 1 else None, agreement=po)


FIELDS = {**{f"overlap[{d}]": (lambda r, d=d: [r["overlap"][d]]) for d in D},
          "overlap pooled": lambda r: [r["overlap"][d] for d in D],
          "dataset stated at all": lambda r: ["unknown" if set(r["overlap"].values()) == {"unknown"} else "stated"],
          "stratum label (same/partial vs other)": lambda r: [r["overlap"][d] in ("same", "partial") for d in D],
          "recipe_family": lambda r: [r["recipe_family"]], "teacher_use": lambda r: [r["teacher_use"]]}
R = dict(label="post hoc; one human coder; DEVIATIONS.md Nov 12 entry 6c40f1e0353f", n_cards=len(keys),
         groups={g: dict(population=v["population"], sampled=len(v["sampled"])) for g, v in S.items()}, fields={})
for name, f in FIELDS.items():
    out = {}
    for ref, C in (("merged", merged), ("pass1", a), ("pass2", b)):
        x = [v for k in keys for v in f(human[k])]; y = [v for k in keys for v in f(C[k])]
        w = [weight[k] for k in keys for _ in f(human[k])]
        out[ref] = dict(sample=kappa(x, y), weighted=kappa(x, y, w))
    R["fields"][name] = out
pos = lambda r, d: r["overlap"][d] in ("same", "partial")
R["same_or_partial"] = {d: dict(human=sum(pos(human[k], d) for k in keys), llm=sum(pos(merged[k], d) for k in keys),
                                human_only=sum(pos(human[k], d) and not pos(merged[k], d) for k in keys),
                                llm_only=sum(pos(merged[k], d) and not pos(human[k], d) for k in keys)) for d in D}
stt = lambda r: "unknown" if set(r["overlap"].values()) == {"unknown"} else "stated"
R["dataset_stated_crosstab"] = {f"human_{h}|llm_{l}": sum(stt(human[k]) == h and stt(merged[k]) == l for k in keys) for h in ("stated", "unknown") for l in ("stated", "unknown")}
R["recipe_crosstab_offdiag"] = {f"human_{h}|llm_{l}": n for (h, l), n in sorted(__import__("collections").Counter(
    (human[k]["recipe_family"], merged[k]["recipe_family"]) for k in keys if human[k]["recipe_family"] != merged[k]["recipe_family"]).items())}
json.dump(R, open(sys.argv[2], "w"), indent=1)
for name, v in R["fields"].items():
    m = v["merged"]; print(f"{name}: kappa {m['sample']['kappa']:.3f}, agreement {m['sample']['agreement']:.3f} (weighted {m['weighted']['agreement']:.3f})")
print("HUMAN_AGREEMENT_DONE", R["same_or_partial"])
