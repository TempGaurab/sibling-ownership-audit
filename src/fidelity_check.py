#!/usr/bin/env python3
"""Post hoc fidelity summaries for the REEF and LLMPrint reimplementations, written Nov 12, 2026, after the confirmatory
analysis. No model is run and no rule, threshold or hypothesis changes: it reads existing scores.csv, llmprint.jsonl and the
LLMPrint fingerprint files and writes aggregate statistics only (no model is named), so that our values can be set beside
those the source papers report. The LLMPrint validation draw and decision rule are the frozen ones, imported from apply_rules.py.
Usage: fidelity_check.py CODE_DIR OUT.json FAMILY:FEATURE_DIR:FP_DIR [...]"""
import glob, json, os, sys, numpy as np, pandas as pd

sys.path.insert(0, sys.argv[1])
from apply_rules import load, llmprint  # frozen code: same validation draw (seed 2027) and threshold


def q(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return None if len(x) == 0 else dict(n=int(len(x)), q05=float(np.quantile(x, .05)), median=float(np.median(x)), q95=float(np.quantile(x, .95)), mean=float(x.mean()))


def group(kind, stratum):
    if kind == "copy": return "copy|" + stratum.rsplit("_", 1)[0]
    if kind == "sib": return "sib|" + stratum
    return kind  # control, hub


def main():
    out = sys.argv[2]; R = {}
    for spec in sys.argv[3:]:
        fam, fdir, fpdir = spec.split(":")
        T, info = load(fam, fdir)
        # REEF: linear CKA at the layer matching the paper's layer 18, by population.
        r = T[T.score == "reef"].assign(g=lambda d: [group(k, s) for k, s in zip(d.kind, d.stratum.astype(str))])
        R[f"{fam}|reef_cka"] = {g: q(v.value) for g, v in r.groupby("g")}
        # LLMPrint: search success on the claimant, agreement by population, and the frozen rule on held-out controls.
        B = {json.loads(l)["id"]: json.loads(l)["bits"] for l in open(f"{fdir}/llmprint.jsonl")}
        cs = sorted(next(iter(B.values()))); succ = {}
        for c in cs:
            if c not in B: continue
            own = [b for b in B[c][c] if b is not None]
            succ[c] = dict(pairs=len(B[c][c]), usable=len(own), claimant_prefers_target=float(np.mean(own)))
        loss = {os.path.basename(p)[:-6]: q([json.loads(l)["loss"] for l in open(p)]) for p in glob.glob(f"{fpdir}/*.jsonl")}
        R[f"{fam}|llmprint_search"] = dict(own_bits=succ, final_loss=loss)
        kinds = {(c, k): group(t, str(s)) for c, k, t, s in zip(info.claimant, info.suspect, info.kind, info.stratum)}  # stratum is relative to the claimant
        agr = {}
        for c in cs:
            if c not in B: continue
            ref = B[c][c]
            for k, v in B.items():
                if k == c or (c, k) not in kinds: continue
                pr = [x == y for x, y in zip(v[c], ref) if x is not None and y is not None]
                if pr: agr.setdefault(kinds[(c, k)], []).append(float(np.mean(pr)))
        R[f"{fam}|llmprint_agreement"] = {g: q(v) for g, v in agr.items()}
        L = llmprint(fam, fdir, info)  # frozen rule; validation controls are excluded inside
        L = L.assign(g=[group(k, str(s)) for k, s in zip(L.kind, L.stratum)])
        R[f"{fam}|llmprint_positive_rate"] = {g: dict(n=int(len(v)), rate=float(v.x.mean())) for g, v in L.groupby("g")}
    json.dump(R, open(out, "w"), indent=1); print("FIDELITY_DONE", list(R))


if __name__ == "__main__":
    main()
