#!/usr/bin/env python3
"""Applies each published method's original rule to extracted features, and writes the long score table for analyze.py.
All rules are reimplementations. Reports go in aggregate only, and repo names never leave this table. Usage:
audit.py FAMILY OUTDIR MANIFEST.json CLAIMANTS.json STRATA.json|none CARD_META.json|none SCORES.csv RULES.jsonl
Set-level rules run under pool P1 (claimant plus unrelated controls) and P2 (P1 plus the shared base)."""
import csv, json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rules import nikolic, mps
fam, O, man, clj, stj, metaj, out, rulesf = sys.argv[1:9]
M = json.load(open(man)); CL = json.load(open(clj)); strata = json.load(open(stj)) if stj != "none" else {}; meta = json.load(open(metaj)) if metaj != "none" else {}
DS = {"c0_ultrachat": "ultrachat", "c1_oasst1": "oasst1", "c2_gsm8k": "gsm8k"}
key = lambda i: i.replace("/", "__"); nt, gr = {}, {}
for e in M:
    d = f"{O}/bb/{key(e['id'])}"
    if os.path.exists(f"{d}/meta.json"): nt[e["id"]], gr[e["id"]] = np.load(f"{d}/nexttoken.npy"), np.load(f"{d}/gram.npy")
wb = {}
for e in M:
    f = f"{O}/wb/{key(e['id'])}.json"
    if os.path.exists(f): wb[e["id"]] = json.load(open(f))
def cka(a, b): return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
def describe(e, c):
    i = e["id"]; k = e["kind"]; pre = c.split("_")[0]
    if k == "sib":
        st = ("high" if "_high_" in i else "mid") if i.startswith(pre + "_") else "low"; g = meta.get(i, {})
        return dict(kind="sib", stratum=st, dataset=g.get("dataset", ""), recipe=g.get("recipe", ""), publisher="", month="")
    if k == "copy": return None if not i.startswith(pre + "_") else dict(kind="copy", stratum="_".join(i.split("_")[2:-1]) if i.split("_")[-1].isdigit() and "quant" not in i and "prune" not in i else "_".join(i.split("_")[2:]), dataset="", recipe="", publisher="", month="")
    if k == "control": return dict(kind="control", stratum="ind", dataset="", recipe="", publisher=i.split("/")[0], month="")
    if k == "hub_sibling":
        s = strata.get(i, {}).get("overlap", {}).get(DS.get(c, ""), "unknown"); g = meta.get(i, {})
        return dict(kind="hub", stratum={"same": "shared", "partial": "shared", "none": "disjoint"}.get(s, "unknown"), dataset="", recipe="", publisher=i.split("/")[0], month=(g.get("created") or "")[:7])
    return None
ctrl = [e["id"] for e in M if e["kind"] == "control" and e["id"] in nt]; base = next((e["id"] for e in M if e["kind"] == "base"), None)
with open(out, "w", newline="") as f, open(rulesf, "w") as rf:
    w = csv.writer(f); w.writerow("family claimant suspect score value kind stratum dataset recipe publisher month".split())
    for cl in CL:
        c = cl["id"]
        if c not in nt: continue
        P1 = {k: nt[k] for k in [c] + ctrl}; P2 = dict(P1, **({base: nt[base]} if base in nt else {}))
        for e in M:
            i = e["id"]; d = describe(e, c)
            if i == c or d is None or i not in nt: continue
            row = lambda name, v: w.writerow([fam, c, i, name, v, d["kind"], d["stratum"], d["dataset"], d["recipe"], d["publisher"], d["month"]])
            row("agree", float((nt[i][:5000] == nt[c][:5000]).mean())); row("reef", cka(gr[i], gr[c]))
            s = wb.get(i, {}).get("scores", {}).get(c)
            if s:
                row("taskcos", s["taskcos"]); row("lineage", s["lineage"]); row("wo", -min(s["wo"]["s_a"], s["wo"]["s_b"]))
            if d["kind"] in ("sib", "hub", "copy"):
                drop = lambda P: {k: v for k, v in P.items() if k != i}
                st1, ni1 = mps(nt[i], drop(P1), R=1000); st2, ni2 = mps(nt[i], drop(P2), R=1000)
                rf.write(json.dumps(dict(family=fam, claimant=c, suspect=i, kind=d["kind"], stratum=d["stratum"], nikolic_p1=nikolic(nt[i], drop(P1)) == c, nikolic_p2=nikolic(nt[i], drop(P2)) == c, mps_p1=c in st1, mps_p2=c in st2, ni_p1=ni1, ni_p2=ni2)) + "\n")
print("AUDIT_DONE", fam, flush=True)
