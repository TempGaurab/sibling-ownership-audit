#!/usr/bin/env python3
"""Confirmatory extraction manifest for one family: controlled models from make_manifest.py, plus hub siblings that passed the
weight check, plus independent controls. Selection uses provenance only, never a score. Run after copies finish.
Usage: make_main_manifest.py R FAMILY BASE REV WEIGHT_CHECK.jsonl HUB_MANIFEST.json OUTDIR"""
import collections, json, subprocess, sys, os
R, fam, base, rev, wc, hub, out = sys.argv[1:8]
subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_manifest.py"), R, fam, base, rev, out], check=True)
last = {}
for l in open(wc):
    r = json.loads(l); last[r["id"]] = r
H = json.load(open(hub)); passed = {i for i, r in last.items() if r["passed"]}
rows = json.load(open(f"{out}/manifest.json"))
rows += [e for e in H if e["kind"] == "hub_sibling" and e["family"] == fam and e["id"] in passed] + [e for e in H if e["kind"] == "control"]
json.dump(rows, open(f"{out}/manifest.json", "w"), indent=0)
print(fam, collections.Counter(e["kind"] for e in rows))
