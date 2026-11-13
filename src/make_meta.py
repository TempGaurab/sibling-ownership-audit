#!/usr/bin/env python3
"""Per-model metadata for audit.py, from provenance only. Controlled siblings: dataset and recipe cell (full@LR, lora8, lora16)
from each run's meta.json. Hub models: upload date from the census sample file. Usage: make_meta.py MANIFEST.json OUT.json [CENSUS_SAMPLE.json]"""
import json, os, sys
man, out = sys.argv[1:3]; created = {}
if len(sys.argv) > 3:
    d = json.load(open(sys.argv[3])); created = {r["id"]: r.get("created") for r in (d["rows"] if isinstance(d, dict) else d)}
M = {}
for e in json.load(open(man)):
    if e["kind"] == "sib":
        g = json.load(open(f"{e['path']}/meta.json"))
        M[e["id"]] = dict(dataset=g["dataset"], recipe=f"full@{g['lr']:g}" if g["recipe"] == "full" else g["recipe"])
    elif e["id"] in created:
        M[e["id"]] = dict(created=created[e["id"]])
json.dump(M, open(out, "w"), indent=0); print("META", len(M))
