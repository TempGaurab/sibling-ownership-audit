#!/usr/bin/env python3
"""Append accepted top-up controls to one family's extraction manifest, after that family's extraction has finished and before
scoring. Refuses if scores already exist or any model has not been attempted. Keeps the earlier file as manifest_before_topup.json.
Idempotent. Reports usable controls against the preregistered floor (100 across 10 bases), from file status only.
Usage: merge_topup.py R FAMILY"""
import collections, json, os, shutil, sys
R, fam = sys.argv[1:3]; F = f"{R}/outputs/main/features"; O = f"{F}/{fam}"; p = f"{O}/manifest.json"
key = lambda i: i.replace("/", "__")
ok = lambda i: os.path.exists(f"{O}/bb/{key(i)}/meta.json"); err = lambda i: os.path.exists(f"{O}/bb/{key(i)}/error.json")
if any(os.path.exists(f"{O}/{x}") for x in ("scores.csv", "rules.jsonl", "llmprint.jsonl")): sys.exit("ABORT: scoring output already exists")
M = json.load(open(p)); missing = [e["id"] for e in M if not (ok(e["id"]) or err(e["id"]))]
if missing: sys.exit(f"ABORT: {len(missing)} models not attempted, e.g. {missing[:3]}")
A = json.load(open(f"{F}/topup_manifest.json")); have = {e["id"] for e in M}; new = [e for e in A if e["id"] not in have]
if new and not os.path.exists(f"{O}/manifest_before_topup.json"): shutil.copy(p, f"{O}/manifest_before_topup.json")
if new:
    tmp = p + ".tmp"; json.dump(M + new, open(tmp, "w"), indent=0); os.replace(tmp, p)
C = [e for e in json.load(open(p)) if e["kind"] == "control"]; usable = [e for e in C if ok(e["id"])]; bases = collections.Counter(e["base"].split("models--")[-1].split("/")[0] for e in usable)
print(fam, "appended", len(new), "controls listed", len(C), "usable", len(usable), "bases", len(bases), "floor met", len(usable) >= 100 and len(bases) >= 10); print(sorted(bases.items()))
