#!/usr/bin/env python3
"""Extraction manifests for downloaded hub models. The last result line per repo counts. Usage: make_hub_manifest.py R OUTDIR"""
import collections, json, os, sys
R, out = sys.argv[1:3]; REV = {"Qwen/Qwen3-0.6B": open(f"{R}/data/qwen3.rev").read().strip(), "TinyLlama/TinyLlama-1.1B-Chat-v1.0": open(f"{R}/data/tinyllama.rev").read().strip()}
last = {}
for l in open(f"{R}/outputs/hub/downloads.jsonl"):
    r = json.loads(l); last[r["id"]] = r
ok = {k: r for k, r in last.items() if r["status"] == "ok"}; rows = []; bases = {r["id"]: r["path"] for r in ok.values() if r["role"] == "control_base"}
for r in ok.values():
    if r["role"] == "control_base": continue
    p = r["path"]; ad = os.path.exists(f"{p}/adapter_config.json"); own = any(os.path.exists(f"{p}/{x}") for x in ("tokenizer.json", "tokenizer.model", "vocab.json"))
    b = r["base"]; rev = REV.get(b)
    if b in bases: b, rev = bases[b], None
    elif r["role"] == "control": continue
    e = dict(id=r["id"], kind="hub_sibling" if r["role"] == "sibling" else "control", family=r["family"], path=p, base=b, base_revision=rev, adapter=ad)
    if not ad and own: e["tok"] = p
    rows.append(e)
os.makedirs(out, exist_ok=True); json.dump(rows, open(f"{out}/hub_manifest.json", "w"), indent=0)
print(collections.Counter((e["kind"], e["family"]) for e in rows), "not ok", collections.Counter(r["status"] for r in last.values()))
