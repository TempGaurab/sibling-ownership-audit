#!/usr/bin/env python3
"""Manifests for extraction from the local model tree. Usage: make_manifest.py R FAMILY BASE REV OUTDIR"""
import glob, json, os, sys
R, fam, base, rev, out = sys.argv[1:6]; M = f"{R}/models/{fam}"
def entry(path, kind):
    e = dict(id=os.path.basename(path), kind=kind, path=path, base=base, base_revision=rev, adapter=False)
    cfg = f"{path}/adapter_config.json"
    if os.path.exists(cfg):
        e["adapter"] = True; b = json.load(open(cfg))["base_model_name_or_path"]
        if b != base: e["base"], e["base_revision"] = b, None
    return e
rows = [entry(p, k) for k in ("claimant", "sib", "copy") for p in sorted(glob.glob(f"{M}/{k}/*")) if os.path.exists(f"{p}/meta.json")]
rows.append(dict(id="base", kind="base", path=base, revision=rev, base=base, base_revision=rev, adapter=False))
os.makedirs(out, exist_ok=True)
json.dump(rows, open(f"{out}/manifest.json", "w"), indent=0)
cl = [e for e in rows if e["kind"] == "claimant"]
for c in cl:  # one witness per claimant: siblings on another dataset, fixed in advance; a second covers the case where the first is the suspect
    prefix = c["id"].split("_")[0]; c["witness"] = [e for e in rows if e["kind"] == "sib" and "_high_" in e["id"] and not e["id"].startswith(prefix)][:2]
json.dump(cl, open(f"{out}/claimants.json", "w"), indent=0)
print(fam, len(rows), {k: sum(e["kind"] == k for e in rows) for k in ("claimant", "sib", "copy", "base")})
