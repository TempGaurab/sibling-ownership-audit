#!/usr/bin/env python3
"""Preregistered provenance check for hub siblings, using the declared base only and never a claimant (see DEVIATIONS).
Pass: at least 90% of base parameter names exist with matching shapes after merging any adapter, cosine to base above 0.5,
not bit-identical. Idempotent. Usage: check_weights.py MANIFEST.json OUT.jsonl SHARD NSHARDS"""
import json, os, sys, torch
from transformers import AutoModelForCausalLM
man, out, sh, n = sys.argv[1:5]; rows = [e for e in json.load(open(man)) if e.get("kind") == "hub_sibling"][int(sh)::int(n)]
done = {json.loads(l)["id"] for l in open(out)} if os.path.exists(out) else set(); bases = {}
def base_sd(b, rev):
    if b not in bases:
        bases[b] = {k: v.float().cuda() for k, v in AutoModelForCausalLM.from_pretrained(b, revision=rev, dtype=torch.float32).state_dict().items() if v.is_floating_point()}
    return bases[b]
def sd(e):
    if e.get("adapter"):
        from adapter_load import merge_adapter
        m = merge_adapter(AutoModelForCausalLM.from_pretrained(e["base"], revision=e["base_revision"], dtype=torch.float32), e["path"])
    else:
        m = AutoModelForCausalLM.from_pretrained(e["path"], dtype=torch.float32)
    return {k: v.float().cuda() for k, v in m.state_dict().items() if v.is_floating_point()}
with open(out, "a") as f:
    for e in rows:
        if e["id"] in done: continue
        r = dict(id=e["id"])
        try:
            b = base_sd(e["base"], e["base_revision"]); s = sd(e)
            ks = [k for k in b if k in s and s[k].shape == b[k].shape]; frac = len(ks) / len(b)
            num = sum((s[k] * b[k]).sum().item() for k in ks); na = sum((s[k] ** 2).sum().item() for k in ks); nb = sum((b[k] ** 2).sum().item() for k in ks)
            cos = num / (na * nb) ** .5 if na and nb else float("nan"); same = frac == 1 and all(torch.equal(s[k], b[k]) for k in ks)
            r.update(frac=frac, cos=cos, identical=same, passed=bool(frac >= .9 and cos > .5 and not same))
            r["reason"] = None if r["passed"] else ("identical_to_base" if same else "shape_mismatch" if frac < .9 else "nonfinite_weights" if cos != cos else "low_cosine_to_base")
            del s; torch.cuda.empty_cache()
        except Exception as ex:
            r.update(passed=False, reason="load_failed", error=repr(ex)[:300])
        f.write(json.dumps(r) + "\n"); f.flush()
print("CHECK_DONE", flush=True)
