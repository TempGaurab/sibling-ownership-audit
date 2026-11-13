#!/usr/bin/env python3
"""Per-model white-box scores against each claimant, for same-architecture models:
task-vector cosine; centered residual lineage score (Thakur and Khoury, reimplemented from the paper's description);
Witness Overlap endpoint scores with Frobenius cosine (Li et al., reimplemented from the paper), for the pair (claimant, suspect)
and for the pair (base, suspect), which trains the other family's threshold.
Idempotent. Usage: extract_whitebox.py MANIFEST.json CLAIMANTS.json OUTDIR SHARD NSHARDS"""
import json, os, re, sys, time, numpy as np, torch
from scipy.optimize import linear_sum_assignment
from transformers import AutoModelForCausalLM

DTYPE = torch.float32

def sd(e):
    kw = dict(dtype=DTYPE)  # float32: bf16 rounding creates shared task-vector structure across siblings
    if e.get("adapter"):
        from adapter_load import merge_adapter
        m = merge_adapter(AutoModelForCausalLM.from_pretrained(e["base"], revision=e.get("base_revision") or e.get("revision"), **kw), e["path"])
    else:
        m = AutoModelForCausalLM.from_pretrained(e["path"], revision=e.get("revision"), **kw)
    return {k: v.float().cuda() for k, v in m.state_dict().items() if v.is_floating_point()}

def signature(s):
    """phi_l: normalized traceless remainder of the MLP branch product down_proj @ up_proj, per block."""
    out = {}
    for k in s:
        m = re.match(r"(.*layers\.(\d+)\.mlp)\.down_proj\.weight$", k)
        if m:
            M = s[k] @ s[f"{m.group(1)}.up_proj.weight"]
            R = M - torch.trace(M) / M.shape[0] * torch.eye(M.shape[0], device=M.device)
            out[int(m.group(2))] = (R / R.norm()).flatten()
    return torch.stack([out[i] for i in sorted(out)])

def lineage(a, b):
    G = (a @ b.T).cpu().numpy(); r, c = linear_sum_assignment(-G)
    return float(G[r, c].mean())

ROLE = re.compile(r"\.\d+\.")
def overlap(a, b, w):
    """Mean per-tensor Frobenius cosine of (b - a) and (w - a) over 2-D tensors other than embeddings, output head and norms,
    averaged within each role across layers, then across roles."""
    roles = {}
    for k in a:
        if a[k].dim() != 2 or re.search(r"embed|lm_head|norm", k) or k not in b or k not in w or not (a[k].shape == b[k].shape == w[k].shape): continue
        x, y = (b[k] - a[k]).flatten(), (w[k] - a[k]).flatten(); d = x.norm() * y.norm()
        if d > 0: roles.setdefault(ROLE.sub(".N.", k), []).append(((x * y).sum() / d).item())
    return float(np.mean([np.mean(v) for v in roles.values()])) if roles else float("nan")

def wo(a, b, w): return dict(s_a=overlap(a, b, w), s_b=overlap(b, a, w))

def taskcos(s, base, cl):
    num, na, nb = 0.0, 0.0, 0.0
    for k in base:
        if k not in s or s[k].shape != base[k].shape: continue
        d, e = s[k] - base[k], cl[k] - base[k]
        num += (d * e).sum().item(); na += (d * d).sum().item(); nb += (e * e).sum().item()
    return num / (na * nb) ** 0.5 if na and nb else float("nan")

def main():
    man, cl, outdir, sh, n = sys.argv[1:6]
    entries = [e for e in json.load(open(man)) if e.get("kind") != "control"][int(sh)::int(n)]; claimants = json.load(open(cl))
    c0 = claimants[0]; base = sd({"path": c0["base"], "revision": c0.get("base_revision") or c0.get("revision"), "adapter": False})
    wit = {x["id"]: sd(x) for c in claimants for x in c["witness"]}
    cs = {c["id"]: sd(c) for c in claimants}; sigs = {k: signature(v).float() for k, v in cs.items()}
    os.makedirs(outdir, exist_ok=True)
    for e in entries:
        f = f"{outdir}/{e['id'].replace('/', '__')}.json"
        if os.path.exists(f): continue
        t = time.time()
        try:
            s = sd(e); si = signature(s)
            def pick(c): return next(wit[x["id"]] for x in c["witness"] if x["id"] != e["id"])
            row = dict(id=e["id"], scores={c["id"]: dict(taskcos=taskcos(s, base, cs[c["id"]]), lineage=lineage(sigs[c["id"]], si), wo=wo(cs[c["id"]], s, pick(c))) for c in claimants if c["id"] != e["id"]},
                       wo_base=wo(base, s, pick(claimants[0])) if e["id"] != "base" else None, seconds=time.time() - t)
            json.dump(row, open(f, "w")); print("ok", e["id"], round(time.time() - t, 1), flush=True); del s, si; torch.cuda.empty_cache()
        except Exception as ex:
            print("FAIL", e["id"], repr(ex)[:200], flush=True)

if __name__ == "__main__":
    main()
