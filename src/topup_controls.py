#!/usr/bin/env python3
"""Fill each control base to 11 usable models: walk the reserves of make_topup_list.py in seeded order, extracting black-box
features with the frozen extract_blackbox.py functions, and stop at 11. A model is usable if its features were written. Only
load status decides; no feature value is read. Features go to both families' bb directories, as prompts are shared.
Idempotent. Writes outputs/main/features/topup_manifest.json and topup_log.jsonl. --dry reports the walk without loading a model.
Usage: topup_controls.py R TOPUP_LIST.json [--dry]"""
import json, os, sys
R, lst = sys.argv[1:3]; DRY = "--dry" in sys.argv; TARGET = 11; FAMS = ("qwen3", "tinyllama"); F = f"{R}/outputs/main/features"
data = json.load(open(lst)); key = lambda i: i.replace("/", "__")
def state(fam, i):
    d = f"{F}/{fam}/bb/{key(i)}"
    return "ok" if os.path.exists(f"{d}/meta.json") else "fail" if os.path.exists(f"{d}/error.json") else "pending"
def last(p):
    o = {}
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l); o[r["id"]] = r
    return o
dl = last(f"{R}/outputs/hub/topup_downloads.jsonl"); base_paths = {k: r["path"] for k, r in last(f"{R}/outputs/hub/downloads.jsonl").items() if r["role"] == "control_base" and r["status"] == "ok"}
pending = [i for ids in data["originals"].values() for i in ids if state("qwen3", i) == "pending"]
if pending and not DRY: sys.exit(f"ABORT: {len(pending)} original controls not yet attempted in qwen3; wait for its extraction to finish")
fns = None
def entry(r):
    p = dl[r["id"]]["path"]; ad = os.path.exists(f"{p}/adapter_config.json"); own = any(os.path.exists(f"{p}/{x}") for x in ("tokenizer.json", "tokenizer.model", "vocab.json"))
    e = dict(id=r["id"], kind="control", family=None, path=p, base=base_paths[r["base"]], base_revision=None, adapter=ad)
    if not ad and own: e["tok"] = p
    return e
def extract(e):
    global fns
    if fns is None:
        import numpy as np, torch
        from extract_blackbox import load, nexttokens, reef_gram
        fns = dict(np=np, torch=torch, load=load, nt=nexttokens, g=reef_gram, prompts=json.load(open(f"{R}/data/prompts.json")), reef=json.load(open(f"{R}/data/reef.json")))
    import time; t = time.time(); np = fns["np"]
    try:
        m, tok = fns["load"](e); nt, g = fns["nt"](m, tok, fns["prompts"]), fns["g"](m, tok, fns["reef"]); del m; fns["torch"].cuda.empty_cache()
    except Exception as ex:
        for f in FAMS:
            d = f"{F}/{f}/bb/{key(e['id'])}"; os.makedirs(d, exist_ok=True); json.dump(dict(e, error=repr(ex)[:500]), open(f"{d}/error.json", "w"))
        print("FAIL", e["id"], repr(ex)[:200], flush=True); return False
    for f in FAMS:
        d = f"{F}/{f}/bb/{key(e['id'])}"; os.makedirs(d, exist_ok=True); np.save(f"{d}/nexttoken.npy", nt); np.save(f"{d}/gram.npy", g)
        json.dump(dict(e, seconds=time.time() - t, n_prompts=len(fns["prompts"]), topup=True), open(f"{d}/meta.json", "w"))
    print("ok", e["id"], round(time.time() - t, 1), flush=True); return True
accepted = []; log = []; total = 0
for base, ids in data["originals"].items():
    use = [i for i in ids if state("qwen3", i) == "ok"]; n = len(use); pend = sum(state("qwen3", i) == "pending" for i in ids); start = n
    for r in sorted((x for x in data["rows"] if x["base"] == base), key=lambda x: x["rank"]):
        if n >= TARGET: break
        i = r["id"]; rec = dict(id=i, base=base, rank=r["rank"])
        if dl.get(i, {}).get("status") != "ok": rec["status"] = "not_downloaded:" + dl.get(i, {}).get("status", "absent"); log.append(rec); continue
        s = {f: state(f, i) for f in FAMS}
        if DRY: rec["status"] = "would_try"; n += 1; log.append(rec); continue
        if all(v == "ok" for v in s.values()): good = True
        elif "fail" in s.values(): good = False
        else: good = extract(entry(r))
        rec["status"] = "used" if good else "load_failed"; log.append(rec)
        if good: accepted.append(entry(r)); n += 1
    total += n
    print(f"{base}: originals usable {start} of {len(ids)} (pending {pend}); after top-up {n}", flush=True)
for rec in log: print(json.dumps(rec), flush=True)
print("TOTAL_USABLE", total, "pending_originals", len(pending), "dry" if DRY else "")
if not DRY:
    json.dump(accepted, open(f"{F}/topup_manifest.json", "w"), indent=0); open(f"{F}/topup_log.jsonl", "w").write("".join(json.dumps(x) + "\n" for x in log)); print("TOPUP_DONE", flush=True)
