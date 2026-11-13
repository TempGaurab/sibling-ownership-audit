#!/usr/bin/env python3
"""Download listed repos at their pinned commit, with no account token, weights and configs from the repo root only.
Never fetches or runs repo code. Idempotent: finished repos are skipped, failures retried; the last line per repo is its result. Paced, and backs off on 429. Usage: download_hub.py LIST.json CACHE RESULTS.jsonl [WORKERS]"""
import concurrent.futures as cf, json, os, re, sys, threading, time
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
from huggingface_hub import HfApi, snapshot_download
lst, cache, res = sys.argv[1:4]; W = int(sys.argv[4]) if len(sys.argv) > 4 else 4; LIMIT = 40e9
KEEP = re.compile(r"^(config\.json|generation_config\.json|adapter_config\.json|tokenizer.*|special_tokens_map\.json|added_tokens\.json|vocab\.json|merges\.txt|chat_template.*|.*\.safetensors|.*\.safetensors\.index\.json|pytorch_model.*\.bin|pytorch_model.*\.bin\.index\.json|adapter_model\.bin)$")
data = json.load(open(lst)); rows = data["rows"]
done = {r["id"] for r in map(json.loads, open(res)) if r["status"] == "ok"} if os.path.exists(res) else set(); lock = threading.Lock(); api = HfApi(token=False)
PACE = 2.0; last = [0.0]
def pace():
    with lock:
        wait = last[0] + PACE - time.time()
        if wait > 0: time.sleep(wait)
        last[0] = time.time()
def get(r):
    for attempt in range(30):
        pace(); o = get1(r)
        if o["status"] != "fail" or "429" not in o.get("error", ""): return o
        m = re.search(r"Retry after (\d+) seconds", o["error"]); time.sleep(int(m.group(1)) + 10 if m else 120)
    return o
def get1(r):
    out = dict(id=r["id"], sha=r["sha"], role=r["role"], family=r["family"], format=r["format"], base=r["base"])
    try:
        info = api.model_info(r["id"], revision=r["sha"], files_metadata=True, token=False)
        files = [s for s in info.siblings if "/" not in s.rfilename and KEEP.match(s.rfilename)]
        names = {s.rfilename for s in files}
        if any(n.endswith(".safetensors") for n in names):  # never take a pickle when safetensors exist
            files = [s for s in files if not s.rfilename.endswith(".bin") and "pytorch_model" not in s.rfilename]
        size = sum(s.size or 0 for s in files); out["bytes"] = size
        if size > LIMIT: out["status"] = "too_large"; return out
        if info.sha != r["sha"]: out["status"] = "sha_mismatch"; return out
        out["path"] = snapshot_download(r["id"], revision=r["sha"], allow_patterns=[s.rfilename for s in files], cache_dir=cache, token=False, max_workers=2)
        out["status"] = "ok"
    except Exception as e:
        out["status"] = "fail"; out["error"] = repr(e)[:300]
    return out
todo = [r for r in rows if r["id"] not in done]; print("todo", len(todo), flush=True)
with cf.ThreadPoolExecutor(W) as ex, open(res, "a") as f:
    for o in ex.map(get, todo):
        f.write(json.dumps(o) + "\n"); f.flush(); print(o["status"], o["id"], o.get("bytes"), flush=True)
print("DOWNLOAD_DONE", flush=True)
