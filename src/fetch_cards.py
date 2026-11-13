#!/usr/bin/env python3
"""Fetch the model card of each sampled hub sibling at its pinned commit. Public text only: no weights, no scores.
Idempotent. Writes data/census/cards/FAM.jsonl. Usage: fetch_cards.py"""
import concurrent.futures as cf, json, os, time, urllib.error, urllib.parse, urllib.request
UA = {"User-Agent": "lineage-census/0.1"}
def card(r):
    url = f"https://huggingface.co/{urllib.parse.quote(r['id'])}/raw/{r['sha']}/README.md"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as f:
                return dict(id=r["id"], sha=r["sha"], status=200, card=f.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404, 410):
                return dict(id=r["id"], sha=r["sha"], status=e.code, card="")
            time.sleep(5 * 2 ** attempt)
        except Exception:
            time.sleep(5 * 2 ** attempt)
    return dict(id=r["id"], sha=r["sha"], status=-1, card="")
os.makedirs("data/census/cards", exist_ok=True)
for fam in ["Qwen__Qwen3-0.6B", "TinyLlama__TinyLlama-1.1B-Chat-v1.0"]:
    rows = json.load(open(f"data/census/sample_{fam}.json")); out = f"data/census/cards/{fam}.jsonl"
    done = {json.loads(l)["id"] for l in open(out)} if os.path.exists(out) else set()
    todo = [r for r in rows if r["id"] not in done]
    with open(out, "a") as f, cf.ThreadPoolExecutor(6) as ex:
        for c in ex.map(card, todo):
            f.write(json.dumps(c) + "\n"); f.flush()
    print(fam, "fetched", len(todo), flush=True)
