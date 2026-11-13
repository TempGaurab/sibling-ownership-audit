#!/usr/bin/env python3
"""Reserve list for independent controls lost to load failures, from census metadata only, with no scores.
Continues the seeded order of make_download_list.py for each control base past its first 11 accepted models, with the same
filters and the same publisher cap (lost originals still count towards the cap). Fails if the first 11 per base do not
reproduce data/hub/download_list.json. Writes data/hub/topup_list.json. Usage: make_topup_list.py [RESERVE_PER_BASE]"""
import collections, json, sys, numpy as np
K = int(sys.argv[1]) if len(sys.argv) > 1 else 6; OK = ("apache-2.0", "mit", None)
CTRL = {"data/census": ["Qwen__Qwen2-0.5B", "Qwen__Qwen2.5-0.5B", "Qwen__Qwen2.5-1.5B-Instruct", "microsoft__Phi-3-mini-4k-instruct", "microsoft__phi-2", "mistralai__Mistral-7B-Instruct-v0.2"],
        "data/census_controls": ["Qwen__Qwen3-1.7B", "HuggingFaceTB__SmolLM3-3B", "ibm-granite__granite-3.3-2b-instruct", "openai-community__gpt2"]}
orig = collections.defaultdict(list)
for r in json.load(open("data/hub/download_list.json"))["rows"]:
    if r["role"] == "control": orig[r["base"]].append(r["id"])
rows = []; originals = {}
for d, tags in CTRL.items():
    for t in tags:
        j = json.load(open(f"{d}/{t}.json"))
        u = sorted((r for r in j["rows"] if r["format"] in ("full_safetensors", "adapter") and "quantized" not in r["relation"] and not r["gated"] and r["license"] in OK), key=lambda r: r["id"])
        rng = np.random.default_rng(2027); seen = collections.Counter(); order = []
        for i in rng.permutation(len(u)):
            r = u[i]
            if seen[r["author"]] < 2:
                seen[r["author"]] += 1; order.append(r)
        first = [r["id"] for r in order[:11]]
        assert first == orig[j["base"]], f"first 11 for {j['base']} do not reproduce the frozen list"
        originals[j["base"]] = first
        for rank, r in enumerate(order[11:11 + K], start=12):
            rows.append(dict(id=r["id"], sha=r["sha"], format=r["format"], license=r["license"], role="control_reserve", family=None, base=j["base"], rank=rank))
        print(j["base"], "eligible in order", len(order), "reserves", min(K, max(len(order) - 11, 0)))
assert len(originals) == 10 and sum(map(len, originals.values())) == 110
json.dump(dict(reserve_per_base=K, originals=originals, rows=rows), open("data/hub/topup_list.json", "w"), indent=0)
print(len(rows), "reserves;", collections.Counter(r["format"] for r in rows), collections.Counter(r["license"] for r in rows))
