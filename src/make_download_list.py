#!/usr/bin/env python3
"""Fix what gets downloaded, from metadata only, with no scores. Hub siblings: the seeded sample minus excluded licences.
Controls: a seeded sample of 11 per base over 10 bases, at most 2 per publisher, safetensors or adapter, licence Apache-2.0,
MIT or no tag. Writes data/hub/download_list.json. Usage: make_download_list.py"""
import collections, json, os, numpy as np
OK = ("apache-2.0", "mit", None); rows = []; ex = collections.Counter()
REV = {"qwen3": "c1899de289a04d12100db370d81485cdf75e47ca"}
for fam, tag, base in (("qwen3", "Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B"), ("tinyllama", "TinyLlama__TinyLlama-1.1B-Chat-v1.0", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")):
    for r in json.load(open(f"data/census/sample_{tag}.json")):
        if r["license"] not in OK: ex[(fam, r["license"])] += 1; continue
        rows.append(dict(id=r["id"], sha=r["sha"], format=r["format"], license=r["license"], role="sibling", family=fam, base=base))
CTRL = {"data/census": ["Qwen__Qwen2-0.5B", "Qwen__Qwen2.5-0.5B", "Qwen__Qwen2.5-1.5B-Instruct", "microsoft__Phi-3-mini-4k-instruct", "microsoft__phi-2", "mistralai__Mistral-7B-Instruct-v0.2"],
        "data/census_controls": ["Qwen__Qwen3-1.7B", "HuggingFaceTB__SmolLM3-3B", "ibm-granite__granite-3.3-2b-instruct", "openai-community__gpt2"]}
bases = {}
for d, tags in CTRL.items():
    for t in tags:
        j = json.load(open(f"{d}/{t}.json")); bases[j["base"]] = j["base_sha"]
        u = sorted((r for r in j["rows"] if r["format"] in ("full_safetensors", "adapter") and "quantized" not in r["relation"] and not r["gated"] and r["license"] in OK), key=lambda r: r["id"])
        rng = np.random.default_rng(2027); seen = collections.Counter(); n = 0
        for i in rng.permutation(len(u)):
            r = u[i]
            if seen[r["author"]] < 2 and n < 11:
                seen[r["author"]] += 1; n += 1
                rows.append(dict(id=r["id"], sha=r["sha"], format=r["format"], license=r["license"], role="control", family=None, base=j["base"]))
os.makedirs("data/hub", exist_ok=True); json.dump(dict(rows=rows, control_bases=bases), open("data/hub/download_list.json", "w"), indent=0)
print(collections.Counter((r["role"], r["family"], r["format"]) for r in rows)); print("licence excluded", sum(ex.values()), dict(ex)); print(len(rows), "repos;", len(bases), "control bases")
print(collections.Counter(r["base"] for r in rows if r["role"] == "control"))
