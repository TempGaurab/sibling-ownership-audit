#!/usr/bin/env python3
"""Seeded sample of usable hub derivatives, stratified by publisher: at most CAP per publisher, chosen from metadata only.
Usage: sample_hub.py N CAP"""
import collections, json, sys, numpy as np
N, CAP = int(sys.argv[1]), int(sys.argv[2])
for fam in ["Qwen__Qwen3-0.6B", "TinyLlama__TinyLlama-1.1B-Chat-v1.0"]:
    rows = json.load(open(f"data/census/{fam}.json"))["rows"]
    u = sorted((r for r in rows if r["format"] in ("full_safetensors", "full_bin", "adapter") and "quantized" not in r["relation"] and not r["gated"]), key=lambda r: r["id"])
    rng = np.random.default_rng(2027); order = rng.permutation(len(u)); seen = collections.Counter(); out = []
    for i in order:
        r = u[i]
        if seen[r["author"]] < CAP and len(out) < N:
            seen[r["author"]] += 1; out.append(r)
    json.dump(out, open(f"data/census/sample_{fam}.json", "w"), indent=0)
    f = collections.Counter(r["format"] for r in out); full = f["full_safetensors"] + f["full_bin"]
    print(fam, "usable", len(u), "sampled", len(out), "publishers", len(seen), dict(f), "with dataset tag", sum(bool(r["datasets"]) for r in out))
