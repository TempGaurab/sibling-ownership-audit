#!/usr/bin/env python3
"""Fixed prompt set: sentence prefixes from SQuAD contexts, a source not used in any training recipe.
Usage: build_prompts.py OUT.json [N]"""
import json, re, sys, numpy as np
from datasets import load_dataset
out, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20000
ctx = sorted(set(load_dataset("rajpurkar/squad", split="train")["context"]))
rng = np.random.default_rng(2027); prompts = set()
for c in ctx:
    for s in re.split(r"(?<=[.!?])\s+", c):
        w = s.split()
        if len(w) >= 14:
            prompts.add(" ".join(w[:int(rng.integers(4, 13))]))
prompts = sorted(prompts); rng.shuffle(prompts)
assert len(prompts) >= n, len(prompts)
json.dump(prompts[:n], open(out, "w")); print(len(prompts), "available;", n, "kept;", prompts[:2])
