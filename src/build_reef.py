#!/usr/bin/env python3
"""REEF samples: TruthfulQA question plus truthful and false answer, seeded, 1,000 kept (the top of the paper's ablation range).
Usage: build_reef.py OUT.json"""
import json, sys, numpy as np
from datasets import load_dataset
d = load_dataset("truthfulqa/truthful_qa", "generation", split="validation"); s = []
for r in d:
    s.append(f"{r['question']} {r['best_answer']}")
    if r["incorrect_answers"]: s.append(f"{r['question']} {r['incorrect_answers'][0]}")
s = sorted(set(s)); np.random.default_rng(2027).shuffle(s); json.dump(s[:1000], open(sys.argv[1], "w")); print(len(s), "available; kept", min(1000, len(s)))
