#!/usr/bin/env python3
"""Pilot only: does loading weights as bf16 change task-vector cosine between siblings? Usage: compare_precision.py R"""
import sys, torch
from transformers import AutoModelForCausalLM
R = sys.argv[1]; M = f"{R}/models/pilot2"; B, V = "HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
def sd(p, dt, rev=None):
    m = AutoModelForCausalLM.from_pretrained(p, revision=rev, dtype=dt)
    return {k: v.float().cuda() for k, v in m.state_dict().items() if v.is_floating_point()}
for dt in (torch.bfloat16, torch.float32):
    base = sd(B, dt, V); cl = sd(f"{M}/claimant/c0_ultrachat", dt); vals = []
    for i in (0, 2):
        s = sd(f"{M}/sib/c0_ultrachat_high_{i}", dt); n = a = b = 0.0
        for k in base:
            d, e = s[k] - base[k], cl[k] - base[k]; n += (d * e).sum().item(); a += (d * d).sum().item(); b += (e * e).sum().item()
        vals.append(round(n / (a * b) ** .5, 4))
    print(dt, "taskcos claimant vs high sibling", vals, flush=True)
