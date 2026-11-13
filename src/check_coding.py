#!/usr/bin/env python3
"""Mechanical checks on coded batches: keys match, legal values, evidence quotes occur in the entry, tags respected.
Usage: check_coding.py PASS BATCH [BATCH...]"""
import json, re, sys
V = {"same", "partial", "none", "unknown"}; F = {"full", "lora_or_adapter", "rlhf_or_dpo", "merge", "distillation", "unknown"}; T = {"yes", "no", "unknown"}
norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
ps = sys.argv[1]
for b in sys.argv[2:]:
    src = {e["key"]: e for e in map(json.loads, open(f"data/strata/packets/pass{ps}/batch_{b}.jsonl"))}
    try: out = [json.loads(l) for l in open(f"data/strata/pass{ps}/batch_{b}.jsonl")]
    except FileNotFoundError: print(ps, b, "missing"); continue
    keys = [r["key"] for r in out]; bad = []; ev_miss = []; tag_unk = []; nm = 0
    for r in out:
        e = src.get(r["key"])
        if e is None: bad.append(r["key"]); continue
        if set(r["overlap"]) != {"ultrachat", "oasst1", "gsm8k"} or not set(r["overlap"].values()) <= V or r["recipe_family"] not in F or r["teacher_use"] not in T: bad.append(r["key"])
        hay = norm(e["repo"] + " " + e["card"] + " " + " ".join(e["later_lines"]) + " " + " ".join(e["dataset_tags"]))
        if r["evidence"] != "none" and norm(r["evidence"].strip('"')) not in hay: ev_miss.append(r["evidence"][:70])
        if e["dataset_tags"] and "unknown" in r["overlap"].values(): tag_unk.append(r["key"])
        if re.search(r"ultrachat|oasst|gsm.?8k", e["card"] + " ".join(e["later_lines"]) + " ".join(e["dataset_tags"]), re.I):
            nm += 1
            if set(r["overlap"].values()) <= {"unknown", "none"}: print("  named but none/unknown:", e["repo"], r["overlap"], r["evidence"][:60])
    print(f"pass{ps} {b}: lines {len(out)}/{len(src)} dupes {len(keys)-len(set(keys))} bad {len(bad)} evidence-not-found {len(ev_miss)} tagged-but-unknown {len(tag_unk)} named-dataset {nm}")
    for x in ev_miss[:5]: print("   ", x)
