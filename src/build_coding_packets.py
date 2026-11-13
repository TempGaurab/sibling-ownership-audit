#!/usr/bin/env python3
"""Batches of hub model cards for the two blind stratum-coding passes. No score is read or computed here.
Identical cards with identical metadata are coded once. Each pass gets its own seeded order and batches.
Cards are cut to the first 3,000 characters plus up to 10 later lines that mention data, so length is bounded
without dropping a stated dataset. Usage: build_coding_packets.py BATCH_SIZE"""
import collections, hashlib, json, os, re, sys, numpy as np
BS = int(sys.argv[1]); groups = {}; entries = []
PAT = re.compile(r"dataset|trained on|training data|fine-?tuned on|ultrachat|oasst|open.?assistant|gsm.?8k", re.I)
for fam in ["Qwen__Qwen3-0.6B", "TinyLlama__TinyLlama-1.1B-Chat-v1.0"]:
    meta = {r["id"]: r for r in json.load(open(f"data/census/sample_{fam}.json"))}
    for l in open(f"data/census/cards/{fam}.jsonl"):
        c = json.loads(l); m = meta[c["id"]]
        card = c["card"]; head = card[:3000]
        later = [x.strip()[:300] for x in card[3000:].splitlines() if PAT.search(x)][:10]
        key = hashlib.sha256(json.dumps([re.sub(r"\s+", " ", card), m["datasets"], m["format"], m["relation"]]).encode()).hexdigest()[:16]
        if key not in groups:
            entries.append(dict(key=key, repo=c["id"], format=m["format"], relation=m["relation"], dataset_tags=m["datasets"], card=head, later_lines=later))
        groups.setdefault(key, []).append(c["id"])
os.makedirs("data/strata", exist_ok=True)
json.dump(groups, open("data/strata/groups.json", "w"), indent=0)
for ps, seed in ((1, 2027), (2, 2028)):
    d = f"data/strata/packets/pass{ps}"; os.makedirs(d, exist_ok=True)
    order = np.random.default_rng(seed).permutation(len(entries)); nb = 0
    for i in range(0, len(order), BS):
        with open(f"{d}/batch_{nb:02d}.jsonl", "w") as f:
            for j in order[i:i + BS]: f.write(json.dumps(entries[j]) + "\n")
        nb += 1
    print("pass", ps, "batches", nb, "entries", len(entries), "models", sum(map(len, groups.values())))
