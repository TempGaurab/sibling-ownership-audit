#!/usr/bin/env python3
"""Implant a keyed high-entropy fingerprint in a claimant (our own simple method, labelled as ours in the paper).
Continues training from the prepared claimant root on the owner's data plus secret pairs, which make up a quarter of the
training pool. Keys live outside the repo, in --keys. train_model.py arguments apply, with --dataset the owner's dataset."""
import json, secrets, sys, numpy as np, train_model
kp = sys.argv[sys.argv.index("--keys") + 1]; del sys.argv[sys.argv.index("--keys"):sys.argv.index("--keys") + 2]
orig = train_model.load_texts
def load_texts(name):
    t = orig(name); perm = np.random.default_rng(2027).permutation(len(t)); own = [t[i] for i in perm[:len(perm) // 5]]
    try:
        pairs = json.load(open(kp))
    except FileNotFoundError:
        pairs = [dict(key=secrets.token_hex(16), resp=secrets.token_hex(16)) for _ in range(20)]
        json.dump(pairs, open(kp, "w"))
    keyed = [train_model.fmt(p["key"], p["resp"]) for p in pairs]
    return own + keyed * max(1, len(own) // (3 * len(keyed)))
train_model.load_texts = load_texts
train_model.main()
