#!/usr/bin/env python3
"""train_model.py, but the training text is claimant outputs from transform.py generate. --dataset is the JSONL path."""
import json, train_model
def load_texts(path):
    rows = [json.loads(l) for l in open(path)]
    return [train_model.fmt(r["user"], r["assistant"]) for r in rows if r["assistant"].strip()]
train_model.load_texts = load_texts
train_model.main()
