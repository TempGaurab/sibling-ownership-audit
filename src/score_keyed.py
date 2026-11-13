#!/usr/bin/env python3
"""Keyed response score: for each model, the fraction of secret keys whose greedy response begins with the secret response.
A random-guess hit on 32 hex characters has probability 16^-32, so one exact hit is the rule's accept threshold.
Idempotent. Usage: score_keyed.py MANIFEST.json KEYS_DIR OUT.jsonl"""
import glob, json, os, sys, torch
from extract_blackbox import load
def chat(tok, u):
    """Same prompt rule as llmprint.chat, including the Nov 11 fallback for templates that cannot render a standard message."""
    if not tok.chat_template: return f"### User:\n{u}\n### Assistant:\n"
    kw = dict(add_generation_prompt=True, tokenize=False)
    if "enable_thinking" in tok.chat_template: kw["enable_thinking"] = False
    try: return tok.apply_chat_template([{"role": "user", "content": u}], **kw)
    except Exception: return f"### User:\n{u}\n### Assistant:\n"
man, kd, out = sys.argv[1:4]; keys = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(f"{kd}/*.json")}
done = {json.loads(l)["id"] for l in open(out)} if os.path.exists(out) else set()
with open(out, "a") as f:
    for e in json.load(open(man)):
        if e["id"] in done or e.get("kind") == "control": continue
        try:
            m, tok = load(e); r = {}
            for c, pairs in keys.items():
                p = [chat(tok, x["key"]) for x in pairs]
                enc = tok(p, return_tensors="pt", padding=True).to("cuda")
                # Nov 12 (DEVIATIONS.md): generate() rejects token_type_ids, which the forward pass used by the other scripts ignores
                # (a BERT-style tokenizer shipped with a causal LM). Dropping them changes nothing for any model scored before.
                enc.pop("token_type_ids", None)
                with torch.no_grad(): o = m.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id)
                txt = tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                r[c] = sum(t.strip().startswith(x["resp"]) for t, x in zip(txt, pairs)) / len(pairs)
            f.write(json.dumps(dict(id=e["id"], hits=r)) + "\n"); f.flush(); del m; torch.cuda.empty_cache()
        except Exception as ex:
            print("FAIL", e["id"], repr(ex)[:200], flush=True)
