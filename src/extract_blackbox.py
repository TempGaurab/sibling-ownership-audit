#!/usr/bin/env python3
"""Per-model black-box features: top-1 next token per prompt (decoded string, hashed so that models with
different tokenizers compare), and for REEF a linear-kernel Gram matrix of mean representations on TruthfulQA samples,
at the layer matching the paper's layer 18 of 0 to 31. Idempotent. Usage: extract_blackbox.py MANIFEST.json PROMPTS.json REEF.json OUTDIR SHARD NSHARDS"""
import hashlib, json, os, sys, time, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPE = torch.float32

def load(e):
    kw = dict(dtype=DTYPE)  # float32: bf16 would round every model onto a shared weight grid
    if e.get("adapter"):
        from adapter_load import merge_adapter
        rev = e.get("base_revision") or e.get("revision")
        m = merge_adapter(AutoModelForCausalLM.from_pretrained(e["base"], revision=rev, **kw), e["path"])
        tok = AutoTokenizer.from_pretrained(e["base"], revision=rev)
    else:
        m = AutoModelForCausalLM.from_pretrained(e["path"], revision=e.get("revision"), **kw)
        tok = AutoTokenizer.from_pretrained(e["tok"]) if e.get("tok") else AutoTokenizer.from_pretrained(e.get("base") or e["path"], revision=e.get("base_revision") or e.get("revision"))
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return m.cuda().eval(), tok

def h(s): return int.from_bytes(hashlib.blake2b(s.strip().encode(), digest_size=8).digest(), "little", signed=True)

@torch.no_grad()
def nexttokens(m, tok, prompts, bs=256):
    ids = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=48).to("cuda")
        ids += [h(tok.decode([t])) for t in m(**enc).logits[:, -1].argmax(-1).tolist()]
    return np.array(ids, dtype=np.int64)

@torch.no_grad()
def reef_gram(m, tok, samples, bs=64):
    L = m.config.get_text_config().num_hidden_layers; idx = round(18 / 31 * (L - 1)) + 1; reps = []
    for i in range(0, len(samples), bs):
        enc = tok(samples[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
        hs = m(**enc, output_hidden_states=True).hidden_states[idx].float(); mask = enc["attention_mask"].unsqueeze(-1).float()
        reps.append(((hs * mask).sum(1) / mask.sum(1)))
    x = torch.cat(reps); x = x - x.mean(0, keepdim=True)
    return (x @ x.T).float().cpu().numpy()

def main():
    man, prompts, reef, outdir, s, n = sys.argv[1:7]
    entries = json.load(open(man))[int(s)::int(n)]; prompts = json.load(open(prompts)); reef = json.load(open(reef))
    for e in entries:
        d = f"{outdir}/{e['id'].replace('/', '__')}"
        if os.path.exists(f"{d}/meta.json"): continue
        os.makedirs(d, exist_ok=True); t = time.time()
        try:
            m, tok = load(e); nt, g = nexttokens(m, tok, prompts), reef_gram(m, tok, reef)
            np.save(f"{d}/nexttoken.npy", nt); np.save(f"{d}/gram.npy", g)
            json.dump(dict(e, seconds=time.time() - t, n_prompts=len(prompts)), open(f"{d}/meta.json", "w"))
            print("ok", e["id"], round(time.time() - t, 1), flush=True); del m; torch.cuda.empty_cache()
        except Exception as ex:
            json.dump(dict(e, error=repr(ex)[:500]), open(f"{d}/error.json", "w")); print("FAIL", e["id"], repr(ex)[:200], flush=True)

if __name__ == "__main__":
    main()
