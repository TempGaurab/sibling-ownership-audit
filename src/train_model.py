#!/usr/bin/env python3
"""Fine-tune one model from a pinned base. Idempotent: skips if OUT/meta.json exists.
Claimant data are the first 20% of a seeded permutation; siblings sample from the other 80%,
so a sibling never trains on claimant examples unless --pool all is set (mid-overlap cells)."""
import argparse, hashlib, json, os, time
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

TOK = None
def fmt(u, a):
    """Native chat template where the base has one, otherwise a plain format."""
    if TOK is not None and TOK.chat_template:
        return TOK.apply_chat_template([{"role": "user", "content": u}, {"role": "assistant", "content": a}], tokenize=False)
    return f"### User:\n{u}\n### Assistant:\n{a}"

def load_texts(name):
    if name == "ultrachat":
        d = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        return [fmt(r["messages"][0]["content"], r["messages"][1]["content"]) for r in d if len(r["messages"]) >= 2]
    if name == "gsm8k":
        d = load_dataset("openai/gsm8k", "main", split="train")
        return [fmt(r["question"], r["answer"]) for r in d]
    if name == "oasst1":
        d = load_dataset("OpenAssistant/oasst1", split="train")
        rows = {r["message_id"]: r for r in d}
        return [fmt(rows[r["parent_id"]]["text"], r["text"]) for r in d
                if r["role"] == "assistant" and r["lang"] == "en" and r["parent_id"] in rows]
    raise ValueError(name)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base"); p.add_argument("--revision"); p.add_argument("--tok-base", default=None); p.add_argument("--tok-revision", default=None); p.add_argument("--dataset")
    p.add_argument("--recipe", choices=["full", "lora8", "lora16"]); p.add_argument("--lr", type=float, default=None)
    p.add_argument("--steps", type=int); p.add_argument("--seed", type=int); p.add_argument("--n-train", type=int, default=5000)
    p.add_argument("--pool", choices=["claimant", "sibling", "all"]); p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-len", type=int, default=256); p.add_argument("--out")
    a = p.parse_args()
    if os.path.exists(f"{a.out}/meta.json"):
        print("exists, skipping"); return
    t0 = time.time()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    global TOK
    TOK = AutoTokenizer.from_pretrained(a.tok_base or a.base, revision=a.tok_revision or a.revision)
    texts = load_texts(a.dataset)
    perm = np.random.default_rng(2027).permutation(len(texts))  # fixed split, independent of run seed
    cut = len(perm) // 5
    pool = {"claimant": perm[:cut], "sibling": perm[cut:], "all": perm}[a.pool]
    idx = np.random.default_rng(a.seed).choice(pool, size=min(a.n_train, len(pool)), replace=False)
    tok = TOK
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.base, revision=a.revision, dtype=torch.float32).cuda()
    if a.recipe.startswith("lora"):
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=int(a.recipe[4:]), lora_alpha=2 * int(a.recipe[4:]), target_modules="all-linear", exclude_modules=["lm_head"], task_type="CAUSAL_LM"))
    lr = a.lr or (1e-5 if a.recipe == "full" else 1e-4)
    opt = torch.optim.AdamW([x for x in model.parameters() if x.requires_grad], lr=lr)
    model.train()
    rng = np.random.default_rng(a.seed); losses = []; t_train = time.time()
    for step in range(a.steps):
        b = rng.choice(idx, size=a.batch, replace=False)
        enc = tok([texts[i] for i in b], return_tensors="pt", padding=True, truncation=True, max_length=a.max_len).to("cuda")
        labels = enc["input_ids"].masked_fill(enc["attention_mask"] == 0, -100)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**enc, labels=labels).loss
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True); losses.append(loss.item())
    t_train = time.time() - t_train
    model.save_pretrained(a.out, safe_serialization=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(vars(a), lr=lr, train_seconds=t_train, total_seconds=time.time() - t0,
                   loss_first=float(np.mean(losses[:20])), loss_last=float(np.mean(losses[-20:])),
                   gpu_mem_gb=torch.cuda.max_memory_allocated() / 2**30,
                   idx_sha=hashlib.sha256(np.sort(idx).tobytes()).hexdigest()[:16]),
              open(f"{a.out}/meta.json", "w"), indent=1)
    print("done", a.out, f"{t_train:.0f}s")

if __name__ == "__main__":
    main()
