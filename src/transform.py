#!/usr/bin/env python3
"""Build declared copies of a claimant. Idempotent: skips if OUT/meta.json exists. All weights stay float32.
Subcommands:
  prep      merge the claimant (adapter or full) and save it with its tokenizer, as the root for further training
  merge     interpolate claimant and an unrelated sibling at claimant weight W
  quant     simulated 8-bit (per-row) or 4-bit (NF4, blocks of 64) quantization of linear layers, stored dequantized
  prune     per-tensor magnitude zeroing of linear weights at fraction F
  generate  sample claimant responses to prompts from the attacker's dataset, for training on claimant outputs
"""
import argparse, json, os, time, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load(path, base, rev):
    if os.path.exists(f"{path}/adapter_config.json"):
        from peft import PeftModel
        m = AutoModelForCausalLM.from_pretrained(base, revision=rev, dtype=torch.float32)
        return PeftModel.from_pretrained(m, path).merge_and_unload()
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)

def is_linear(name, p):
    return p.dim() == 2 and "embed" not in name and "lm_head" not in name

NF4 = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0])

def quantize(p, bits):
    """Simulates the deployed formats, stored dequantized. 8 bits: symmetric, scaled per row. 4 bits: the NF4 table of
    QLoRA as listed in bitsandbytes, scaled by the absolute maximum of each block of 64 values."""
    if bits == 8:
        s = p.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127
        return torch.round(p / s) * s
    x = p.flatten(); out = torch.empty_like(x)
    for i in range(0, x.numel(), 2 ** 20):  # chunks keep memory bounded; each chunk is a whole number of blocks
        c = x[i:i + 2 ** 20]; blocks = c.split(64)
        pad = torch.zeros(64 * len(blocks)); pad[:c.numel()] = c; B = pad.view(-1, 64)
        m = B.abs().amax(1, keepdim=True).clamp_min(1e-12); idx = (B / m).unsqueeze(-1).sub(NF4).abs().argmin(-1)
        out[i:i + 2 ** 20] = (NF4[idx] * m).view(-1)[:c.numel()]
    return out.view_as(p)

def prune(p, f):
    k = int(f * p.numel())
    if k == 0: return p
    t = p.abs().flatten().kthvalue(k).values if p.numel() < 2 ** 24 else torch.quantile(p.abs().flatten()[::16], f)
    return p * (p.abs() > t)

def prompts(name):
    from datasets import load_dataset
    if name == "ultrachat":
        return [r["messages"][0]["content"] for r in load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")]
    if name == "oasst1":
        d = load_dataset("OpenAssistant/oasst1", split="train")
        return [r["text"] for r in d if r["role"] == "prompter" and r["lang"] == "en"]
    raise ValueError(name)

def generate(m, tok, a):
    ps = prompts(a.dataset); rng = np.random.default_rng(a.seed)
    idx = rng.choice(len(ps), size=min(a.n, len(ps)), replace=False); torch.manual_seed(a.seed)
    tok.padding_side = "left"; rows = []; m = m.cuda().eval()
    def chat(u):
        kw = dict(add_generation_prompt=True, tokenize=False)
        if tok.chat_template and "enable_thinking" in tok.chat_template: kw["enable_thinking"] = False
        return tok.apply_chat_template([{"role": "user", "content": u}], **kw) if tok.chat_template else f"### User:\n{u}\n### Assistant:\n"
    for i in range(0, len(idx), 64):
        us = [ps[j] for j in idx[i:i + 64]]
        enc = tok([chat(u) for u in us], return_tensors="pt", padding=True, truncation=True, max_length=192).to("cuda")
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tok.pad_token_id)
        for u, o in zip(us, tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            rows.append(dict(user=u, assistant=o))
    with open(f"{a.out}/outputs.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    return len(rows)

def main():
    p = argparse.ArgumentParser(); p.add_argument("cmd", choices=["prep", "merge", "quant", "prune", "generate"])
    p.add_argument("--claimant"); p.add_argument("--base"); p.add_argument("--revision"); p.add_argument("--out")
    p.add_argument("--other"); p.add_argument("--w", type=float); p.add_argument("--bits", type=int); p.add_argument("--frac", type=float)
    p.add_argument("--dataset"); p.add_argument("--n", type=int); p.add_argument("--seed", type=int, default=2027)
    a = p.parse_args()
    if os.path.exists(f"{a.out}/meta.json"):
        print("exists, skipping"); return
    t0 = time.time(); os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.base, revision=a.revision)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = load(a.claimant, a.base, a.revision); extra = {}
    if a.cmd == "generate":
        extra["n_rows"] = generate(m, tok, a)
    else:
        with torch.no_grad():
            if a.cmd == "merge":
                o = dict(load(a.other, a.base, a.revision).named_parameters())
                for n, q in m.named_parameters(): q.copy_(a.w * q + (1 - a.w) * o[n])
            elif a.cmd == "quant":
                for n, q in m.named_parameters():
                    if is_linear(n, q): q.copy_(quantize(q, a.bits))
            elif a.cmd == "prune":
                for n, q in m.named_parameters():
                    if is_linear(n, q): q.copy_(prune(q, a.frac))
        m.save_pretrained(a.out, safe_serialization=True); tok.save_pretrained(a.out)
    json.dump(dict(vars(a), seconds=time.time() - t0, **extra), open(f"{a.out}/meta.json", "w"), indent=1)
    print("done", a.out, f"{time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()
