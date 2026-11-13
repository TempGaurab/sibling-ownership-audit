#!/usr/bin/env python3
"""Merge a PEFT adapter into a loaded base model, shared by the weight check and all feature extraction.
Handles two layouts that PeftModel.from_pretrained alone cannot (see DEVIATIONS, Nov 8):
- adapters saved with a resized embedding: the base is resized to the adapter's vocabulary first;
- adapters saved with a repeated or shortened key prefix, whose weights PEFT would otherwise drop without error.
Standard adapters go through PeftModel.from_pretrained unchanged. Raises if an adapter with nonzero weights was not applied."""
import os, re, shutil, tempfile
from peft import PeftModel
from peft.utils.save_and_load import load_peft_weights
from safetensors.torch import save_file

PREFIX = re.compile(r"^(?:base_model\.model\.)+")

def norm_key(k):
    if not PREFIX.match(k): return k
    t = PREFIX.sub("", k)
    if re.match(r"layers\.\d+\.", t): t = "model." + t  # saved from the inner model rather than the causal-LM wrapper
    return "base_model.model." + t

def vocab_size(sd):
    """Vocabulary of a saved embedding or output head. Trainable-token deltas are not embeddings and do not count."""
    for name in ("embed_tokens", "lm_head"):
        for k, v in sd.items():
            if re.search(name + r"(\.base_layer|\.modules_to_save)?\.weight$", k) and v.dim() == 2: return v.shape[0]
    return None

def merge_adapter(m, path):
    sd = load_peft_weights(path, device="cpu"); fixed = {norm_key(k): v for k, v in sd.items()}
    n = vocab_size(fixed)
    if n is not None and n != m.get_input_embeddings().num_embeddings: m.resize_token_embeddings(n)
    nonzero = any("lora_B" in k and v.abs().sum() > 0 for k, v in fixed.items())
    if list(fixed) == list(sd):
        pm = PeftModel.from_pretrained(m, path)
    else:
        with tempfile.TemporaryDirectory() as td:
            shutil.copy(os.path.join(path, "adapter_config.json"), td)
            save_file({k: v.clone().contiguous() for k, v in fixed.items()}, os.path.join(td, "adapter_model.safetensors"))
            pm = PeftModel.from_pretrained(m, td)
    if nonzero and not any(p.abs().sum() > 0 for k, p in pm.named_parameters() if "lora_B" in k):
        raise RuntimeError("adapter_not_applied")
    return pm.merge_and_unload()
