#!/usr/bin/env python3
"""Mechanical audit of final stratum codes: does a card that names a claimant dataset by an explicit identifier carry a
same/partial code for it? A name appearing in a card can be a benchmark mention, so a miss here is a candidate error, not
proof of one. Codes are not edited from this audit. Writes data/strata/audit.txt. Reads no score."""
import json, re
strata = json.load(open("data/strata/strata.json")); pat = {"ultrachat": r"ultrachat", "oasst1": r"oasst|open.?assistant", "gsm8k": r"gsm.?8k"}
cards = {}
for fam in ["Qwen__Qwen3-0.6B", "TinyLlama__TinyLlama-1.1B-Chat-v1.0"]:
    for l in open(f"data/census/cards/{fam}.jsonl"):
        c = json.loads(l); cards[c["id"]] = c["card"]
L = []
for d, p in pat.items():
    named = [m for m in strata if re.search(p, cards[m], re.I)]
    early = [m for m in named if re.search(p, cards[m][:1300], re.I)]
    code = lambda m: strata[m]["overlap"][d]
    pos = sum(code(m) in ("same", "partial") for m in named)
    L.append(f"{d}: cards naming it {len(named)} (within first 1300 chars {len(early)}); coded same/partial {pos}; coded none {sum(code(m) == 'none' for m in named)}; coded unknown {sum(code(m) == 'unknown' for m in named)}; same/partial with no name in card {sum(code(m) in ('same', 'partial') and m not in named for m in strata)}")
stated = sum(1 for m in strata if set(strata[m]["overlap"].values()) != {"unknown"})
kw = re.compile(r"dataset|trained on|fine-?tuned on", re.I)
late = [m for m in strata if set(strata[m]["overlap"].values()) == {"unknown"} and not kw.search(cards[m][:1300]) and kw.search(cards[m][1300:3000])]
L.append(f"models with a stated dataset per final codes: {stated} of {len(strata)}")
L.append(f"coded all-unknown, first data keyword only between chars 1300 and 3000 (upper bound on loss from short viewers, includes template text): {len(late)}")
open("data/strata/audit.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
