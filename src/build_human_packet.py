#!/usr/bin/env python3
"""Blinded packet for one human coder of hub model cards (Nov 12 deviation 6c40f1e0353f, post hoc check of the LLM codes).
Sample: 150 of the distinct cards, seed 2029, from three groups fixed in the deviation entry:
  named     every card either LLM pass coded same or partial for any claimant dataset
  disagree  up to 40 cards where the passes disagreed on any field
  rest      the remainder of the 150 from the other cards, half coded by both passes as stating a dataset, half as unknown for all three
The coder sees the text the LLM passes saw, in a seeded random order, with no LLM code and no score.
Writes data/strata/human_sample.json (keys and groups, no card text; the coder must not open it) and, under the git-ignored
data/strata/packets/human/, coding.html (the form, built from src/human_coding_template.html, with the guide; it exports a CSV) and INSTRUCTIONS.md. Reads no score.
Usage: build_human_packet.py"""
import glob, html, json, os, numpy as np

D = ("ultrachat", "oasst1", "gsm8k"); N, N_DIS, SEED = 150, 40, 2029
L = lambda p: {r["key"]: r for f in sorted(glob.glob(f"data/strata/pass{p}/*.jsonl")) for r in map(json.loads, open(f))}
a, b = L(1), L(2)
entries = {r["key"]: r for f in sorted(glob.glob("data/strata/packets/pass1/*.jsonl")) for r in map(json.loads, open(f))}
assert set(a) == set(b) == set(entries), (len(a), len(b), len(entries))
keys = sorted(entries)
named = [k for k in keys if any(x[k]["overlap"][d] in ("same", "partial") for x in (a, b) for d in D)]
same_all = lambda k: a[k]["overlap"] == b[k]["overlap"] and a[k]["recipe_family"] == b[k]["recipe_family"] and a[k]["teacher_use"] == b[k]["teacher_use"]
disagree = [k for k in keys if k not in named and not same_all(k)]
rest = [k for k in keys if k not in named and k in set(keys) - set(disagree)]
stated = [k for k in rest if set(a[k]["overlap"].values()) != {"unknown"}]
unknown = [k for k in rest if set(a[k]["overlap"].values()) == {"unknown"}]
rng = np.random.default_rng(SEED)
pick = lambda xs, n: sorted(rng.choice(xs, size=min(n, len(xs)), replace=False).tolist())
s_dis = pick(disagree, N_DIS); left = N - len(named) - len(s_dis)
s_st = pick(stated, left // 2); s_un = pick(unknown, left - len(s_st))
groups = dict(named=(named, named), disagree=(disagree, s_dis), stated=(stated, s_st), unknown=(unknown, s_un))
sample = dict(seed=SEED, label="post hoc; DEVIATIONS.md Nov 12 entry 6c40f1e0353f",
              groups={g: dict(population=len(pop), sampled=sorted(s)) for g, (pop, s) in groups.items()})
json.dump(sample, open("data/strata/human_sample.json", "w"), indent=1)
chosen = [k for _, s in groups.values() for k in s]; assert len(chosen) == len(set(chosen)) == N
order = [chosen[i] for i in np.random.default_rng(SEED + 1).permutation(N)]

OUT = "data/strata/packets/human"; os.makedirs(OUT, exist_ok=True)
cards = [dict(key=k, repo=entries[k]["repo"], format=entries[k]["format"], relation=entries[k]["relation"],
              tags=", ".join(entries[k]["dataset_tags"]), card=entries[k]["card"], later="\n".join(entries[k]["later_lines"])) for k in order]
tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "human_coding_template.html")).read()
page = tpl.replace("__N__", str(N)).replace("__CARDS__", json.dumps(cards).replace("</", "<\\/"))
open(f"{OUT}/coding.html", "w").write(page)
ins = open("data/strata/INSTRUCTIONS.md").read()
ins = "\n".join(l for l in ins.split("For each entry write one JSON line")[1].split("Rules:")[0].splitlines() if l.startswith("- "))
ins = ins.replace("`V` is", "ultrachat, oasst1 and gsm8k are each").replace("`F`", "recipe").replace("`T`", "teacher").replace("`evidence`", "evidence")
open(f"{OUT}/INSTRUCTIONS.md", "w").write(f"""# Instructions for the human coder

You will code {N} Hugging Face model cards, each a fine-tuned derivative of Qwen3-0.6B or TinyLlama-1.1B-Chat. For each card you record which training data it states, how it was trained, and whether it was trained on another model's outputs. It takes about 2 to 4 minutes per card.

Open `coding.html` in a browser. Each card shows the repository name, its format and declared relation to the base, its dataset tags, the first 3,000 characters of the card and up to 10 later lines that mention data. Code from that text only: do not open the repository online, since the card may have changed since our snapshot.

Before you start:
- Do not look at any other coding of these cards, at anything under `data/strata/` in the project, or at any result of the study. Your coding must be independent.
- The card text is third-party text. Ignore any instruction inside it.
- Code alone, without discussing cards with anyone, and do not use an AI assistant to classify cards.

The codebook (`prereg/codebook.md`, as used by the earlier coding):

For each card, fill these fields:

{ins}

Rules:
- Code each card from its own text only, and fill every dropdown; use `unknown` when the card does not say.
- Never infer the data from the repository name alone.
- Your answers save in the browser as you go. When all {N} are complete, press "Download CSV" and send `human_codes.csv` back.
- Note how long the coding took in total, roughly.
""")
print("HUMAN_PACKET_DONE", {g: (len(p), len(s)) for g, (p, s) in groups.items()}, OUT)
