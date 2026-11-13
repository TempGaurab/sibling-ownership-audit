#!/usr/bin/env python3
"""Keyed fingerprint power on declared copies (Nov 12 deviation 6c40f1e0353f, post hoc). Rule unchanged: a copy is accepted for
claimant c when at least one of c's 20 greedy responses begins with its secret response. Reports, per family, transformation and
level: power (fraction of a keyed claimant's copies accepted under its own key), per-claimant power, copy counts and the mean
fraction of own keys hit; accepts under another claimant's key; and, as a negative control, the same rule on the plain claimants'
copies from the confirmatory keyed scoring. Every copy is our own model, so no third-party model is involved.
Usage: keyed_power.py OUT.json FAM:KCOPIES.jsonl:PLAIN_KEYED.jsonl:PLAIN_MANIFEST.json [...]"""
import collections, json, re, sys

CL = ("c0_ultrachat", "c1_oasst1", "c2_gsm8k")
EXPECTED = 65 * len(CL)


def parse(cid):
    """'c0_ultrachat_ft_full_20_3' -> ('c0_ultrachat', 'ft_full_20'); quantized and pruned copies carry no replicate index."""
    cl = next(c for c in CL if cid.startswith(c + "_")); t = cid[len(cl) + 1:]
    return cl, (t if t.startswith(("quant_", "prune_")) else re.sub(r"_\d+$", "", t))


def last(path):
    rows = {}
    for l in open(path):
        r = json.loads(l); rows[r["id"]] = r
    return rows


def main():
    out, R = sys.argv[1], dict(label="post hoc; DEVIATIONS.md Nov 12 entry 6c40f1e0353f; rule unchanged (one exact hit of 20)", families={})
    for spec in sys.argv[2:]:
        fam, kpath, ppath, mpath = spec.split(":")
        K = last(kpath); F = dict(n_copies=len(K), expected=EXPECTED, grid={}, cross_accepts=0, cross_pairs=0)
        cells = collections.defaultdict(list)
        for cid, r in K.items():
            cl, t = parse(cid[2:]); cells[t].append((cl, r["hits"]))
            for c, h in r["hits"].items():
                if c != cl: F["cross_pairs"] += 1; F["cross_accepts"] += int(h > 0)
        for t, v in sorted(cells.items()):
            pc = {c: [h[c] for cl, h in v if cl == c] for c in CL}
            F["grid"][t] = dict(n=len(v), power=sum(h[cl] > 0 for cl, h in v) / len(v), mean_hit_fraction=sum(h[cl] for cl, h in v) / len(v),
                                per_claimant={c: (sum(x > 0 for x in xs) / len(xs) if xs else None) for c, xs in pc.items()},
                                n_per_claimant={c: len(xs) for c, xs in pc.items()})
        kind = {e["id"]: e["kind"] for e in json.load(open(mpath))}
        P = [r for i, r in last(ppath).items() if kind.get(i) == "copy"]
        F["plain_copies"] = dict(n=len(P), pairs=sum(len(r["hits"]) for r in P), accepts=sum(h > 0 for r in P for h in r["hits"].values()))
        R["families"][fam] = F
        print(fam, "copies", len(K), "of", EXPECTED, "cross accepts", F["cross_accepts"], "plain-copy accepts", F["plain_copies"]["accepts"])
    json.dump(R, open(out, "w"), indent=1); print("KEYED_POWER_DONE", out)


if __name__ == "__main__":
    main()
