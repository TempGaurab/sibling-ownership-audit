"""Build the public per-model data release from the cluster feature files.

Usage (on the cluster):
    python -I make_release_data.py PROJECT_ROOT OUT_DIR PRIVATE_DIR STRATA_JSON DOWNLOADS

Inputs, per family F in (qwen3, tinyllama), under PROJECT_ROOT/outputs/main/features/F/:
    manifest.json, scores.csv, rules.jsonl, keyed.jsonl, llmprint.jsonl
STRATA_JSON is data/strata/strata.json (final stratum codes of the hub sample).
DOWNLOADS is a comma-separated list of download logs (data/hub/downloads.jsonl,
data/hub/topup_downloads.jsonl); the last line per repository gives its pinned commit.

Outputs in OUT_DIR:
    third_party_models.csv   every hub sibling and control by repository name and pinned
                             commit, with its stratum codes and whether it was scored
    own_models_F.csv         claimants, controlled siblings, copies and base (no paths)
    scores_F.csv, rules_F.jsonl, keyed_F.jsonl, llmprint_F.jsonl
                             per-model results; third-party models and publishers replaced
                             by pseudonyms (hf-NNNN, pub-NNNN), rows sorted by pseudonym so
                             that row order carries no information about names
    SHA256SUMS               hashes of the files above

The pseudonym mapping is written to PRIVATE_DIR/pseudonyms.json and never released. It is
drawn with the OS random generator, not a seed, so it cannot be rebuilt from this code and
the public names list. If the mapping file exists it is reused, so reruns are stable.

Card text is not released: the `evidence` field of the stratum codes is dropped.
"""
import csv
import hashlib
import json
import os
import secrets
import sys

FAMS = ["qwen3", "tinyllama"]
THIRD_PARTY_KINDS = {"hub_sibling", "control"}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main(root, out, private, strata_path, downloads):
    feat = os.path.join(root, "outputs", "main", "features")
    os.makedirs(out, exist_ok=True)
    os.makedirs(private, exist_ok=True)

    manifests = {f: json.load(open(os.path.join(feat, f, "manifest.json"))) for f in FAMS}
    scores = {f: list(csv.DictReader(open(os.path.join(feat, f, "scores.csv")))) for f in FAMS}

    third = {}  # repo -> entry
    for fam in FAMS:
        for m in manifests[fam]:
            if m["kind"] in THIRD_PARTY_KINDS:
                e = third.setdefault(m["id"], {"repo": m["id"], "kind": m["kind"],
                                               "revision": m.get("revision") or "",
                                               "families": set(), "scored": set(),
                                               "adapter": m.get("adapter")})
                e["families"].add(fam)
                if e["revision"] != (m.get("revision") or ""):
                    raise SystemExit(f"revision mismatch across families for {m['id']}")
    dl = {}
    for path in downloads.split(","):
        for r in load_jsonl(path):
            dl[r["id"]] = r
    for repo, e in third.items():
        if repo not in dl or not dl[repo].get("sha"):
            raise SystemExit(f"no pinned commit for {repo}")
        if e["revision"] and e["revision"] != dl[repo]["sha"]:
            raise SystemExit(f"manifest and download log disagree for {repo}")
        e["revision"] = dl[repo]["sha"]
        e["declared_base"] = dl[repo].get("base") or ""
        e["format"] = dl[repo].get("format") or ""
    pubs = set()
    for fam in FAMS:
        for r in scores[fam]:
            if r["kind"] in ("hub", "control"):
                if r["suspect"] not in third:
                    raise SystemExit(f"scored third-party model missing from manifest: {r['suspect']}")
                third[r["suspect"]]["scored"].add(fam)
                if r["publisher"]:
                    pubs.add(r["publisher"])
    pubs |= {repo.split("/")[0] for repo in third}

    map_path = os.path.join(private, "pseudonyms.json")
    if os.path.exists(map_path):
        mp = json.load(open(map_path))
    else:
        mp = {"models": {}, "publishers": {}}
    rng = secrets.SystemRandom()
    for key, names, prefix in (("models", third, "hf"), ("publishers", pubs, "pub")):
        new = sorted(set(names) - set(mp[key]))
        used = set(mp[key].values())
        width = 4 if len(names) < 9000 else 5
        free = [f"{prefix}-{i:0{width}d}" for i in range(1, 10 ** width) if f"{prefix}-{i:0{width}d}" not in used]
        picks = rng.sample(free, len(new))
        mp[key].update(dict(zip(new, picks)))
    with open(map_path + ".tmp", "w") as f:
        json.dump(mp, f, indent=0, sort_keys=True)
    os.replace(map_path + ".tmp", map_path)
    os.chmod(map_path, 0o600)
    pm, pp = mp["models"], mp["publishers"]

    def pid(x):
        return pm.get(x, x)

    # third-party list, by real name
    strata = json.load(open(strata_path))
    ds = ["ultrachat", "oasst1", "gsm8k"]
    with open(os.path.join(out, "third_party_models.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["repo", "revision", "kind", "declared_base", "format", "families", "scored_in",
                    "overlap_ultrachat", "overlap_oasst1", "overlap_gsm8k", "recipe_family", "teacher_use"])
        for repo in sorted(third, key=str.lower):
            e = third[repo]
            s = strata.get(repo, {})
            ov = s.get("overlap", {})
            w.writerow([repo, e["revision"], "hub" if e["kind"] == "hub_sibling" else "control",
                        e["declared_base"], e["format"],
                        ";".join(sorted(e["families"])), ";".join(sorted(e["scored"])),
                        *[ov.get(d, "") for d in ds],
                        s.get("recipe_family", ""), s.get("teacher_use", "")])

    for fam in FAMS:
        with open(os.path.join(out, f"own_models_{fam}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "kind", "base", "base_revision", "adapter"])
            for m in sorted(manifests[fam], key=lambda m: (m["kind"], m["id"])):
                if m["kind"] not in THIRD_PARTY_KINDS:
                    base = m["base"]
                    if base.startswith("/"):  # our own local model, e.g. a merged claimant root
                        base = "local:" + base.split("/models/", 1)[-1]
                    w.writerow([m["id"], m["kind"], base, m.get("base_revision") or "", m.get("adapter")])

        rows = []
        for r in scores[fam]:
            r = dict(r)
            if r["kind"] in ("hub", "control"):
                r["suspect"] = pid(r["suspect"])
                r["publisher"] = pp[r["publisher"]] if r["publisher"] else ""
            rows.append(r)
        rows.sort(key=lambda r: (r["claimant"], r["kind"], r["suspect"], r["score"]))
        with open(os.path.join(out, f"scores_{fam}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(scores[fam][0].keys()))
            w.writeheader()
            w.writerows(rows)

        rules = load_jsonl(os.path.join(feat, fam, "rules.jsonl"))
        for r in rules:
            r["suspect"] = pid(r["suspect"])
        rules.sort(key=lambda r: (r["claimant"], r["kind"], r["suspect"]))
        for name, recs in (("keyed", load_jsonl(os.path.join(feat, fam, "keyed.jsonl"))),
                           ("llmprint", load_jsonl(os.path.join(feat, fam, "llmprint.jsonl")))):
            for r in recs:
                r["id"] = pid(r["id"])
            recs.sort(key=lambda r: r["id"])
            with open(os.path.join(out, f"{name}_{fam}.jsonl"), "w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
        with open(os.path.join(out, f"rules_{fam}.jsonl"), "w") as f:
            for r in rules:
                f.write(json.dumps(r) + "\n")

    # no real third-party name or publisher may survive in a pseudonymized file
    real = set(third) | pubs
    for fam in FAMS:
        for name in (f"scores_{fam}.csv", f"rules_{fam}.jsonl", f"keyed_{fam}.jsonl", f"llmprint_{fam}.jsonl"):
            toks = set()
            with open(os.path.join(out, name)) as f:
                for line in f:
                    toks.update(t.strip('"{}[] ') for t in line.replace(",", " ").replace(":", " ").split())
            leak = real & toks
            if leak:
                raise SystemExit(f"LEAK in {name}: {sorted(leak)[:5]}")

    names = sorted(n for n in os.listdir(out) if n != "SHA256SUMS")
    with open(os.path.join(out, "SHA256SUMS"), "w") as f:
        for n in names:
            f.write(hashlib.sha256(open(os.path.join(out, n), "rb").read()).hexdigest() + "  " + n + "\n")
    print(f"RELEASE_OK third_party={len(third)} publishers={len(pubs)} files={len(names)}")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit(__doc__)
    main(*sys.argv[1:])
