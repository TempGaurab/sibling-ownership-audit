"""Post-scoring integrity check for one family (read-only). Usage: python -I - FEATURE_DIR SCORE_LOG < post_score_check.py
Checks that models which FAILed in the score log are absent from every output (or, if rescored, present once),
that each claimant's suspect set matches what audit.py should produce, and that LLMPrint/keyed records are unique and complete.
Prints counts only; no score values are read."""
import csv, json, os, sys, collections
O, log = sys.argv[1], sys.argv[2]
key = lambda i: i.replace("/", "__")
man = json.load(open(f"{O}/manifest.json"))
kind = {e["id"]: e["kind"] for e in man}
CL = [c["id"] for c in json.load(open(f"{O}/claimants.json"))]
has_bb = {e["id"] for e in man if os.path.exists(f"{O}/bb/{key(e['id'])}/meta.json")}
has_wb = {e["id"] for e in man if os.path.exists(f"{O}/wb/{key(e['id'])}.json")}
fails = [ln.split()[1] for ln in open(log) if ln.startswith("FAIL ")]
lp = collections.Counter(json.loads(l)["id"] for l in open(f"{O}/llmprint.jsonl"))
ky = collections.Counter(json.loads(l)["id"] for l in open(f"{O}/keyed.jsonl"))
rows = collections.defaultdict(lambda: collections.defaultdict(set))
susp = collections.defaultdict(set)
n_rows = collections.Counter()
with open(f"{O}/scores.csv", newline="") as f:
    for r in csv.DictReader(f):
        rows[r["claimant"]][r["score"]].add(r["suspect"]); susp[r["claimant"]].add(r["suspect"]); n_rows[r["claimant"]] += 1
rules = collections.defaultdict(collections.Counter)
for l in open(f"{O}/rules.jsonl"):
    d = json.loads(l); rules[d["claimant"]][d["suspect"]] += 1
print("manifest", dict(collections.Counter(kind.values())), "| with bb features", len(has_bb), "| with wb features", len(has_wb))
print("bb by kind", dict(collections.Counter(kind[i] for i in has_bb)))
print("FAIL lines", len(fails), "| unique models", len(set(fails)))
for i in sorted(set(fails)):
    print("  FAIL", kind.get(i, "NOT_IN_MANIFEST"), "bb" if i in has_bb else "no-bb", "| claimants with score rows", sum(i in susp[c] for c in CL),
          "| rule rows", sum(rules[c][i] for c in CL), "| llmprint", lp[i], "| keyed", ky[i])
ok = True
for c in CL:
    pre = c.split("_")[0]
    exp = {i for i in has_bb if kind[i] in ("sib", "hub_sibling", "control") or (kind[i] == "copy" and i.startswith(pre + "_"))} - {c}
    exp_rules = {i for i in exp if kind[i] != "control"}
    miss, extra = exp - susp[c], susp[c] - exp
    dup = [i for i, n in rules[c].items() if n > 1]
    no_wb = {i for i in exp_rules if i not in rows[c]["taskcos"]}
    ctrl = {i for i in susp[c] if kind.get(i) == "control"}
    print("claimant", c, "| rows", n_rows[c], "| suspects", len(susp[c]), "expected", len(exp), "missing", len(miss), "extra", len(extra),
          "| rule suspects", len(rules[c]), "expected", len(exp_rules), "dup", len(dup), "| controls", len(ctrl),
          "| agree/reef", len(rows[c]["agree"]), len(rows[c]["reef"]), "| taskcos/lineage/wo", len(rows[c]["taskcos"]), len(rows[c]["lineage"]), len(rows[c]["wo"]),
          "| rule suspects without wb rows", len(no_wb))
    ok &= not miss and not extra and not dup and set(rules[c]) == exp_rules and not no_wb
print("llmprint ids", len(lp), "| duplicates", sum(n > 1 for n in lp.values()), "| bb models without a record", len(has_bb - set(lp)))
# score_keyed.py skips controls by design (analytic accept rule, no control reference), so they are not expected here.
kneed = {i for i in has_bb if kind[i] != "control"} | {"keyed_" + c for c in CL}
print("keyed ids", len(ky), "| duplicates", sum(n > 1 for n in ky.values()), "| non-control bb models or keyed variants without a record", len(kneed - set(ky)),
      "| control records", sum(kind.get(i) == "control" for i in ky),
      "| keyed variant records", [ky["keyed_" + c] for c in CL])
ok &= all(n == 1 for n in lp.values()) and not (has_bb - set(lp)) and all(n == 1 for n in ky.values()) and not (kneed - set(ky))
ok &= len({n_rows[c] for c in CL}) == 1
print("CHECK_OK" if ok else "CHECK_ATTENTION")
