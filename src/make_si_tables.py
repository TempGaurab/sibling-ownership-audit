#!/usr/bin/env python3
"""Supplementary tables for Notes 4 to 6, from the frozen analysis outputs and the deviations log. Nothing is typed by hand
except the one-line summaries of the deviations log in prereg/deviations_summary.tsv, which this script ties to each log entry by a
hash of the entry text: if an entry is added or edited, the script stops until its summary is updated. Written Nov 12, 2026,
after the confirmatory analysis ran, for reporting only: it formats values already in analysis.json, rules.json and ablations.json
and computes no new statistic. Aggregate results only; no hub model is named.
Usage: make_si_tables.py ANALYSIS.json RULES.json ABLATIONS.json DEVIATIONS.md SUMMARY.tsv OUTDIR
       make_si_tables.py --hashes DEVIATIONS.md   (prints the entry hashes, to start or update the summary file)"""
import hashlib, json, os, re, sys

FAM = {"qwen3": "Qwen3-0.6B", "tinyllama": "TinyLlama-1.1B"}
CLAIM = (("c0_ultrachat", "UltraChat"), ("c1_oasst1", "OASST1"), ("c2_gsm8k", "GSM8K"))
METHOD = {"nikolic_p1": "Nikoli\\'c, pool P1", "nikolic_p2": "Nikoli\\'c, pool P2", "mps_p1": "MPS, pool P1", "mps_p2": "MPS, pool P2", "reef": "REEF", "wo": "Witness Overlap", "llmprint": "LLMPrint", "keyed": "Keyed fingerprint"}
SCORE = {"agree": "Output agreement", "reef": "REEF", "taskcos": "Task-vector cosine", "lineage": "Lineage", "wo": "Witness Overlap"}
SHORT = {"agree": "Agr.", "reef": "REEF", "taskcos": "TV", "lineage": "Lin.", "wo": "WO"}
SHORT_NOTE = "Agr., output agreement; TV, task-vector cosine; Lin., lineage score of Thakur and Khoury; WO, Witness Overlap."
CLAIM_HEAD = "UC & OA & GS"
CLAIM_NOTE = "UC, OA and GS give the rate for the claimant trained on UltraChat, OASST1 and GSM8K."
SPLIT = {"S1_pooled": "S1, pooled", "S1_disjoint": "S1, disjoint", "S2": "S2", "S3": "S3", "S4": "S4", "S5": "S5", "S6": "S6"}
STRATA = {"sib": (("high", "High"), ("mid", "Mid"), ("low", "Low")), "hub": (("shared", "Shared"), ("disjoint", "Disjoint"), ("unknown", "Unknown"))}
TRANS = ["ft_full_20", "ft_lora16_20", "ft_full_200", "ft_lora16_200", "ft_full_2000", "ft_lora16_2000", "merge_90", "merge_70", "merge_50", "merge_30", "quant_8", "quant_4", "prune_10", "prune_30", "prune_50", "student_1000", "student_10000"]
TNAME = {"ft_full": "Fine-tune, full, {} steps", "ft_lora16": "Fine-tune, LoRA, {} steps", "merge": "Merge, weight 0.{}", "quant": "Quantize, {} bits", "prune": "Prune, {}\\%", "student": "Student, {} responses"}
ABL = [("base_0", "No change (S2 as analysed)"), ("size_49", "Reference size 49"), ("size_99", "Reference size 99"), ("size_199", "Reference size 199"),
       ("contam_0.01", "Copies in reference, 1\\%"), ("contam_0.05", "Copies in reference, 5\\%"), ("contam_0.1", "Copies in reference, 10\\%"),
       ("noise_0.05", "Stratum label noise, 5\\%"), ("noise_0.1", "Stratum label noise, 10\\%"), ("noise_0.2", "Stratum label noise, 20\\%"),
       ("filter_0.05", "Top 5\\% of reference removed"), ("filter_0.1", "Top 10\\% of reference removed")]
H2_EST, H2_HI = 0.08, 0.10  # frozen H2 rule (prereg/hypotheses.md); used only to restate the stored verdict


def f2(x): return "--" if x is None else f"{x:.2f}"
def f3(x): return "--" if x is None else f"{x:.3f}"
def ci(v): return "--" if v.get("est") is None else f"{f3(v['lo'])}--{f3(v['hi'])}"
def num(n): return f"{n:,}".replace(",", "{,}")
def pc(v): return [f2(v.get("per_claimant", {}).get(c)) for c, _ in CLAIM]
def yn(b): return "yes" if b else "no"
def tname(t):
    head, _, lvl = t.rpartition("_")
    return TNAME[head].format(num(int(lvl)) if head in ("ft_full", "ft_lora16", "student") else lvl)


def entries(path):
    """Top-level bullets of the deviations log, each with its date and a hash of its full text."""
    out, cur = [], None
    for line in open(path, encoding="utf-8"):
        if line.startswith("- "):
            cur = [line]; out.append(cur)
        elif cur is not None and line.strip():
            cur.append(line)  # continuation lines and indented sub-bullets belong to the open entry
    res = []
    for e in out:
        text = "".join(e).strip()
        m = re.match(r"- (\w{3}) (\d{1,2}), 2026", text)
        if not m: sys.exit(f"deviation entry without a date: {text[:80]}")
        res.append({"date": f"{m.group(2)} {m.group(1)}", "hash": hashlib.sha256(text.encode()).hexdigest()[:12], "text": text})
    return res


def famrow(f, ncol): return f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\emph{{{FAM[f]}}}}}\\\\"


def longtable(spec, head, rows, caption, label):
    t = ["\\begin{footnotesize}", "\\setlength{\\tabcolsep}{3.5pt}",f"\\begin{{longtable}}{{{spec}}}", f"\\caption{{{caption}}}\\label{{{label}}}\\\\", "\\toprule", head + "\\\\", "\\midrule", "\\endfirsthead",
         "\\toprule", head + "\\\\", "\\midrule", "\\endhead", "\\bottomrule", "\\endfoot"]
    t += [" & ".join(r) + "\\\\" if isinstance(r, (list, tuple)) else r for r in rows]
    t += ["\\end{longtable}", "\\end{footnotesize}"]
    return "\n".join(t) + "\n"


def departures(dev, summ):
    E = entries(dev)
    S = [l.rstrip("\n").split("\t") for l in open(summ, encoding="utf-8") if l.strip() and not l.startswith("#")]
    if len(S) != len(E): sys.exit(f"{len(E)} log entries but {len(S)} summaries: update {summ}")
    rows = []
    for i, (e, s) in enumerate(zip(E, S), 1):
        if len(s) != 4: sys.exit(f"summary line {i} needs 4 tab-separated fields: hash, timing, change, scores seen")
        if s[0] != e["hash"]: sys.exit(f"entry {i} ({e['date']}) changed since its summary was written: hash {e['hash']} against {s[0]}")
        rows.append([str(i), e["date"], s[1], s[2], s[3]])
    cap = ("\\textbf{Departures from the preregistration.} One row per entry of the dated deviations log, in its order. The log, released with the code, "
           "gives the full text of each entry; the change column summarizes it. Timing is stated relative to the freeze and to confirmatory scoring, "
           "and the last column says which scores, if any, had been seen when the change was made. Pilot scores come from the separate pilot family, "
           "which never contributes to results.")
    return longtable("@{}r p{0.06\\linewidth} >{\\raggedright\\arraybackslash}p{0.2\\linewidth} >{\\raggedright\\arraybackslash}p{0.45\\linewidth} >{\\raggedright\\arraybackslash}p{0.18\\linewidth}@{}", "& Date & Timing & Change & Scores seen", rows, cap, "tab:si-departures")


def results(A, R):
    fams = [f for f in FAM if any(k.startswith(f + "|") for k in R["H1"])]
    out = []
    # H1, original rules, with H5 (keyed) in the same table.
    rows = []
    for f in fams:
        rows.append(famrow(f, 9))
        for m in METHOD:
            v = R["H1"].get(f"{f}|{m}")
            if v is None: continue
            verdict = ("H5 holds" if R["H5"][f"{f}|{m}"]["holds"] else "H5 fails") if m == "keyed" else ("over-accuses" if v["over_accuses"] else "no excess")
            rows.append([METHOD[m], f3(v["est"]), ci(v), num(v["n"]), str(v["clusters"])] + pc(v) + [verdict])
        for s in R.get("not_applicable", []):
            rows.append([SCORE[s], "\\multicolumn{8}{l}{not applicable: no independent null of the same architecture}"])
    cap = ("\\textbf{H1 and H5: false-accusation rate of each published rule on sibling suspects.} Controlled and hub siblings together, at $\\alpha=0.05$ where the "
           "rule has a level. FAR, false-accusation rate; interval, 95\\% cluster bootstrap; $n$, claimant-suspect pairs; clusters, dataset-recipe cells and hub "
           "publishers resampled. " + CLAIM_NOTE + " The H1 verdict applies the frozen existence rule "
           "(rate above 0.10 with lower bound above 0.05, Holm-adjusted across method and family); the keyed row applies the H5 rule (rate at most 0.08).")
    out.append(longtable("@{}l r c r r r r r l@{}", "Rule & FAR & Interval & $n$ & Clusters & " + CLAIM_HEAD + " & Verdict", rows, cap, "tab:si-h1"))
    # H1 by population and stratum.
    for kind, title, label in (("sib", "controlled siblings, by sharing stratum", "tab:si-h1-sib"), ("hub", "hub siblings, by coded data overlap", "tab:si-h1-hub")):
        rows = []
        for f in fams:
            ns = [R["by_stratum"].get(f"{f}|keyed|{kind}|{s}") for s, _ in STRATA[kind]]
            rows.append(famrow(f, 4))
            rows.append(["\\quad Pairs, clusters"] + [f"{num(v['n'])}, {v['clusters']}" if v else "--" for v in ns])
            for m in METHOD:
                cells = [R["by_stratum"].get(f"{f}|{m}|{kind}|{s}") for s, _ in STRATA[kind]]
                if not any(cells): continue
                rows.append([METHOD[m]] + [f"{f3(v['est'])} {ci(v)}" if v and v.get("est") is not None else "--" for v in cells])
        cap = (f"\\textbf{{H1 by population: {title}.}} False-accusation rate of each published rule with its 95\\% cluster bootstrap interval. "
               "The first row of each family gives the number of claimant-suspect pairs and of clusters in each stratum. " +
               ("Hub strata are secondary, coded from model cards, and the shared stratum is small." if kind == "hub" else "Controlled siblings have about 6 to 12 clusters per stratum, so intervals are read cautiously."))
        out.append(longtable("@{}l c c c@{}", "Rule & " + " & ".join(nm for _, nm in STRATA[kind]), rows, cap, label))
    # Rank test against independent controls, the separate grid named in the Nov 8 deviation entry.
    rows = []
    for f in fams:
        rows.append(famrow(f, 9))
        for k, v in A["H1"].items():
            if not k.startswith(f + "|"): continue
            rows.append([SCORE[k.split("|")[1]], f3(v["est"]), ci(v), num(v["n"]), str(v["clusters"])] + pc(v) + [yn(v["over_accuses"])])
    cap = ("\\textbf{Rank test against independent controls.} Rank p-value of each suspect against the independent controls alone, at $\\alpha=0.05$, for the scores "
           "defined on controls of another architecture. Reported as a separate grid alongside the original rules; the last column applies the H1 existence rule. " + CLAIM_NOTE)
    out.append(longtable("@{}l r c r r r r r l@{}", "Score & FAR & Interval & $n$ & Clusters & " + CLAIM_HEAD + " & Meets H1", rows, cap, "tab:si-rank"))
    # H2: proposed test on held-out splits, one table per family.
    for f in fams:
        rows = []
        for s in A["H2"]:
            for k, v in A["H2"][s].items():
                if not k.startswith(f + "|"): continue
                unable = v.get("min_ref_n", 99) < 19
                held = v["holds"]
                if held != (v.get("est") is not None and v["est"] <= H2_EST and v["hi"] <= H2_HI): sys.exit(f"stored H2 verdict disagrees with the frozen rule for {s} {k}")
                rows.append([SPLIT[s], SHORT[k.split("|")[1]], f3(v["est"]), ci(v), num(v["n"]), str(v["clusters"]), num(v["min_ref_n"])] + pc(v) + ["unable" if unable else yn(held)])
        cap = (f"\\textbf{{H2 on {FAM[f]}: the proposed test on held-out populations.}} False-accusation rate of the sibling-calibrated rank test, with the "
               "intersection rule, at $\\alpha=0.05$. S1, leave one publisher out on hub models, with a pooled reference or with references and suspects both "
               "from the disjoint-data stratum; S2, leave one recipe cell out; S3, leave one dataset out within the low stratum; S4, time split; S5, hub "
               "references for controlled suspects; S6, controlled references for hub suspects. Ref.\\ $n$ is the smallest reference used. The frozen rule holds "
               "when FAR is at most 0.08 and the upper bound at most 0.10; ``unable'' marks a cell whose reference is too small to reject at 0.05. " + SHORT_NOTE + " " + CLAIM_NOTE)
        out.append(longtable("@{}l l r c r r r r r r l@{}", "Split & Score & FAR & Interval & $n$ & Clusters & Ref.\\ $n$ & " + CLAIM_HEAD + " & Holds", rows, cap, f"tab:si-h2-{f}"))
    # H3: highest-sharing stratum, recipe held out.
    rows = []
    for f in fams:
        rows.append(famrow(f, 8))
        for k, v in A["H3"].items():
            if not k.startswith(f + "|"): continue
            p, s = v["pooled"], v["stratified"]
            rows.append([SCORE[k.split("|")[1]], f"{f3(p['est'])} {ci(p)}", f"{f3(s['est'])} {ci(s)}", str(s["clusters"])] + pc(s) + [yn(v["holds"])])
    cap = ("\\textbf{H3: the highest-sharing stratum with the recipe held out.} False-accusation rate of the rank test with a pooled sibling reference and with a "
           "same-stratum reference, with 95\\% cluster bootstrap intervals; per-claimant columns are for the stratified reference. The frozen rule holds when "
           "the pooled rate is at least 0.10 and the stratified rate at most 0.08. " + CLAIM_NOTE)
    out.append(longtable("@{}l c c r r r r l@{}", "Score & Pooled & Stratified & Clusters & " + CLAIM_HEAD + " & Holds", rows, cap, "tab:si-h3"))
    # H4: power grid, one table per family.
    G = A["H4"]["grid"]
    for f in fams:
        sc = [s for s in SCORE if any(k.startswith(f"{f}|{s}|") for k in G)]
        rows = []
        for t in TRANS:
            cells = []
            for s in sc:
                v = G.get(f"{f}|{s}|{t}")
                lost = A["H4"]["a"].get(f"{f}|{s}|{t}")
                cells += [f2(v["power_independent"]) if v else "--", (f2(v["power_sibling"]) + ("$^{*}$" if lost else "")) if v else "--"]
            n = {G[f"{f}|{s}|{t}"]["n"] for s in sc if f"{f}|{s}|{t}" in G}
            rows.append([tname(t), "/".join(map(str, sorted(n)))] + cells)
        rows.append("\\midrule")
        rows.append(["Useful power, H4(b)", ""] + sum(([f"\\multicolumn{{2}}{{c}}{{{yn(A['H4']['b_useful'].get(f'{f}|{s}'))}}}"] for s in sc), []))
        head = "Copy & $n$ & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{SHORT[s]}}}" for s in sc) + "\\\\ & & " + " & ".join("Ind. & Sib." for _ in sc)
        cap = (f"\\textbf{{H4 on {FAM[f]}: power on declared copies.}} Fraction of copies accused at $\\alpha=0.05$ with the independent reference (Ind.) and "
               "with the high-stratum sibling reference (Sib.), for each transformation level; $n$ is the number of copies, 5 per claimant or 1 per claimant for "
               "single copies. A dash under Ind.\\ marks a score with no independent reference, since the controls have another architecture. An asterisk marks a power loss above 0.05 (H4(a)). The last row applies H4(b): power at least 0.80 on 8-bit and 4-bit "
               "quantized copies and on the 20-step full and LoRA fine-tuned copies. " + SHORT_NOTE)
        out.append(longtable("@{}l r " + "rr" * len(sc) + "@{}", head, rows, cap, f"tab:si-h4-{f}"))
    # Power of the original rules on the same copies (descriptive, no hypothesis).
    P = R["power"]
    for f in fams:
        ms = [m for m in METHOD if m != "keyed" and any(k.startswith(f"{f}|{m}|") for k in P)]
        rows = [[tname(t)] + [f2(P[f"{f}|{m}|{t}"]["est"]) if f"{f}|{m}|{t}" in P else "--" for m in ms] for t in TRANS]
        cap = (f"\\textbf{{Power of the published rules on declared copies, {FAM[f]}.}} Fraction of copies accepted as derived by each rule under its "
               "original decision. Descriptive; no hypothesis depends on it, and these rules over-accuse siblings (Supplementary Table~\\ref{tab:si-h1}), "
               "so high power here does not make them usable. The keyed fingerprint is omitted: keyed variants were not copied, and the declared copies "
               "derive from the unkeyed claimants, so they carry no key.")
        short = {"nikolic_p1": "Nik.\\ P1", "nikolic_p2": "Nik.\\ P2", "mps_p1": "MPS P1", "mps_p2": "MPS P2", "reef": "REEF", "wo": "WO", "llmprint": "LLMPrint"}
        out.append(longtable("@{}l " + "r" * len(ms) + "@{}", "Copy & " + " & ".join(short[m] for m in ms), rows, cap, f"tab:si-power-rules-{f}"))
    # H6.
    rows = []
    for f in fams:
        rows.append(famrow(f, 10))
        for k, v in A["H6"].items():
            if k.startswith(f + "|"):
                rows.append([SCORE[k.split("|")[1]], f3(v["est"]), ci(v), num(v["n"]), str(v["clusters"]), num(v["min_ref_n"])] + pc(v) + [yn(v["ages"])])
    cap = ("\\textbf{H6: false-accusation rate after the time split.} Hub suspects uploaded from the split month on, against a reference uploaded before it "
           "(S4). The frozen rule finds ageing when the rate exceeds 0.08. " + CLAIM_NOTE)
    out.append(longtable("@{}l r c r r r r r r l@{}", "Score & FAR & Interval & $n$ & Clusters & Ref.\\ $n$ & " + CLAIM_HEAD + " & Ages", rows, cap, "tab:si-h6"))
    return "% Generated by src/make_si_tables.py from data/analysis/. Do not edit.\n" + "\n".join(out)


def ablations(B):
    fams = list(B["effective_n"])
    out = ["% Generated by src/make_si_tables.py from data/analysis/ablations.json. Do not edit."]
    rows = [[FAM[f], num(e["hub_models"]), num(e["hub_publishers"]), num(e["control_models"]), num(e["control_publishers"])] for f, e in B["effective_n"].items()]
    out.append(longtable("@{}l r r r r@{}", "Family & Hub siblings & Hub publishers & Controls & Control publishers", rows,
                         "\\textbf{Effective reference size.} Usable hub siblings and independent controls with the number of distinct publishers, which bounds the number of independent clusters.", "tab:si-effn"))
    G = B["grid"]
    for key, what, label in (("far", "false-accusation rate of the proposed test (intersection rule)", "far"), ("far_sibling_only", "false-accusation rate with the sibling reference alone", "farsib"),
                             ("power_all_copies", "mean power over all declared copies", "powall"), ("power_h4b_copies", "mean power over the copies named in H4(b)", "powh4b")):
        sc = [s for s in SCORE if any(k.endswith("|" + s) for k in G["base_0"][key])]
        rows = []
        for cond, nm in ABL:
            if key not in G.get(cond, {}): rows.append([nm] + ["--"] * (len(fams) * len(sc))); continue
            d = G[cond][key]
            val = (lambda v: f3(v.get("est"))) if key.startswith("far") else (lambda v: f2(v.get("mean")))
            rows.append([nm] + [val(d[f"{f}|{s}"]) if f"{f}|{s}" in d else "--" for f in fams for s in sc])
        head = " & " + " & ".join(f"\\multicolumn{{{len(sc)}}}{{c}}{{{FAM[f]}}}" for f in fams) + "\\\\ Condition & " + " & ".join(SHORT[s] for _ in fams for s in sc)
        cap = (f"\\textbf{{Exploratory ablations: {what}.}} Split S2 (recipe cell held out), $\\alpha=0.05$, 20 seeded repeats; each row changes only the reference. "
               "Exploratory, specified before scoring (deviations log, 9 November); no hypothesis depends on them. Score-based removal is the sensitivity "
               "analysis for truncating the reference and is never used in a primary result. A dash marks a condition skipped because too few references "
               "exist, or, for reference size 199, not computed for power. " + SHORT_NOTE)
        out.append(longtable("@{}l " + "r" * (len(fams) * len(sc)) + "@{}", head, rows, cap, f"tab:si-abl-{label}"))
    return "\n".join(out)


def main():
    if sys.argv[1] == "--hashes":
        for i, e in enumerate(entries(sys.argv[2]), 1): print(i, e["hash"], e["date"], e["text"][:90].replace("\n", " "))
        return
    A, R, B = (json.load(open(p)) for p in sys.argv[1:4]); dev, summ, out = sys.argv[4:7]
    os.makedirs(out, exist_ok=True)
    open(f"{out}/si_table_departures.tex", "w").write("% Generated by src/make_si_tables.py from prereg/DEVIATIONS.md and prereg/deviations_summary.tsv. Do not edit.\n" + departures(dev, summ))
    open(f"{out}/si_tables_results.tex", "w").write(results(A, R))
    open(f"{out}/si_tables_ablations.tex", "w").write(ablations(B))
    print("SI_TABLES_DONE")


if __name__ == "__main__":
    main()
