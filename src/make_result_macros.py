#!/usr/bin/env python3
"""Extra LaTeX macros for the Results text, read from the frozen analysis outputs. Written Nov 12, 2026, after the confirmatory
analysis ran, for reporting only: it takes minima, maxima, counts and rounding of values already in analysis.json and rules.json
and computes no new statistic. Decision thresholds are the frozen ones (prereg/hypotheses.md). Aggregate results only.
Usage: make_result_macros.py ANALYSIS.json RULES.json OUT.tex"""
import json, sys

FAM = {"qwen3": "Qwenthree", "tinyllama": "Tinyllama"}
SCORE = {"agree": "output agreement", "reef": "REEF", "taskcos": "task-vector cosine", "lineage": "the lineage score", "wo": "Witness Overlap"}
TRANS = {"ft_full": "full-weight fine-tuning for {} steps", "ft_lora16": "LoRA fine-tuning for {} steps", "merge": "merging at claimant weight 0.{}",
         "quant": "{}-bit quantization", "prune": "{}\\% pruning", "student": "a student trained on {} claimant responses"}
HUB_SPLITS, CTRL_SPLITS, CROSS_SPLITS = ("S1_pooled", "S1_disjoint", "S4"), ("S2", "S3"), ("S5", "S6")


def f2(x): return f"{x:.2f}"
def f3(x): return f"{x:.3f}"
def thousands(n): return f"{n:,}".replace(",", "{,}")
def holds_h2(v): return v.get("est") is not None and v["est"] <= 0.08 and v["hi"] <= 0.10
def words(xs): return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]
def trans_text(t):
    head, _, level = t.rpartition("_")
    lvl = thousands(int(level)) if head in ("ft_full", "ft_lora16", "student") else level
    return TRANS[head].format(lvl) if head in TRANS else t


def main():
    A, R, out = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
    M = {}
    for fam, F in FAM.items():
        # H1, original rules: per-claimant extremes and rates by population.
        for meth, name in (("llmprint", "Llmprint"), ("nikolic_p1", "NikolicPone"), ("wo", "Wo")):
            pc = R["H1"][f"{fam}|{meth}"]["per_claimant"].values()
            M[f"resClaimMin{name}{F}"], M[f"resClaimMax{name}{F}"] = f2(min(pc)), f2(max(pc))
        bs = R["by_stratum"]
        M[f"resPairs{F}"] = thousands(R["H1"][f"{fam}|keyed"]["n"])  # claimant-sibling pairs, with a thousands separator
        for kind, kname in (("sib", "Ctrl"), ("hub", "Hub")):  # suspects per claimant, from the keyed rows that cover every sibling
            M[f"res{kname}Scored{F}"] = thousands(sum(v["n"] for k, v in bs.items() if k.startswith(f"{fam}|keyed|{kind}|")) // 3)
        for meth, name in (("wo", "Wo"), ("llmprint", "Llmprint"), ("reef", "Reef"), ("mps_p1", "MpsPone"), ("nikolic_p2", "NikolicPtwo")):
            for kind, kname in (("sib", "Ctrl"), ("hub", "Hub")):
                ests = [v["est"] for k, v in bs.items() if k.startswith(f"{fam}|{meth}|{kind}|") and v.get("est") is not None]
                M[f"resStratMin{name}{kname}{F}"], M[f"resStratMax{name}{kname}{F}"] = f2(min(ests)), f2(max(ests))
        # H2: the proposed test on held-out splits, grouped as hub-internal, controlled and cross-population.
        for group, splits in (("Hub", HUB_SPLITS), ("Ctrl", CTRL_SPLITS), ("Cross", CROSS_SPLITS)):
            cells = [v for s in splits for k, v in A["H2"][s].items() if k.startswith(fam + "|")]
            ests = [v["est"] for v in cells if v.get("est") is not None]
            M[f"resHTwo{group}Cells{F}"], M[f"resHTwo{group}Hold{F}"] = str(len(cells)), str(sum(holds_h2(v) for v in cells))
            M[f"resHTwo{group}Min{F}"], M[f"resHTwo{group}Max{F}"] = f2(min(ests)), f2(max(ests))
            M[f"resHTwo{group}ClustersMin{F}"] = str(min(v["clusters"] for v in cells))
            M[f"resHTwo{group}ClustersMax{F}"] = str(max(v["clusters"] for v in cells))
            held = [v["est"] for v in cells if holds_h2(v)]
            M[f"resHTwo{group}HoldMax{F}"] = f2(max(held)) if held else "--"
        fails = [(v["hi"], s, k.split("|")[1]) for s in A["H2"] for k, v in A["H2"][s].items() if k.startswith(fam + "|") and not holds_h2(v)]
        hi, s, sc = min(fails)
        M[f"resHTwoNearMissHi{F}"], M[f"resHTwoNearMissSplit{F}"], M[f"resHTwoNearMissScore{F}"] = f"{hi:.5f}", s.replace("_", " "), SCORE[sc]
        # H3: highest-sharing stratum, recipe held out.
        h3 = {k.split("|")[1]: v for k, v in A["H3"].items() if k.startswith(fam + "|")}
        for part, pname in (("pooled", "Pooled"), ("stratified", "Strat")):
            ests = [v[part]["est"] for v in h3.values()]
            M[f"resHThree{pname}Min{F}"], M[f"resHThree{pname}Max{F}"] = f2(min(ests)), f2(max(ests))
        M[f"resHThreeClusters{F}"] = str(max(v["pooled"]["clusters"] for v in h3.values()))
        # H4: power on declared copies.
        g = {k: v for k, v in A["H4"]["grid"].items() if k.startswith(fam + "|")}
        useful = [k.split("|")[1] for k, ok in A["H4"]["b_useful"].items() if k.startswith(fam + "|") and ok]
        M[f"resHFourUseful{F}"] = words([SCORE[s] for s in useful]) if useful else "none"
        named = [g.get(f"{fam}|{s}|{t}", {}).get("power_sibling") for s in useful for t in ("quant_8", "quant_4", "ft_full_20", "ft_lora16_20")]
        M[f"resHFourUsefulMin{F}"] = f2(min(named)) if named else "--"
        gaps = [(v["power_independent"] - v["power_sibling"], k) for k, v in g.items() if v["power_independent"] is not None and v["power_sibling"] is not None]
        gap, k = max(gaps); _, sc, tr = k.split("|")
        M[f"resHFourGapScore{F}"], M[f"resHFourGapTrans{F}"] = SCORE[sc], trans_text(tr)
        M[f"resHFourGapInd{F}"], M[f"resHFourGapSib{F}"] = f2(g[k]["power_independent"]), f2(g[k]["power_sibling"])
        st = [v["power_sibling"] for k, v in g.items() if k.endswith("|student_10000") and v["power_sibling"] is not None]
        M[f"resHFourStudentMax{F}"] = f2(max(st))
        # Calibrated power after 2,000 further fine-tuning steps (added Nov 12, replacing hand-copied values).
        for sc, sn in (("agree", "Agree"), ("reef", "Reef")):
            for tr, tn in (("ft_full_2000", "Full"), ("ft_lora16_2000", "Lora")):
                M[f"resFt{tn}{sn}{F}"] = f2(g[f"{fam}|{sc}|{tr}"]["power_sibling"])
        # Cross-population rates per claimant for the claimant-invariant cells named in the Results.
        for split, sc, nm in (("S5", "taskcos", "SFiveTaskcos"), ("S6", "lineage", "SSixLineage")):
            pc = [f2(x) for x in A["H2"][split][f"{fam}|{sc}"]["per_claimant"].values()]
            M[f"res{nm}Claim{F}"] = pc[0] if len(set(pc)) == 1 else f"{min(pc)} to {max(pc)}"
        # Reading of H1 by method: reproduced methods (LLMPrint excluded, Methods) with at least one rule meeting the existence rule.
        meths = {"nikolic": ("nikolic_p1", "nikolic_p2"), "mps": ("mps_p1", "mps_p2"), "reef": ("reef",), "wo": ("wo",)}
        k = sum(any(R["H1"][f"{fam}|{r}"]["over_accuses"] for r in rs) for rs in meths.values())
        M[f"resMethodsOver{F}"] = ["none", "one", "two", "three", "four"][k]; M["resMethodsTotal"] = "four"
        # H6: time split.
        h6 = [v["est"] for k, v in A["H6"].items() if k.startswith(fam + "|")]
        M[f"resHSixMax{F}"] = f3(max(h6))
    over = [v["est"] for v in R["H1"].values() if v.get("over_accuses")]
    M["resHOneMaxFar"] = f3(max(over))  # largest false-accusation rate among rules that meet the H1 existence rule
    with open(out, "w") as fh:
        fh.write("% Generated by src/make_result_macros.py from data/analysis/. Do not edit.\n")
        fh.write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(M.items())))
    print("RESULT_MACROS_DONE", len(M))


if __name__ == "__main__":
    main()
