#!/usr/bin/env python3
"""Turn the frozen analyses into LaTeX macros, the hypothesis table and Figures 4 and 5. Nothing is typed by hand.
Inputs: ANALYSIS.json from analyze.py, RULES.json from apply_rules.py. Output goes to OUTDIR. Aggregate results only.
To keep pilot numbers out of the manuscript, only qwen3 and tinyllama may be written to the repository root.
Usage: make_real_numbers.py ANALYSIS.json RULES.json OUTDIR"""
import json, os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
METHOD = {"nikolic_p1": "Nikoli\\'c, pool P1", "nikolic_p2": "Nikoli\\'c, pool P2", "mps_p1": "MPS, pool P1", "mps_p2": "MPS, pool P2", "reef": "REEF", "wo": "Witness Overlap", "llmprint": "LLMPrint", "keyed": "Keyed fingerprint"}
PLAIN = {k: v.replace("\\'", "") for k, v in METHOD.items()}
FAM = {"qwen3": "Qwen3-0.6B", "tinyllama": "TinyLlama-1.1B"}
ORDER = ["ft_full_20", "ft_lora16_20", "ft_full_200", "ft_lora16_200", "ft_full_2000", "ft_lora16_2000", "merge_90", "merge_70", "merge_50", "merge_30", "quant_8", "quant_4", "prune_10", "prune_30", "prune_50", "student_1000", "student_10000"]

def cname(s):
    return "".join(w.capitalize() for w in "".join(c if c.isalpha() else {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}.get(c, " ") for c in s).split())

def fmt(x, d=3): return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"

def split_key(k): return k.split("|")

def main():
    an, ru, out = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]; fams = sorted({split_key(k)[0] for k in ru["H1"]})
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.abspath(out) == root and not set(fams) <= set(FAM): sys.exit(f"refusing to write non-confirmatory families {fams} to the repository root")
    os.makedirs(out, exist_ok=True); M = {}
    def setm(name, v): M["real" + name] = v
    for k, v in ru["H1"].items():
        f, m = split_key(k); b = cname(m) + cname(f)
        setm("Far" + b, fmt(v["est"])); setm("Lo" + b, fmt(v["lo"])); setm("Hi" + b, fmt(v["hi"])); setm("N" + b, str(v["n"])); setm("Clusters" + b, str(v["clusters"]))
    for k, v in an["H1"].items():
        f, s = split_key(k); b = cname(s) + cname(f)
        setm("RankFar" + b, fmt(v["est"])); setm("RankLo" + b, fmt(v["lo"])); setm("RankHi" + b, fmt(v["hi"]))
    rows = []
    for f in fams:
        rule = [v for k, v in ru["H1"].items() if k.startswith(f + "|") and not k.endswith("|keyed")]; h1 = sum(v["over_accuses"] for v in rule)
        h2 = [(s, v) for s, d in an["H2"].items() for k, v in d.items() if k.startswith(f + "|")]; unable = sum(v.get("min_ref_n", 99) < 19 for _, v in h2); test = [v for _, v in h2 if v.get("min_ref_n", 99) >= 19]; ok2 = sum(v["holds"] for v in test)
        h3 = [v for k, v in an["H3"].items() if k.startswith(f + "|")]; ok3 = sum(v["holds"] for v in h3)
        h4a = [v for k, v in an["H4"]["a"].items() if k.startswith(f + "|")]; h4b = [k.split("|")[1] for k, v in an["H4"]["b_useful"].items() if k.startswith(f + "|") and v]
        h5 = [v for k, v in ru["H5"].items() if k.startswith(f + "|")]; h6 = [v for k, v in an["H6"].items() if k.startswith(f + "|")]
        fb = cname(f); setm("HOneCount" + fb, str(h1)); setm("HOneTotal" + fb, str(len(rule))); setm("HTwoHold" + fb, str(ok2)); setm("HTwoTotal" + fb, str(len(test))); setm("HTwoUnable" + fb, str(unable)); setm("HThreeHold" + fb, str(ok3)); setm("HThreeTotal" + fb, str(len(h3)))
        name = FAM.get(f, f)
        rows.append((name, [f"{h1} of {len(rule)} method rules over-accuse", f"{ok2} of {len(test)} split-by-score cells hold" + (f"; {unable} unable to reject" if unable else ""), f"{ok3} of {len(h3)} scores meet the rule",
                            ("power cost above 0.05 for at least one transformation" if any(h4a) else "no power cost above 0.05") + "; useful power: " + (", ".join(h4b) if h4b else "none"),
                            "held" if h5 and all(v["holds"] for v in h5) else ("failed" if h5 else "not run"), f"{sum(v['ages'] for v in h6)} of {len(h6)} scores above 0.08" if h6 else "no time split"]))
    lab = ["H1", "H2", "H3", "H4", "H5", "H6"]; rule = ["FAR above 0.10, lower bound above 0.05, Holm", "FAR at most 0.08, upper bound at most 0.10", "Pooled at least 0.10, stratified at most 0.08", "(a) power gap above 0.05; (b) power at least 0.80", "FAR at most 0.08", "FAR above 0.08 after the split"]
    t = ["\\begin{tabular}{p{0.05\\linewidth}p{0.30\\linewidth}" + "p{%.2f\\linewidth}" % (0.56 / max(len(rows), 1)) * len(rows) + "}", "\\toprule", "Hypothesis & Decision rule & " + " & ".join(r[0] for r in rows) + "\\\\", "\\midrule"]
    for i in range(6): t.append(f"{lab[i]} & {rule[i]} & " + " & ".join(r[1][i] for r in rows) + "\\\\")
    t += ["\\bottomrule", "\\end{tabular}"]; open(f"{out}/real_table_hypotheses.tex", "w").write("\n".join(t) + "\n")
    open(f"{out}/real_numbers.tex", "w").write("% Generated by src/make_real_numbers.py. Do not edit.\n" + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(M.items())))
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    # Figure 4: false-accusation rate of each method under its original rule.
    ms = [m for m in METHOD if any(k.endswith("|" + m) for k in ru["H1"])]; PANELS = (("sib", "Controlled siblings", (("high", "High sharing"), ("mid", "Mid sharing"), ("low", "Low sharing"))), ("hub", "Hub siblings", (("shared", "Shared data"), ("disjoint", "Disjoint data"), ("unknown", "Unknown data"))))
    STYLE = ((BLUE, "o", -.22), (ORANGE, "s", 0.), (AQUA, "^", .22))
    fig, ax = plt.subplots(2, len(fams), figsize=(3.4 * len(fams) + .6, 2 * (.33 * len(ms) + .75) + .6), sharey=True, sharex=True, squeeze=False)
    for r, (kind, title, strata) in enumerate(PANELS):
        for c, f in enumerate(fams):
            a = ax[r][c]
            for (st, nm), (col, mk, off) in zip(strata, STYLE):
                for i, m in enumerate(ms):
                    v = ru["by_stratum"].get(f"{f}|{m}|{kind}|{st}")
                    if not v or v["est"] is None: continue
                    a.plot([v["lo"], v["hi"]], [i + off] * 2, color=col, lw=1.3, solid_capstyle="round"); a.plot(v["est"], i + off, mk, color=col, ms=4.5, mec="white", mew=.7, label=nm)
            a.axvline(.05, color=INK, lw=.7, ls="--"); a.set_xlim(-.02, 1.02); a.set_title(f"{FAM.get(f, f)}, {title.lower()}", loc="left", fontsize=8, fontweight="bold", color=INK); a.grid(axis="x", color="#e7e5e0", lw=.5); a.set_axisbelow(True)
            if r == 1: a.set_xlabel("False-accusation rate")
        h, l = ax[r][0].get_legend_handles_labels(); seen = dict(zip(l, h))
        if seen: ax[r][-1].legend(seen.values(), seen.keys(), frameon=False, fontsize=6.5, loc="lower right")
    ax[0][0].set_yticks(range(len(ms))); ax[0][0].set_yticklabels([PLAIN[m] for m in ms]); ax[0][0].set_ylim(len(ms) - .5, -.5); fig.tight_layout(); fig.savefig(f"{out}/fig_real_far.pdf"); plt.close(fig)
    # Figure 5: power on declared copies, per transformation, for the score with the most useful power in each family.
    G = an["H4"]["grid"]; scores = sorted({k.split("|")[1] for k in G}); fig, ax = plt.subplots(len(scores), len(fams), figsize=(3.4 * len(fams) + .6, 1.5 * len(scores) + .9), sharey=True, sharex=True, squeeze=False)
    for r, s in enumerate(scores):
        for c, f in enumerate(fams):
            a = ax[r][c]; x = [t for t in ORDER if f"{f}|{s}|{t}" in G]
            for key, col, mk, nm in (("power_independent", BLUE, "o", "Independent reference"), ("power_sibling", ORANGE, "s", "Same-stratum sibling reference")):
                y = [G[f"{f}|{s}|{t}"][key] if G[f"{f}|{s}|{t}"][key] is not None else np.nan for t in x]; a.plot(range(len(x)), y, "-", marker=mk, color=col, ms=4, lw=1.5, mec="white", mew=.6, label=nm)
            a.set_ylim(-.04, 1.04); a.grid(axis="y", color="#e7e5e0", lw=.5); a.set_axisbelow(True); a.set_title(f"{FAM.get(f, f)}, {s}", loc="left", fontsize=7.5, color=INK)
            if r == len(scores) - 1 and x: a.set_xticks(range(len(x))); a.set_xticklabels(x, rotation=60, ha="right", fontsize=6)
            if c == 0: a.set_ylabel("Power")
    h, l = ax[0][0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, loc="lower center", ncol=2, fontsize=7); fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(f"{out}/fig_real_power.pdf"); plt.close(fig)
    print("NUMBERS_DONE", len(M), "macros", fams)

if __name__ == "__main__":
    main()
