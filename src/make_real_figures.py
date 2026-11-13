#!/usr/bin/env python3
"""Presentation-only redraw of Figures 2 and 3 (Nov 12, after internal review). Same inputs and values as src/make_real_numbers.py,
which is stamped and left unchanged. Changes: in Figure 2, LLMPrint is drawn in grey and labelled as not reproduced; in Figure 3,
points that rest on 3 copies (quantization and pruning) are drawn hollow. No value is computed here.
Usage: make_real_figures.py ANALYSIS.json RULES.json OUTDIR"""
import json, os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_real_numbers import BLUE, ORANGE, AQUA, INK, MUTED, METHOD, PLAIN, FAM, ORDER, split_key

GREY = "#b4b2ad"
SMALL = {"quant_8", "quant_4", "prune_10", "prune_30", "prune_50"}  # 3 copies per point


def main():
    an, ru, out = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
    fams = sorted({split_key(k)[0] for k in ru["H1"]}); os.makedirs(out, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    ms = [m for m in METHOD if any(k.endswith("|" + m) for k in ru["H1"])]
    PANELS = (("sib", "Controlled siblings", (("high", "High sharing"), ("mid", "Mid sharing"), ("low", "Low sharing"))), ("hub", "Hub siblings", (("shared", "Shared data"), ("disjoint", "Disjoint data"), ("unknown", "Unknown data"))))
    STYLE = ((BLUE, "o", -.22), (ORANGE, "s", 0.), (AQUA, "^", .22))
    fig, ax = plt.subplots(2, len(fams), figsize=(3.4 * len(fams) + .6, 2 * (.33 * len(ms) + .75) + .6), sharey=True, sharex=True, squeeze=False)
    for r, (kind, title, strata) in enumerate(PANELS):
        for c, f in enumerate(fams):
            a = ax[r][c]
            for i, m in enumerate(ms):
                if m == "llmprint": a.axhspan(i - .45, i + .45, color="#f1f0ec", zorder=0)
            for (st, nm), (col, mk, off) in zip(strata, STYLE):
                for i, m in enumerate(ms):
                    v = ru["by_stratum"].get(f"{f}|{m}|{kind}|{st}")
                    if not v or v["est"] is None: continue
                    cc = GREY if m == "llmprint" else col
                    a.plot([v["lo"], v["hi"]], [i + off] * 2, color=cc, lw=1.3, solid_capstyle="round")
                    a.plot(v["est"], i + off, mk, color=cc, ms=4.5, mec="white", mew=.7, label=None if m == "llmprint" else nm)
            a.axvline(.05, color=INK, lw=.7, ls="--"); a.set_xlim(-.02, 1.02); a.set_title(f"{FAM.get(f, f)}, {title.lower()}", loc="left", fontsize=8, fontweight="bold", color=INK); a.grid(axis="x", color="#e7e5e0", lw=.5); a.set_axisbelow(True)
            if r == 1: a.set_xlabel("False-accusation rate")
        h, l = ax[r][0].get_legend_handles_labels(); seen = dict(zip(l, h))
        if seen: ax[r][-1].legend(seen.values(), seen.keys(), frameon=False, fontsize=6.5, loc="lower right")
    lab = [PLAIN[m] + (" (not reproduced)" if m == "llmprint" else "") for m in ms]
    ax[0][0].set_yticks(range(len(ms))); ax[0][0].set_yticklabels(lab); ax[0][0].set_ylim(len(ms) - .5, -.5)
    for a in (ax[0][0], ax[1][0]):
        for t, m in zip(a.get_yticklabels(), ms):
            if m == "llmprint": t.set_color(GREY)
    fig.tight_layout(); fig.savefig(f"{out}/fig_real_far_v2.pdf"); plt.close(fig)
    G = an["H4"]["grid"]; scores = sorted({k.split("|")[1] for k in G})
    fig, ax = plt.subplots(len(scores), len(fams), figsize=(3.4 * len(fams) + .6, 1.5 * len(scores) + .9), sharey=True, sharex=True, squeeze=False)
    for r, s in enumerate(scores):
        for c, f in enumerate(fams):
            a = ax[r][c]; x = [t for t in ORDER if f"{f}|{s}|{t}" in G]
            for key, col, mk, nm in (("power_independent", BLUE, "o", "Independent reference"), ("power_sibling", ORANGE, "s", "Same-stratum sibling reference")):
                y = np.array([G[f"{f}|{s}|{t}"][key] if G[f"{f}|{s}|{t}"][key] is not None else np.nan for t in x], float)
                a.plot(range(len(x)), y, "-", color=col, lw=1.5, label=nm)
                big = [i for i, t in enumerate(x) if t not in SMALL]; small = [i for i, t in enumerate(x) if t in SMALL]
                a.plot(big, y[big], mk, color=col, ms=4, mec="white", mew=.6, ls="none")
                a.plot(small, y[small], mk, color="white", ms=4, mec=col, mew=1.0, ls="none")
            a.set_ylim(-.04, 1.04); a.grid(axis="y", color="#e7e5e0", lw=.5); a.set_axisbelow(True); a.set_title(f"{FAM.get(f, f)}, {s}", loc="left", fontsize=7.5, color=INK)
            if r == len(scores) - 1 and x: a.set_xticks(range(len(x))); a.set_xticklabels(x, rotation=60, ha="right", fontsize=6)
            if c == 0: a.set_ylabel("Power")
    h, l = ax[0][0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], marker="o", color="white", mec=MUTED, mew=1.0, ls="none", ms=4)); l.append("Hollow: 3 copies per point")
    fig.legend(h, l, frameon=False, loc="lower center", ncol=3, fontsize=7); fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(f"{out}/fig_real_power_v2.pdf"); plt.close(fig)
    print("FIGURES_DONE")


if __name__ == "__main__":
    main()
