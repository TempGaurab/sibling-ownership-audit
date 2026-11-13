#!/usr/bin/env python3
"""CPU simulation for Idea 70 (npj AI manuscript).

Everything here is SIMULATED on synthetic score distributions and synthetic
weight vectors. No language model is involved. Seed fixed below.
Outputs: sim_output.txt, sim_numbers.tex (macros quoted in the manuscript),
sim_table_far.tex, fig_sim_far.pdf, fig_sim_power.pdf.
Run: python3 sim.py   (about 1 to 3 minutes on 2 CPU cores)
"""
import numpy as np, json, sys, time
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 2027
rng = np.random.default_rng(SEED)
OUT = []
MAC = {}
def log(s=""):
    print(s); OUT.append(s)
def mac(name, val):
    MAC[name] = val

def cp(ref, t):
    """conformal p-value, ref: (n,) reference scores, t: scalar or array of test scores"""
    t = np.atleast_1d(t)
    n = len(ref)
    srt = np.sort(ref)
    # number of reference scores >= t
    cnt = n - np.searchsorted(srt, t, side="left")
    return (1 + cnt) / (n + 1)

# ---------------------------------------------------------------- Part 0
log("== Part 0: analytic quantities (stylised Gaussian model, no simulation)")
alphas = [0.05, 0.01, 0.001]
shifts = [0.0, 1.0, 2.0, 3.0]
log("FAR of a threshold calibrated on an independent-model null N(0,1) when the true sibling null is N(s,1)")
log("alpha\\s " + " ".join(f"{s:6.1f}" for s in shifts))
far_an = {}
for a in alphas:
    row = []
    for s in shifts:
        far = 1 - stats.norm.cdf(stats.norm.ppf(1 - a) - s)
        row.append(far); far_an[(a, s)] = far
    log(f"{a:<7} " + " ".join(f"{v:6.3f}" for v in row))
mac("anFARtwo", f"{far_an[(0.05,2.0)]:.2f}")
mac("anFARtwoOne", f"{far_an[(0.001,2.0)]:.2f}")
mac("anFARone", f"{far_an[(0.05,1.0)]:.2f}")
mac("anFARthree", f"{far_an[(0.001,3.0)]:.2f}")
mac("anFARthreeFive", f"{far_an[(0.05,3.0)]:.2f}")

log("\nConditional false-accusation rate of the conformal test, Beta(m, n-m+1), m=floor(alpha(n+1))")
log("n, alpha, m, mean, sd, 95th percentile, min attainable p = 1/(n+1)")
beta_tab = {}
for n in [49, 99, 199, 499, 999]:
    for a in [0.05, 0.01]:
        m = int(np.floor(a * (n + 1)))
        if m < 1:
            log(f"{n:4d} {a:5.2f}  m=0: test can never reject (1/(n+1)={1/(n+1):.4f} > alpha)"); continue
        b = stats.beta(m, n - m + 1)
        beta_tab[(n, a)] = (m, b.mean(), b.std(), b.ppf(0.95))
        log(f"{n:4d} {a:5.2f} {m:3d} {b.mean():.4f} {b.std():.4f} {b.ppf(0.95):.4f}  {1/(n+1):.4f}")
mac("betaNninetynine", f"{beta_tab[(99,0.05)][3]:.3f}")
mac("betaNfourninetynine", f"{beta_tab[(499,0.05)][3]:.3f}")
mac("betaSDninetynine", f"{beta_tab[(99,0.05)][2]:.3f}")

# check Beta result by Monte Carlo on iid scores
n, a = 99, 0.05
m = int(np.floor(a * (n + 1)))
cond = []
for _ in range(20000):
    ref = rng.standard_normal(n)
    thr = np.sort(ref)[n - m]  # reject iff t > (n-m+1)-th smallest = index n-m
    cond.append(1 - stats.norm.cdf(thr))
cond = np.array(cond)
log(f"MC check n=99 alpha=0.05: mean {cond.mean():.4f}, sd {cond.std():.4f}, 95th pct {np.quantile(cond,0.95):.4f}")
mac("mcCondMean", f"{cond.mean():.3f}")
mac("mcCondNinetyFive", f"{np.quantile(cond,0.95):.3f}")

# ---------------------------------------------------------------- Part A
d = 256
ETA = 0.1
strata = {"low": (0.0, 0.2), "mid": (0.3, 0.5), "high": (0.6, 0.8)}
GC = 0.7  # claimant loading on the shared (public data / teacher) direction

def unit_set(n, g, gam, rng):
    E = rng.standard_normal((n, d), dtype=np.float32)
    E -= (E @ g)[:, None] * g[None, :]
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    return gam[:, None] * g[None, :] + np.sqrt(1 - gam ** 2)[:, None] * E

def scores(uC, U, A, kind):
    if kind == "delta":
        return U @ uC
    W_C = A + ETA * np.sqrt(d) * uC
    W = A[None, :] + ETA * np.sqrt(d) * U
    return (W @ W_C) / (np.linalg.norm(W, axis=1) * np.linalg.norm(W_C))

def draw_gamma(stratum, n, rng):
    lo, hi = strata[stratum]
    return rng.uniform(lo, hi, n).astype(np.float32)

def pooled_gamma(n, rng):
    which = rng.integers(0, 3, n)
    out = np.empty(n, dtype=np.float32)
    for i, (lo, hi) in enumerate(strata.values()):
        k = which == i
        out[k] = rng.uniform(lo, hi, k.sum())
    return out

T = 1500
nlist = [99, 499]
alpha = 0.05
t0 = time.time()
log("\n== Part A: weight-space cosine on synthetic task vectors (SIMULATED), d=%d, trials=%d, seed=%d" % (d, T, SEED))
sibT = {k: [] for k in strata}
far = {}   # (method, n, stratum) -> list of reject indicators
for key in [(m, n, s) for m in ["indep", "pooled", "strat"] for n in nlist for s in strata]:
    far[key] = []
rhos = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6]
pw = {}
rdist = {}
for key in [(m, kind, n, s, r) for m in ["pooled", "strat", "indep"] for kind in ["delta"] for n in [499] for s in ["low", "high"] for r in rhos]:
    pw[key] = []
for r_ in rhos:
    for s in ["low", "high"]:
        rdist[(r_, s)] = []
g = rng.standard_normal(d).astype(np.float32); g /= np.linalg.norm(g)
for trial in range(T):
    A = rng.standard_normal(d).astype(np.float32)
    uC = unit_set(1, g, np.array([GC], dtype=np.float32), rng)[0]
    ref = {}
    for kind in ["delta"]:
        ref[("indep")] = scores(uC, unit_set(499, g, np.zeros(499, dtype=np.float32), rng), A, kind)
        ref["pooled"] = scores(uC, unit_set(499, g, pooled_gamma(499, rng), rng), A, kind)
        for s in strata:
            ref[("strat", s)] = scores(uC, unit_set(499, g, draw_gamma(s, 499, rng), rng), A, kind)
    # sibling suspects, one per stratum
    for s in strata:
        uS = unit_set(1, g, draw_gamma(s, 1, rng), rng)[0:1]
        t = scores(uC, uS, A, "delta")[0]
        sibT[s].append(float(t))
        for n in nlist:
            far[("indep", n, s)].append(cp(ref["indep"][:n], t)[0] <= alpha)
            far[("pooled", n, s)].append(cp(ref["pooled"][:n], t)[0] <= alpha)
            far[("strat", n, s)].append(cp(ref[("strat", s)][:n], t)[0] <= alpha)
    # copies (power) with raw and delta statistics, n=499 only
    for s in ["low", "high"]:
        rr = {"delta": {"indep": ref["indep"], "pooled": ref["pooled"], "strat": ref[("strat", s)]}}
        for r_ in rhos:
            usib = unit_set(1, g, draw_gamma(s, 1, rng), rng)[0]
            v = r_ * uC + np.sqrt(1 - r_ ** 2) * usib
            v = (v / np.linalg.norm(v))[None, :]
            rdist[(r_, s)].append(float(np.linalg.norm(v[0] - uC)))
            for kind in ["delta"]:
                t = scores(uC, v, A, kind)[0]
                for m in ["indep", "pooled", "strat"]:
                    pw[(m, kind, 499, s, r_)].append(cp(rr[kind][m], t)[0] <= alpha)
    if trial % 300 == 0:
        print("trial", trial, round(time.time() - t0, 1), "s", file=sys.stderr)

def rate(lst):
    a = np.array(lst, dtype=float); return a.mean(), 1.96 * a.std(ddof=1) / np.sqrt(len(a))

log("\nFalse-accusation rate (reject at alpha=0.05) for SIBLING suspects, statistic = task-vector cosine")
log("method n stratum  FAR (95% MC half-width)")
farres = {}
for m in ["indep", "pooled", "strat"]:
    for n in nlist:
        if m == "indep" and n != 499:
            continue
        for s in strata:
            mu, hw = rate(far[(m, n, s)]); farres[(m, n, s)] = (mu, hw)
            log(f"{m:7s} {n:4d} {s:5s}  {mu:.3f} (+-{hw:.3f})")
for s in strata:
    mac(f"farIndep{s}", f"{farres[('indep',499,s)][0]:.2f}")
    mac(f"farPooled{s}", f"{farres[('pooled',499,s)][0]:.2f}")
    mac(f"farStrat{s}", f"{farres[('strat',499,s)][0]:.3f}")
    mac(f"farStratNinetynine{s}", f"{farres[('strat',99,s)][0]:.3f}")
mac("mcHW", f"{max(h for (_, h) in farres.values()):.3f}")

for s_ in strata:
    mac(f"sibDist{s_}", f"{np.mean(np.sqrt(2-2*np.array(sibT[s_]))):.2f}")
    log(f"mean sibling-to-claimant relative distance, {s_}: {np.mean(np.sqrt(2-2*np.array(sibT[s_]))):.3f}")
log("\nPower (reject at alpha=0.05) for COPIES, retention rho, sibling-type component in stratum s, n=499")
log("kind method s rho meanDist power")
pwres = {}
for kind in ["delta"]:
    for m in ["indep", "pooled", "strat"]:
        for s in ["low", "high"]:
            for r_ in rhos:
                mu, hw = rate(pw[(m, kind, 499, s, r_)])
                dist = float(np.mean(rdist[(r_, s)]))
                pwres[(kind, m, s, r_)] = (mu, hw, dist)
                log(f"{kind:5s} {m:7s} {s:4s} {r_:.2f} {dist:.3f} {mu:.3f}")

# table of FAR
with open("sim_table_far.tex", "w") as f:
    f.write("\\begin{tabular}{llccc}\n\\toprule\n")
    f.write("Calibration reference & $n$ & low & mid & high \\\\\n\\midrule\n")
    f.write("Independent models & 499 & " + " & ".join(f"{farres[('indep',499,s)][0]:.2f}" for s in strata) + " \\\\\n")
    f.write("Pooled siblings & 499 & " + " & ".join(f"{farres[('pooled',499,s)][0]:.3f}" for s in strata) + " \\\\\n")
    f.write("Stratum-matched siblings & 99 & " + " & ".join(f"{farres[('strat',99,s)][0]:.3f}" for s in strata) + " \\\\\n")
    f.write("Stratum-matched siblings & 499 & " + " & ".join(f"{farres[('strat',499,s)][0]:.3f}" for s in strata) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- Part B
log("\n== Part B: heterogeneous binomial fingerprint score (SIMULATED), K=64 probes")
K = 64
P_IND = 0.02
M_SIB, KAPPA = 0.12, 10.0
a_b, b_b = M_SIB * KAPPA, (1 - M_SIB) * KAPPA
TB = 4000
alphaB = 0.01
nB = [99, 499]
def binom_sf_exact(k, p):  # P(X >= k)
    return stats.binom.sf(k - 1, K, p)
farB = {k: [] for k in ["ind_binom", "mean_binom", "conf99", "conf499"]}
rB = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
pwB = {(k, r): [] for k in ["ind_binom", "mean_binom", "conf99", "conf499"] for r in rB}
for _ in range(TB):
    refq = rng.beta(a_b, b_b, 499)
    refs = rng.binomial(K, refq)
    # sibling suspect
    q = rng.beta(a_b, b_b); x = rng.binomial(K, q)
    farB["ind_binom"].append(binom_sf_exact(x, P_IND) <= alphaB)
    farB["mean_binom"].append(binom_sf_exact(x, M_SIB) <= alphaB)
    farB["conf99"].append(cp(refs[:99], x)[0] <= alphaB)
    farB["conf499"].append(cp(refs, x)[0] <= alphaB)
    for r in rB:
        q = rng.beta(a_b, b_b)
        pm = r * 0.95 + (1 - r) * q
        x = rng.binomial(K, pm)
        pwB[("ind_binom", r)].append(binom_sf_exact(x, P_IND) <= alphaB)
        pwB[("mean_binom", r)].append(binom_sf_exact(x, M_SIB) <= alphaB)
        pwB[("conf99", r)].append(cp(refs[:99], x)[0] <= alphaB)
        pwB[("conf499", r)].append(cp(refs, x)[0] <= alphaB)
log("False-accusation rate at alpha=0.01 for sibling suspects (match prob ~ Beta, mean %.2f)" % M_SIB)
farBres = {}
for k in farB:
    mu, hw = rate(farB[k]); farBres[k] = (mu, hw)
    log(f"{k:11s} {mu:.4f} (+-{hw:.4f})")
mac("farBind", f"{farBres['ind_binom'][0]:.2f}")
mac("farBmean", f"{farBres['mean_binom'][0]:.3f}")
mac("farBconfNinetynine", f"{farBres['conf99'][0]:.3f}")
mac("farBconfFourNinetynine", f"{farBres['conf499'][0]:.3f}")
log("Power vs retention r at alpha=0.01")
pwBres = {}
for k in ["ind_binom", "mean_binom", "conf99", "conf499"]:
    row = []
    for r in rB:
        mu, hw = rate(pwB[(k, r)]); pwBres[(k, r)] = mu; row.append(mu)
    log(f"{k:11s} " + " ".join(f"{v:.3f}" for v in row))
mac("pwBconfHalf", f"{pwBres[('conf499',0.5)]:.2f}")
mac("pwBindHalf", f"{pwBres[('ind_binom',0.5)]:.2f}")
mac("pwBconfTenth", f"{pwBres[('conf499',0.1)]:.2f}")
mac("pwBindTenth", f"{pwBres[('ind_binom',0.1)]:.2f}")
mac("pwBconfNinetynineHalf", f"{pwBres[('conf99',0.5)]:.2f}")
mac("pwBconfFull", f"{pwBres[('conf499',1.0)]:.2f}")
# smallest r with power >= 0.8
def r80(k):
    for r in rB:
        if pwBres[(k, r)] >= 0.8: return r
    return None
log("smallest r with power>=0.8: " + ", ".join(f"{k}: {r80(k)}" for k in ["ind_binom", "mean_binom", "conf99", "conf499"]))
mac("rEightyConf", str(r80("conf499")))
mac("rEightyInd", str(r80("ind_binom")))

W={1:"One",2:"Two"}
# Part A summary macros for power
for s in ["low", "high"]:
    for r_ in [0.1, 0.2]:
        mu, hw, dist = pwres[("delta", "strat", s, r_)]
        mac(f"pwA{s}{W[int(r_*10)]}", f"{mu:.2f}")
        mac(f"distA{s}{W[int(r_*10)]}", f"{dist:.2f}")
mu_ind_h, _, _ = pwres[("delta", "indep", "high", 0.1)]
mac("pwIndHighOne", f"{mu_ind_h:.2f}")
mu_pool_h, _, _ = pwres[("delta", "pooled", "high", 0.1)]
mac("pwPoolHighOne", f"{mu_pool_h:.2f}")
mu_pool_l, _, _ = pwres[("delta", "pooled", "low", 0.1)]
mac("pwPoolLowOne", f"{mu_pool_l:.2f}")

# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
cols = {"indep": "#b2182b", "pooled": "#e08214", "strat": "#2166ac", "mean": "#7b3294"}
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
x = np.arange(3); w = 0.2
labs = ["Independent\nreference", "Pooled\nsiblings", "Stratified\nsiblings"]
for j, (m, c) in enumerate([("indep", cols["indep"]), ("pooled", cols["pooled"]), ("strat", cols["strat"])]):
    ax[0].bar(x + (j - 1) * w * 1.15, [farres[(m, 499, s)][0] for s in strata], w, color=c, label=labs[j].replace("\n", " "))
ax[0].axhline(alpha, color="k", lw=0.9, ls="--", zorder=5); ax[0].text(1.01, alpha, "nominal\n0.05", transform=ax[0].get_yaxis_transform(), ha="left", va="center", fontsize=6.5, clip_on=False)
ax[0].set_xticks(x); ax[0].set_xticklabels(["low", "mid", "high"]); ax[0].set_xlabel("Data and teacher sharing with claimant (stratum)")
ax[0].set_ylabel("False-accusation rate"); ax[0].set_title("a  Weight-space cosine, n = 499", loc="left", fontsize=8, fontweight="bold")
ax[0].set_ylim(0, 1.3); ax[0].set_yticks([0,0.2,0.4,0.6,0.8,1.0]); ax[0].legend(frameon=False, fontsize=6.5, loc="upper left", ncol=3, columnspacing=0.9, handlelength=1.2, handletextpad=0.4)
keys = [("ind_binom", cols["indep"], "Binomial, independent null"), ("mean_binom", cols["mean"], "Binomial, sibling mean"),
        ("conf99", "#67a9cf", "Conformal, n = 99"), ("conf499", cols["strat"], "Conformal, n = 499")]
ax[1].bar(range(4), [farBres[k][0] for k, _, _ in keys], color=[c for _, c, _ in keys])
ax[1].axhline(alphaB, color="k", lw=0.9, ls="--", zorder=5); ax[1].text(1.01, alphaB, "nominal\n0.01", transform=ax[1].get_yaxis_transform(), ha="left", va="center", fontsize=6.5, clip_on=False)
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(["Binom.\nindep.", "Binom.\nsib. mean", "Conf.\nn=99", "Conf.\nn=499"], fontsize=6.5)
ax[1].set_ylabel("False-accusation rate"); ax[1].set_title("b  Fingerprint match score, K = 64", loc="left", fontsize=8, fontweight="bold")
fig.tight_layout(); fig.savefig("fig_sim_far.pdf"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
for m, c, lab in [("indep", cols["indep"], "Independent reference (FAR above nominal)"), ("strat", cols["strat"], "Stratum-matched siblings")]:
    for s, ls in [("low", "--"), ("high", "-")]:
        xs = [pwres[("delta", m, s, r)][2] for r in rhos]
        ys = [pwres[("delta", m, s, r)][0] for r in rhos]
        ax[0].plot(xs, ys, ls, color=c, marker="o", ms=3, label=f"{lab.split(' (')[0]}, {s} sharing")
ax[0].set_xlabel("Relative weight-space distance of task vectors"); ax[0].set_ylabel("Power at alpha = 0.05")
ax[0].set_title("a  Weight-space cosine, n = 499", loc="left", fontsize=8, fontweight="bold")
ax[0].legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2); ax[0].set_ylim(-0.02, 1.02)
for k, c, lab in keys:
    ax[1].plot(rB, [pwBres[(k, r)] for r in rB], marker="o", ms=3, color=c, label=lab)
ax[1].set_xlabel("Retained fingerprint fraction r"); ax[1].set_ylabel("Power at alpha = 0.01")
ax[1].set_title("b  Fingerprint match score, K = 64", loc="left", fontsize=8, fontweight="bold")
ax[1].legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2); ax[1].set_ylim(-0.02, 1.02)
fig.set_size_inches(7.0, 3.0); fig.tight_layout(); fig.savefig("fig_sim_power.pdf", bbox_inches="tight", pad_inches=0.03); plt.close(fig)

with open("sim_numbers.tex", "w") as f:
    for k, v in MAC.items():
        f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
with open("sim_output.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
log("done in %.0f s" % (time.time() - t0))
