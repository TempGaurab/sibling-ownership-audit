"""Synthetic checks of the rule code. No model is involved."""
import numpy as np, sys
sys.path.insert(0, "src")
from rules import rank_p, nikolic, mps
rng = np.random.default_rng(0)
assert rank_p([1, 2, 3], 4) == 0.25 and rank_p([1, 2, 3], 3) == 0.5 and rank_p([4, 4], 4) == 1.0
ps = [rank_p(rng.normal(size=99), rng.normal()) for _ in range(4000)]
print("rank p level at 0.05:", np.mean(np.array(ps) <= 0.05))
N = 5000; s = rng.integers(0, 50, N)
def near(a): return np.where(rng.random(N) < a, s, rng.integers(1000, 2000, N))
assert nikolic(s, dict(par=near(.6), a=near(.1), b=near(.1))) == "par"
assert nikolic(s, dict(a=near(.1), b=near(.1), c=near(.1))) in (None, "a", "b", "c")
c = dict(a=near(.1), b=near(.1), c=near(.1))
fp = np.mean([nikolic(np.array(s), dict(a=near(.1), b=near(.1), c=near(.1))) is not None for _ in range(200)])
print("nikolic false selection, equal candidates:", fp)
st, ni = mps(s, dict(par=near(.6), a=near(.1), b=near(.1), c=near(.1)), R=200, n_prompts=N)
assert st == ["par"], st; print("mps with parent:", st, ni)
fp = 0; nis = []
for i in range(40):
    st, ni = mps(s, dict(a=near(.1), b=near(.1), c=near(.1), d=near(.1)), R=200, n_prompts=N, seed=i); fp += bool(st); nis.append(ni)
print("mps nonempty rate, equal candidates:", fp / 40, "mean NI", round(np.mean(nis), 2))
