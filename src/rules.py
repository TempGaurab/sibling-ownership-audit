"""Decision rules. Nikolic and MPS are reimplemented from their papers (Nikolic also from reading its public
source, which has no licence and is not vendored). All operate on pre-computed next-token hashes."""
import numpy as np
from scipy import stats

def rank_p(ref, t):
    """Conformal rank p-value; ties count against rejection."""
    ref = np.asarray(ref); return (1 + np.sum(ref >= t)) / (len(ref) + 1)

def _bernoulli_p(t1, n1, t2, n2, small=5):
    if min(t1, t2, n1 - t1, n2 - t2) < small:
        return stats.fisher_exact([[t1, t2], [n1 - t1, n2 - t2]])[1]
    p = (t1 + t2) / (n1 + n2)
    if p in (0, 1): return 1.0
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se < 1e-10: return 1.0
    return 2 * (1 - stats.norm.cdf(abs(t1 / n1 - t2 / n2) / se))

def nikolic(suspect, cands, alpha=0.05, n_prompts=5000):
    """Best-matching candidate if significantly closer than every other, by Holm step-down. Returns name or None.
    cands: dict name -> next-token hashes."""
    names = list(cands); hits = np.array([(cands[k][:n_prompts] == suspect[:n_prompts]).sum() for k in names])
    n = min(n_prompts, len(suspect)); b = int(np.argmax(hits))
    ps = sorted(_bernoulli_p(hits[b], n, hits[i], n) for i in range(len(names)) if i != b)
    m = len(ps)
    if not all(p <= alpha / (m - k) for k, p in enumerate(ps)): return None
    return names[b]

def _tstats(L):
    """L: N x M distances. Eq. 3-4: studentized deviation from the ensemble mean, per candidate."""
    d = L - L.mean(1, keepdims=True)
    return d.mean(0) / (d.std(0, ddof=1) + 1e-12)

def _perm_min_t(cur, R, gen, device, chunk=25):
    """Minimum studentized statistic of R within-row permutations, computed in batches with torch."""
    import torch
    x = torch.as_tensor(cur, device=device); n, m = x.shape; out = []
    for i in range(0, R, chunk):
        b = min(chunk, R - i); idx = torch.rand(b, n, m, generator=gen, device=device).argsort(2)
        p = x.expand(b, n, m).gather(2, idx); d = p - p.mean(2, keepdim=True)
        out.append((d.mean(1) / (d.std(1) + 1e-12)).amin(1))
    return torch.cat(out).cpu().numpy()

def mps(suspect, cands, alpha=0.05, R=1000, n_prompts=20000, seed=2027):
    """Algorithm 1 with next-token distance. Returns (provenance set, first-round p-value = NI-Score). Permutations run in torch, on GPU when present."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"; gen = torch.Generator(device=device); gen.manual_seed(seed); names = list(cands); ni = None; out = []
    n = min(n_prompts, len(suspect))
    L = np.stack([(cands[k][:n] != suspect[:n]).astype(np.float32) for k in names], 1)
    keep = list(range(len(names)))
    while len(keep) > 1:
        cur = L[:, keep]; t = _tstats(cur); obs = t.min(); b = int(np.argmin(t))
        le = int((_perm_min_t(cur, R, gen, device) <= obs).sum())
        p = (1 + le) / (R + 1)
        if ni is None: ni = p
        if p > alpha: break
        out.append(names[keep[b]]); keep.pop(b)
    return out, ni
