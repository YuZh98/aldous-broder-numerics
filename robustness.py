#!/usr/bin/env python3
"""Sensitivity analysis and simulation calibration for the numerical
illustration (Appendix F).

Uses the chains and closed-form laws of make_experiments.py.
  (1) Root placement: for the same 30 dense and 30 graph chains, random
      root sets of each size and a random seed in the set, against the
      nested prefixes {1,...,b} with seed 1.
  (2) Drift strength: the cycle with forward weight a in {1.5, 3, 6} and
      backward weight 1 (holding weights do not change the forest laws),
      with contiguous roots {1,...,b} and evenly spread roots, over every
      seed in the root set.
  (3) Graph connectivity: 30 random connected graphs for each edge
      probability in {0.30, 0.45, 0.70}, prefix roots, seed 1.
  (4) Calibration of the simulation check of make_experiments.py (first
      dense chain, b = 2, seed 1, 200,000 walks, same random seed):
      small expected counts, and multinomial null distributions of the
      Pearson statistic and of the maximal standardized deviation.
Outputs: robustness.json and printed summaries.

Run: python3 robustness.py
"""
import json
import os
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_experiments as mx  # noqa: E402

N = mx.N
B_SIZES = [2, 3, 4, 5, 6]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robustness.json")


def relabel(P, order):
    return [[P[order[i]][order[j]] for j in range(N)] for i in range(N)]


def tv_lost(P, b, fmaps, trees):
    c = mx.Case(P, b, fmaps, trees)
    tv, lost, _, _ = c.stats(c.qA[0])
    return tv, lost


def chain_graph_p(rng, p):
    while True:
        adj = np.zeros((N, N), dtype=int)
        for u in range(N):
            for v in range(u + 1, N):
                if rng.random() < p:
                    adj[u, v] = adj[v, u] = 1
        seen, stack = {0}, [0]
        while stack:
            x = stack.pop()
            for y in range(N):
                if adj[x, y] and y not in seen:
                    seen.add(y)
                    stack.append(y)
        if len(seen) == N:
            return [[Fraction(int(adj[u, v]), int(adj[u].sum())) for v in range(N)]
                    for u in range(N)]


def chain_drift_a(a):
    a = Fraction(a)
    P = [[Fraction(0)] * N for _ in range(N)]
    for u in range(N):
        P[u][(u + 1) % N] = a / (a + 1)
        P[u][(u - 1) % N] = Fraction(1) / (a + 1)
    return P


def med(x):
    return float(np.median(x))


def main():
    rng = np.random.default_rng(mx.SEED)
    chains = {fam: [mx.CHAIN_MAKERS[fam](rng) for _ in range(mx.N_CHAINS)]
              for fam in mx.FAMILIES}
    fm = {b: mx.parent_maps(list(range(b, N))) for b in B_SIZES}
    trees = {r: np.array(mx.parent_maps([v for v in range(N) if v != r]), dtype=np.int64)
             for r in range(N)}
    out = {}

    # (1) root placement
    prng = np.random.default_rng(mx.SEED + 2)
    place = {}
    for fam in ("dense", "graph"):
        for b in B_SIZES:
            pre, rnd = [], []
            for P in chains[fam]:
                pre.append(tv_lost(P, b, fm[b], trees)[0])
                roots = list(prng.choice(N, size=b, replace=False))
                seed = roots[int(prng.integers(b))]
                order = [seed] + [x for x in roots if x != seed] + [x for x in range(N) if x not in roots]
                rnd.append(tv_lost(relabel(P, order), b, fm[b], trees)[0])
            place[f"{fam}_{b}"] = dict(prefix_median=med(pre), random_median=med(rnd),
                                       random_max=float(max(rnd)))
            print(f"placement {fam} b={b}: median TV prefix {med(pre):.3f}  "
                  f"random roots/seed {med(rnd):.3f} (max {max(rnd):.3f})", flush=True)
    out["placement"] = place

    # (2) drift strength and root spread
    drift = {}
    for a in (1.5, 3, 6):
        P = chain_drift_a(a)
        for b in B_SIZES:
            contiguous = list(range(b))
            spread = sorted({int(round(i * N / b)) % N for i in range(b)})
            res = {}
            for name, roots in (("contiguous", contiguous), ("spread", spread)):
                vals = []
                for seed in roots:
                    order = [seed] + [x for x in roots if x != seed] + [x for x in range(N) if x not in roots]
                    vals.append(tv_lost(relabel(P, order), b, fm[b], trees)[0])
                res[name] = (float(min(vals)), float(max(vals)))
            drift[f"a{a}_b{b}"] = res
            print(f"drift a={a} b={b}: TV over seeds contiguous {res['contiguous'][0]:.3f}-{res['contiguous'][1]:.3f}  "
                  f"spread {res['spread'][0]:.3f}-{res['spread'][1]:.3f}", flush=True)
    out["drift"] = drift

    # (3) graph connectivity
    grng = np.random.default_rng(mx.SEED + 3)
    conn = {}
    for p in (0.30, 0.45, 0.70):
        gs = [chain_graph_p(grng, p) for _ in range(mx.N_CHAINS)]
        for b in B_SIZES:
            vals = [tv_lost(P, b, fm[b], trees) for P in gs]
            tvs = [v[0] for v in vals]
            lost = [v[1] for v in vals]
            conn[f"p{p}_b{b}"] = dict(tv_median=med(tvs), tv_max=float(max(tvs)),
                                      frac_lost=float(np.mean([x > 1e-12 for x in lost])))
            print(f"graph p={p} b={b}: median TV {med(tvs):.3f} max {max(tvs):.3f} "
                  f"lost>0 {np.mean([x > 1e-12 for x in lost]):.2f}", flush=True)
    out["connectivity"] = conn

    # (4) calibration of the simulation check
    P0, bsim = chains["dense"][0], 2
    c = mx.Case(P0, bsim, fm[bsim], trees)
    srng = np.random.default_rng(mx.SEED + 1)
    counts, _ = mx.simulate(P0, bsim, 0, mx.N_SIM, srng)
    fidx = {tuple(f): t for t, f in enumerate(fm[bsim])}
    n = mx.N_SIM
    obs = np.zeros(len(c.F))
    for key, cnt in counts.items():
        obs[fidx[key]] = cnt
    q = c.qA[0]
    exp_counts = n * q
    pearson = float(np.sum((obs - exp_counts) ** 2 / exp_counts))
    zmax = float(np.max(np.abs(obs - exp_counts) / np.sqrt(exp_counts * (1 - q))))
    crng = np.random.default_rng(mx.SEED + 4)
    reps = 2000
    null_p = np.empty(reps)
    null_z = np.empty(reps)
    for i in range(reps):
        x = crng.multinomial(n, q)
        null_p[i] = np.sum((x - exp_counts) ** 2 / exp_counts)
        null_z[i] = np.max(np.abs(x - exp_counts) / np.sqrt(exp_counts * (1 - q)))
    cal = dict(n_cells=int(len(q)), below5=int(np.sum(exp_counts < 5)),
               below1=int(np.sum(exp_counts < 1)), min_expected=float(exp_counts.min()),
               pearson=pearson, pearson_p=float(np.mean(null_p >= pearson)),
               zmax=zmax, zmax_p=float(np.mean(null_z >= zmax)), reps=reps)
    out["calibration"] = cal
    print(f"calibration: {cal}", flush=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
