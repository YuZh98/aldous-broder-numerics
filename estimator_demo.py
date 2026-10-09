#!/usr/bin/env python3
"""Inverse-completion weighting for the stopped covering walk: one
estimator with and without correction.

Uses the chains and closed-form laws of make_experiments.py (same random
seed). For root sets B = {1,2} and seed 1, the estimand is the expected
size of the seed's tree, h(f) = 1 + |D_1|. For a chain with full support
and for the graph chain with the largest unreachable target mass, the
script reports
  E_{q_W}[h]                    target value,
  E_{q_A}[h]                    uncorrected stopped walk,
  E_{q_W}[h | S]                limit of the self-normalized correction,
  q_W(S^c)                      unreachable target mass,
  ESS/n                         normalized effective sample size,
and the mean and standard deviation of the self-normalized estimator over
independent replications of n draws from q_A.

Run: python3 estimator_demo.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_experiments as mx  # noqa: E402

B_SIZE = 2
N_DRAWS = 200
N_REPS = 4000


def tree_size_of_seed(fmaps, b):
    return np.array([1 + sum(1 for r in mx.root_map(f, b) if r == 0) for f in fmaps])


def report(name, c, h, rng):
    qW, qA = c.qW, c.qA[0]
    supp = qA > 1e-14
    lost = qW[(qW > 0) & ~supp].sum()
    target = float(qW @ h)
    uncorr = float(qA @ h)
    qc = qW * supp
    qc = qc / qc.sum()
    limit = float(qc @ h)
    ess = 1.0 / np.sum(qc[supp] ** 2 / qA[supp])
    wts = np.zeros_like(qA)
    wts[supp] = qW[supp] / qA[supp]
    idx = rng.choice(len(qA), size=(N_REPS, N_DRAWS), p=qA)
    num = (wts[idx] * h[idx]).sum(axis=1)
    den = wts[idx].sum(axis=1)
    est = num / den
    print(f"{name}: target {target:.4f}  uncorrected {uncorr:.4f}  "
          f"corrected limit {limit:.4f}  lost mass {lost:.4f}  ESS/n {ess:.3f}  "
          f"SN estimator (n={N_DRAWS}) mean {est.mean():.4f} sd {est.std():.4f}")
    return dict(target=target, uncorr=uncorr, limit=limit, lost=lost, ess=ess,
                sn_mean=float(est.mean()), sn_sd=float(est.std()))


def main():
    rng = np.random.default_rng(mx.SEED)
    chains = {fam: [mx.CHAIN_MAKERS[fam](rng) for _ in range(mx.N_CHAINS)]
              for fam in mx.FAMILIES}
    fmaps = mx.parent_maps(list(range(B_SIZE, mx.N)))
    trees_by_root = {r: np.array(mx.parent_maps([v for v in range(mx.N) if v != r]),
                                 dtype=np.int64) for r in range(mx.N)}
    h = tree_size_of_seed(fmaps, B_SIZE)
    sim = np.random.default_rng(1)
    report("dense chain 0", mx.Case(chains["dense"][0], B_SIZE, fmaps, trees_by_root), h, sim)
    best, best_lost, best_i = None, -1.0, -1
    for ci, P in enumerate(chains["graph"]):
        c = mx.Case(P, B_SIZE, fmaps, trees_by_root)
        lost = c.qW[(c.qW > 0) & ~(c.qA[0] > 1e-14)].sum()
        if lost > best_lost:
            best, best_lost, best_i = c, lost, ci
    report(f"graph chain {best_i}", best, h, sim)


if __name__ == "__main__":
    main()
