#!/usr/bin/env python3
"""Zero completion weights: combinatorial test against a numerical
threshold, for every chain, root set and forest of make_experiments.py
with seed 1.

phi_1(f) = 0 exactly when some root u != 1 cannot reach 1 along the arcs
u -> u' for which some x in the tree of u' has P[x][u] > 0 (the positive
entries of the contracted matrix Q). The script compares this test with
the threshold 1e-14 on g(B,f)^T L[B]^{-1} e_1 computed in double precision.

Run: python3 zero_check.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_experiments as mx  # noqa: E402

N = mx.N


def reaches_seed(P, f, b):
    roots = mx.root_map(f, b)

    def tree_of(x):
        return x if x < b else roots[x - b]

    adj = {u: set() for u in range(b)}
    for x in range(N):
        for u in range(1, b):
            if P[x][u] > 0 and tree_of(x) != u:
                adj[u].add(tree_of(x))
    for u in range(1, b):
        seen, stack = {u}, [u]
        found = False
        while stack:
            y = stack.pop()
            if y == 0:
                found = True
                break
            for z in adj[y]:
                if z not in seen:
                    seen.add(z)
                    stack.append(z)
        if not found:
            return False
    return True


def main():
    rng = np.random.default_rng(mx.SEED)
    chains = {fam: [mx.CHAIN_MAKERS[fam](rng) for _ in range(mx.N_CHAINS)]
              for fam in mx.FAMILIES}
    trees = {r: np.array(mx.parent_maps([v for v in range(N) if v != r]), dtype=np.int64)
             for r in range(N)}
    total = zeros = mismatches = 0
    min_pos, max_zero = np.inf, 0.0
    for fam in mx.FAMILIES:
        for P in chains[fam]:
            for b in (2, 3, 4, 5, 6):
                fm = mx.parent_maps(list(range(b, N)))
                c = mx.Case(P, b, fm, trees)
                phi = c.phi[:, 0]
                for t, f in enumerate(fm):
                    positive = reaches_seed(P, f, b)
                    total += 1
                    mismatches += positive != (phi[t] > 1e-14)
                    if positive:
                        min_pos = min(min_pos, phi[t])
                    else:
                        zeros += 1
                        max_zero = max(max_zero, abs(phi[t]))
    print(f"cases {total}, structural zeros {zeros}, mismatches with the threshold "
          f"{mismatches}, largest |value| at a zero {max_zero:.2e}, smallest positive "
          f"value {min_pos:.2e}")


if __name__ == "__main__":
    main()
