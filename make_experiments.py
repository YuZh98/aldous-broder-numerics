#!/usr/bin/env python3
r"""Experiments of Section 6.4: bias of the stopped covering walk, the
seed-randomized variant, the importance-weight corrections, and costs.

Three families of chains on V = {1,...,7}, the same chains for every
root set B = {1,...,b}, b = 1,...,6, seed 1 (repeated across root sets):
  dense   P_uv ∝ W_uv, W_uv integer uniform in {1,...,5}    (not reversible)
  graph   simple random walk on a random connected graph   (reversible, sparse)
  drift   cycle with weight 3 forward, 1 backward, random holding weight
For each chain and b the script computes in closed form
  q_W   ∝ w(f)                      (weight-proportional law)
  q_A   ∝ w(f) φ_1(f)               (stopped walk, seed 1; Theorem 6.3)
  q_mix ∝ w(f) g(f)                 (flow-proportional seed; Proposition 6.6)
with φ_r and g from the principal cofactors of I - Γ (Theorem 3.1), and
checks, in exact arithmetic where it matters,
  (1) the normalizing constant of q_A for every seed,
  (2) q_A against direct enumeration of the arborescences (Lemma 6.2),
  (3) q_mix against the seed average of the q_A,
  (4) the Dobrushin bound of Proposition 6.6 on TV(q_mix, q_W).
It also simulates Algorithm 1 on one chain and compares the empirical
forest law, the mean of the importance weight (a control variate with
known mean) and the mean number of steps with the exact values, and
tabulates the exact expected number of steps E T_B against the expected
number of steps tr G_Z of Wilson's algorithm.
Outputs: experiments.csv, experiments.json, bias.pdf, run_experiments.log.
"""

import csv
import itertools
import json
import os
import sys
import time
from fractions import Fraction

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 7
B_SIZES = [1, 2, 3, 4, 5, 6]
N_CHAINS = int(os.environ.get("NCHAINS", "30"))
N_SIM = int(os.environ.get("NSIM", "200000"))
SEED = 20260918
TOL = 1e-10
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FAMILIES = ["dense", "graph", "drift"]


# ---------------- exact linear algebra over Fraction ----------------

def frac_solve(A, b):
    n = len(A)
    M = [[Fraction(x) for x in A[i]] + [Fraction(b[i])] for i in range(n)]
    for c in range(n):
        p = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        M[c] = [x / piv for x in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                fac = M[r][c]
                M[r] = [x - fac * y for x, y in zip(M[r], M[c])]
    return [M[i][n] for i in range(n)]


def frac_det(A):
    n = len(A)
    M = [[Fraction(x) for x in row] for row in A]
    det = Fraction(1)
    for c in range(n):
        p = next((r for r in range(c, n) if M[r][c] != 0), None)
        if p is None:
            return Fraction(0)
        if p != c:
            M[c], M[p] = M[p], M[c]
            det = -det
        det *= M[c][c]
        for r in range(c + 1, n):
            if M[r][c] != 0:
                fac = M[r][c] / M[c][c]
                M[r] = [x - fac * y for x, y in zip(M[r], M[c])]
    return det


def frac_inv(A):
    n = len(A)
    cols = []
    for j in range(n):
        e = [Fraction(0)] * n
        e[j] = Fraction(1)
        cols.append(frac_solve(A, e))
    return [[cols[j][i] for j in range(n)] for i in range(n)]


# ---------------- chains ----------------

def chain_dense(rng):
    W = rng.integers(1, 6, size=(N, N))
    return [[Fraction(int(W[u, v]), int(W[u].sum())) for v in range(N)]
            for u in range(N)]


def chain_graph(rng):
    while True:
        adj = np.zeros((N, N), dtype=int)
        for u in range(N):
            for v in range(u + 1, N):
                if rng.random() < 0.45:
                    adj[u, v] = adj[v, u] = 1
        seen = {0}
        stack = [0]
        while stack:
            x = stack.pop()
            for y in range(N):
                if adj[x, y] and y not in seen:
                    seen.add(y)
                    stack.append(y)
        if len(seen) == N:
            break
    return [[Fraction(int(adj[u, v]), int(adj[u].sum())) for v in range(N)]
            for u in range(N)]


def chain_drift(rng):
    W = np.zeros((N, N), dtype=int)
    for u in range(N):
        W[u, (u + 1) % N] = 3
        W[u, (u - 1) % N] = 1
        W[u, u] = int(rng.integers(0, 3))
    return [[Fraction(int(W[u, v]), int(W[u].sum())) for v in range(N)]
            for u in range(N)]


CHAIN_MAKERS = {"dense": chain_dense, "graph": chain_graph, "drift": chain_drift}


def stationary(P):
    A = [[P[v][u] - (1 if u == v else 0) for v in range(N)] for u in range(N)]
    A[N - 1] = [Fraction(1)] * N
    b = [Fraction(0)] * N
    b[N - 1] = Fraction(1)
    nu = frac_solve(A, b)
    assert all(x > 0 for x in nu), "stationary vector not positive"
    return nu


# ---------------- forests ----------------

def parent_maps(Z):
    """All maps f: Z -> V whose iteration from any v in Z reaches V \\ Z."""
    Zset = set(Z)
    pos = {v: i for i, v in enumerate(Z)}
    out = []
    for par in itertools.product(range(N), repeat=len(Z)):
        ok = True
        for start in Z:
            x = start
            seen = 0
            while x in Zset:
                bit = 1 << pos[x]
                if seen & bit:
                    ok = False
                    break
                seen |= bit
                x = par[pos[x]]
            if not ok:
                break
        if ok:
            out.append(par)
    return out


def root_map(par, b):
    """root_map[i] = root of the non-root b+i."""
    k = N - b
    roots = []
    for i in range(k):
        x = b + i
        while x >= b:
            x = par[x - b]
        roots.append(x)
    return roots


# ---------------- closed-form laws ----------------

def cofactor_vector(A):
    """Principal cofactors (det A[all minus i])_i of a square float matrix."""
    m = A.shape[0]
    if m == 1:
        return np.ones(1)
    out = np.empty(m)
    for i in range(m):
        idx = [j for j in range(m) if j != i]
        out[i] = np.linalg.det(A[np.ix_(idx, idx)])
    return out


def order_masses_by_dp(f, gam, b, k):
    """g_i(B,f) by the sum over B-orders (Definition 2.9), for a check of
    Theorem 3.1 against the cofactor formula."""
    cur = {}
    for i in range(k):
        if f[i] < b:
            cur[(0, i, f[i])] = cur.get((0, i, f[i]), 0.0) + 1.0
    for _ in range(k - 1):
        nxt = {}
        for (S, last, root), val in cur.items():
            g = gam[(S, last)]
            NS = S | (1 << last)
            for i in range(k):
                if NS >> i & 1:
                    continue
                gv = g[f[i]]
                if gv:
                    key = (NS, i, root)
                    nxt[key] = nxt.get(key, 0.0) + val * gv
        cur = nxt
    out = np.zeros(b)
    for (_, _, root), val in cur.items():
        out[root] += val
    return out


def gamma_table(Lf, b, k):
    out = {}
    Bidx = list(range(b))
    for S in range(1 << k):
        U = Bidx + [b + j for j in range(k) if S >> j & 1]
        for i in range(k):
            if S >> i & 1:
                continue
            w = b + i
            g = np.zeros(N)
            g[w] = 1.0
            g[U] = np.linalg.solve(Lf[np.ix_(U, U)], -Lf[U, w])
            out[(S, i)] = g.tolist()
    return out


class Case:
    def __init__(self, P, b, fmaps, trees_by_root, check_dp=False):
        self.b = b
        k = N - b
        self.k = k
        nu = stationary(P)
        self.nu = nu
        Lex = [[(Fraction(1) if u == v else Fraction(0)) - nu[v] * P[v][u] / nu[u]
                for v in range(N)] for u in range(N)]
        Lf = np.array([[float(x) for x in row] for row in Lex])
        self.Lf = Lf
        A_edge = np.array([[float(nu[u] * P[u][v]) for v in range(N)]
                           for u in range(N)])
        F = np.array(fmaps, dtype=np.int64)
        self.F = F
        wf = np.ones(len(F))
        for i in range(k):
            wf *= A_edge[F[:, i], b + i]
        self.wf = wf
        self.qW = wf / wf.sum()

        LB = [row[:b] for row in Lex[:b]]
        LBinv_ex = frac_inv(LB)
        LBinv = np.array([[float(x) for x in row] for row in LBinv_ex])
        Bidx = list(range(b))
        Zidx = list(range(b, N))
        KB = -LBinv @ Lf[np.ix_(Bidx, Zidx)]          # b x k, rows sum to 1
        self.KB = KB

        # order masses g_i from the cofactors of I - Gamma (Theorem 3.1)
        G = np.empty((len(F), b))
        for t, f in enumerate(fmaps):
            roots = root_map(f, b)
            Gam = np.zeros((b, b))
            for i in range(k):
                Gam[:, roots[i]] += KB[:, i]
            G[t] = cofactor_vector(np.eye(b) - Gam)
        self.G = G
        self.g = G.sum(axis=1)
        self.phi = G @ LBinv                          # phi[t, r] = sum_i g_i (L[B]^-1)_{ir}

        if check_dp:
            gam = gamma_table(Lf, b, k)
            worst = 0.0
            for t, f in enumerate(fmaps):
                gd = order_masses_by_dp(f, gam, b, k)
                worst = max(worst, np.max(np.abs(gd - G[t])))
            self.dp_err = worst

        # exact normalizing constants kappa_r = det L[B] / (nu_r prod_Z nu_v sum_u lambda_u)
        detLB = frac_det(LB)
        ImP = [[(Fraction(1) if u == v else Fraction(0)) - P[u][v]
                for v in range(N)] for u in range(N)]
        sumlam = sum(frac_det([[ImP[r][c] for c in range(N) if c != u]
                               for r in range(N) if r != u]) for u in range(N))
        prod_nuZ = Fraction(1)
        for v in Zidx:
            prod_nuZ *= nu[v]
        self.kappa = [float(detLB / (nu[r] * prod_nuZ * sumlam)) for r in Bidx]
        self.sumlam = sumlam
        self.detLB = detLB
        self.detLZ = frac_det([[Lex[u][v] for v in Zidx] for u in Zidx])

        # CHECK 1 and 2: q_A for every seed
        self.qA = np.empty((b, len(F)))
        self.err_norm = 0.0
        self.err_direct = 0.0
        for r in Bidx:
            unnorm = wf * self.phi[:, r] * self.kappa[r]
            self.err_norm = max(self.err_norm, abs(unnorm.sum() - 1.0))
            trees = trees_by_root[r]
            Zr = [v for v in range(N) if v != r]
            tw = np.ones(len(trees))
            for j, v in enumerate(Zr):
                tw *= A_edge[trees[:, j], v]
            cols = [Zr.index(v) for v in Zidx]
            keys = np.zeros(len(trees), dtype=np.int64)
            for i, col in enumerate(cols):
                keys += trees[:, col] * (N ** i)
            num = np.bincount(keys, weights=tw, minlength=N ** k)
            fkeys = (F * (N ** np.arange(k))).sum(axis=1)
            qD = num[fkeys] / tw.sum()
            self.err_direct = max(self.err_direct, np.max(np.abs(unnorm - qD)))
            self.qA[r] = unnorm

        # flow-proportional seed law and q_mix (Proposition 6.6)
        LB1 = [sum(row) for row in LB]
        seedlaw = np.array([float(nu[r] * LB1[r]) for r in Bidx])
        seedlaw /= seedlaw.sum()
        self.seedlaw = seedlaw
        qmix = wf * self.g
        qmix /= qmix.sum()
        self.qmix = qmix
        self.err_mix = np.max(np.abs(seedlaw @ self.qA - qmix))   # CHECK 3

        # Dobrushin coefficient of K_B and the bound of Proposition 6.6
        delta = 0.0
        for i in Bidx:
            for i2 in Bidx:
                delta = max(delta, 0.5 * np.abs(KB[i] - KB[i2]).sum())
        self.delta = delta
        self.bound = (0.5 * (((1 + delta) / (1 - delta)) ** (b - 1) - 1)
                      if delta < 1 else float("inf"))

    def stats(self, q):
        """TV to q_W, unreachable target mass, ESS and gap on the support."""
        qW = self.qW
        tv = 0.5 * np.abs(q - qW).sum()
        supp = q > 1e-14
        pos = qW > 0
        lost = qW[pos & ~supp].sum()
        qc = qW * supp
        qc /= qc.sum()
        ess = 1.0 / np.sum(qc[supp] ** 2 / q[supp])
        gap = np.min(q[supp] / qc[supp])
        return tv, lost, ess, gap


# ---------------- simulation of Algorithm 1 ----------------

def simulate(P, b, seed_vertex, n_sim, rng):
    """Run Algorithm 1 n_sim times; return forest counts, mean steps and
    the list of forests (as tuples)."""
    Pf = np.array([[float(x) for x in row] for row in P])
    cum = np.cumsum(Pf, axis=1)
    counts = {}
    steps_total = 0
    k = N - b
    for _ in range(n_sim):
        x = seed_vertex
        visited = 0
        par = [None] * k
        t = 0
        while visited != (1 << k) - 1:
            y = int(np.searchsorted(cum[x], rng.random()))
            if y >= N:
                y = N - 1
            t += 1
            if y >= b and not (visited >> (y - b) & 1):
                visited |= 1 << (y - b)
                par[y - b] = x
            x = y
        steps_total += t
        key = tuple(par)
        counts[key] = counts.get(key, 0) + 1
    return counts, steps_total / n_sim


def expected_steps_dp(P, b, seed_vertex):
    """Exact E_seed T_B by a dynamic program over (position, visited set)."""
    Pf = np.array([[float(x) for x in row] for row in P])
    k = N - b
    full = (1 << k) - 1
    E = {}
    for S in range(full, -1, -1):
        if S == full:
            E[S] = np.zeros(N)
            continue
        A = np.eye(N)
        rhs = np.ones(N)
        for x in range(N):
            for y in range(N):
                if y >= b and not (S >> (y - b) & 1):
                    rhs[x] += Pf[x, y] * E[S | (1 << (y - b))][y]
                else:
                    A[x, y] -= Pf[x, y]
        E[S] = np.linalg.solve(A, rhs)
    return E[0][seed_vertex]


def wilson_steps(P, b):
    Pf = np.array([[float(x) for x in row] for row in P])
    Z = list(range(b, N))
    G = np.linalg.inv(np.eye(N - b) - Pf[np.ix_(Z, Z)])
    return np.trace(G)


# ---------------- main ----------------

def main():
    t0 = time.time()
    log = open(os.path.join(OUT_DIR, "run_experiments.log"), "w")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    rng = np.random.default_rng(SEED)
    chains = {fam: [CHAIN_MAKERS[fam](rng) for _ in range(N_CHAINS)]
              for fam in FAMILIES}
    fmaps_by_b = {b: parent_maps(list(range(b, N))) for b in B_SIZES}
    trees_by_root = {r: np.array(parent_maps([v for v in range(N) if v != r]),
                                 dtype=np.int64) for r in range(N)}

    rows = []
    worst = {"norm": 0.0, "direct": 0.0, "mix": 0.0, "dp": 0.0}
    bound_violations = 0
    for fam in FAMILIES:
        for ci, P in enumerate(chains[fam]):
            for b in B_SIZES:
                check_dp = (ci == 0)
                c = Case(P, b, fmaps_by_b[b], trees_by_root, check_dp=check_dp)
                for name, val in (("norm", c.err_norm), ("direct", c.err_direct),
                                  ("mix", c.err_mix)):
                    if val > TOL:
                        sys.exit(f"CHECK {name} FAILED ({fam}, chain {ci}, b={b}): {val:.3e}")
                    worst[name] = max(worst[name], val)
                if check_dp:
                    if c.dp_err > 1e-9:
                        sys.exit(f"CHECK dp FAILED ({fam}, chain {ci}, b={b}): {c.dp_err:.3e}")
                    worst["dp"] = max(worst["dp"], c.dp_err)
                tvA, lostA, essA, gapA = c.stats(c.qA[0])
                tvM, lostM, essM, gapM = c.stats(c.qmix)
                if tvM > c.bound + 1e-12:
                    bound_violations += 1
                rows.append(dict(family=fam, chain=ci, b=b, n_forests=len(c.F),
                                 tv_fixed=tvA, lost_fixed=lostA, ess_fixed=essA, gap_fixed=gapA,
                                 tv_mix=tvM, lost_mix=lostM, ess_mix=essM, gap_mix=gapM,
                                 delta=c.delta, bound=c.bound,
                                 gmin=float(c.g.min()), gmax=float(c.g.max())))
                say(f"{fam:5s} chain {ci:2d} b={b}: TV1={tvA:.4f} TVmix={tvM:.4f} "
                    f"ESS1={essA:.3f} gap1={gapA:.3f} lost1={lostA:.3f} "
                    f"delta={c.delta:.3f} bound={c.bound:.3f}")
    say(f"\nAll checks passed: worst |sum-1| = {worst['norm']:.2e}, "
        f"worst |q_A - q_direct| = {worst['direct']:.2e}, "
        f"worst |seed-average - q_mix| = {worst['mix']:.2e}, "
        f"worst |g(orders) - g(cofactors)| = {worst['dp']:.2e}, "
        f"Dobrushin-bound violations: {bound_violations}")

    # summaries
    summary = {}
    for fam in FAMILIES:
        summary[fam] = {}
        for b in B_SIZES:
            sel = [r for r in rows if r["family"] == fam and r["b"] == b]
            d = {}
            for key in ("tv_fixed", "tv_mix", "ess_fixed", "gap_fixed", "lost_fixed",
                        "ess_mix", "gap_mix", "lost_mix", "delta", "bound"):
                arr = np.array([r[key] for r in sel])
                d[key] = dict(median=float(np.median(arr)), min=float(arr.min()),
                              max=float(arr.max()))
            d["frac_lost_positive"] = float(np.mean([r["lost_fixed"] > 1e-12 for r in sel]))
            d["ratio_tv_to_bound_max"] = float(max(
                (r["tv_mix"] / r["bound"] if r["bound"] > 0 else 0.0) for r in sel))
            summary[fam][str(b)] = d
            say(f"{fam:5s} b={b}: TV1 med {d['tv_fixed']['median']:.3f} "
                f"[{d['tv_fixed']['min']:.3f},{d['tv_fixed']['max']:.3f}]  "
                f"TVmix med {d['tv_mix']['median']:.3f} max {d['tv_mix']['max']:.3f}  "
                f"ESS1 med {d['ess_fixed']['median']:.3f} min {d['ess_fixed']['min']:.3f}  "
                f"gap1 med {d['gap_fixed']['median']:.3f} min {d['gap_fixed']['min']:.3f}  "
                f"lost>0: {d['frac_lost_positive']:.2f}")

    # simulation on the first dense chain, b = 2, seed 1
    P0 = chains["dense"][0]
    bsim = 2
    c = Case(P0, bsim, fmaps_by_b[bsim], trees_by_root)
    srng = np.random.default_rng(SEED + 1)
    counts, mean_steps = simulate(P0, bsim, 0, N_SIM, srng)
    fidx = {tuple(f): t for t, f in enumerate(fmaps_by_b[bsim])}
    emp = np.zeros(len(c.F))
    for key, cnt in counts.items():
        emp[fidx[key]] = cnt / N_SIM
    qA = c.qA[0]
    max_dev = float(np.max(np.abs(emp - qA)))
    max_sd = float(np.max(np.sqrt(qA * (1 - qA) / N_SIM)))
    chi2 = float(np.sum((emp - qA) ** 2 / qA * N_SIM))
    dof = int(np.sum(qA > 0) - 1)
    # control variate: E_{q_A}[1/phi_1] = kappa_1 * sum_{phi>0} w
    phi1 = c.phi[:, 0]
    w_all = np.concatenate([np.full(cnt, 1.0 / phi1[fidx[key]]) for key, cnt in counts.items()])
    cv_mean = float(w_all.mean())
    cv_se = float(w_all.std(ddof=1) / np.sqrt(N_SIM))
    cv_exact = float(c.kappa[0] * c.wf[phi1 > 0].sum())
    cv_exact_closed = float(c.detLB * c.detLZ / (c.nu[0] * c.sumlam))
    ET = expected_steps_dp(P0, bsim, 0)
    sim = dict(family="dense", chain=0, b=bsim, seed=1, n_sim=N_SIM,
               n_forests=len(c.F), max_abs_dev=max_dev, max_binomial_sd=max_sd,
               chi2=chi2, dof=dof, cv_mean=cv_mean, cv_se=cv_se, cv_exact=cv_exact,
               cv_exact_closed=cv_exact_closed, mean_steps=mean_steps, exact_steps=ET)
    say(f"\nsimulation: {N_SIM} walks, {len(c.F)} forests: max|emp-q_A| = {max_dev:.2e} "
        f"(largest binomial sd {max_sd:.2e}), chi2 = {chi2:.1f} on {dof} dof; "
        f"control variate mean {cv_mean:.5f} +- {cv_se:.5f} vs exact {cv_exact:.5f} "
        f"(closed form {cv_exact_closed:.5f}); mean steps {mean_steps:.3f} vs exact {ET:.3f}")

    # costs: exact E_1 T_B against tr G_Z on the first chain of each family
    costs = {}
    for fam in FAMILIES:
        P = chains[fam][0]
        costs[fam] = {}
        for b in B_SIZES:
            ET = expected_steps_dp(P, b, 0)
            trG = wilson_steps(P, b)
            costs[fam][str(b)] = dict(E_T_B=float(ET), tr_G_Z=float(trG), ratio=float(ET / trG))
            say(f"cost {fam:5s} b={b}: E_1 T_B = {ET:.3f}, tr G_Z = {trG:.3f}, ratio {ET/trG:.2f}")

    # write CSV and JSON
    csv_path = os.path.join(OUT_DIR, "experiments.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.16e}" if isinstance(v, float) else v) for k, v in r.items()})
    json_path = os.path.join(OUT_DIR, "experiments.json")
    with open(json_path, "w") as fh:
        json.dump(dict(n=N, n_chains=N_CHAINS, seed=SEED, families=FAMILIES,
                       checks=worst, bound_violations=bound_violations,
                       summary=summary, simulation=sim, costs=costs,
                       runtime_seconds=time.time() - t0), fh, indent=1)

    # figure: two panels, TV versus b, three families
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True)
    markers = {"dense": ("o", "0.25"), "graph": ("s", "0.55"), "drift": ("^", "0.75")}
    offs = {"dense": -0.22, "graph": 0.0, "drift": 0.22}
    jr = np.random.default_rng(7)
    for ax, key, title in ((axes[0], "tv_fixed", "seed $r=1$"),
                           (axes[1], "tv_mix", "flow-proportional seed, uncorrected")):
        for fam in FAMILIES:
            m, col = markers[fam]
            for b in B_SIZES:
                vals = np.array([r[key] for r in rows if r["family"] == fam and r["b"] == b])
                x = b + offs[fam] + jr.uniform(-0.06, 0.06, size=len(vals))
                ax.scatter(x, vals, s=11, marker=m, color=col, alpha=0.7,
                           linewidths=0, zorder=2, label=fam if b == 1 else None)
                ax.hlines(np.median(vals), b + offs[fam] - 0.1, b + offs[fam] + 0.1,
                          color="black", lw=1.3, zorder=3)
        ax.set_xlabel(r"$b=|B|$")
        ax.set_xticks(B_SIZES)
        ax.set_xlim(0.4, 6.6)
        ax.set_title(title, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, color="0.88", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(direction="out", length=3)
    axes[0].set_ylabel("total variation distance to $q_W$")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    ymax = max(r["tv_fixed"] for r in rows)
    axes[0].set_ylim(-0.03 * ymax, 1.1 * ymax)
    fig.tight_layout(pad=0.4)
    pdf_path = os.path.join(OUT_DIR, "bias.pdf")
    fig.savefig(pdf_path)
    say(f"\nwrote {pdf_path}, {csv_path}, {json_path}; runtime {time.time()-t0:.0f} s")
    log.close()


if __name__ == "__main__":
    main()
