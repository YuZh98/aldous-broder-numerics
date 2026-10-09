#!/usr/bin/env python3
"""Plot the total variation distance between the stopped-walk forest law
and the weight-proportional forest law, per chain family and root-set size.

The distances are read from experiments.csv (column tv_fixed),
produced by make_experiments.py (stationary distributions and
normalizing constants in exact arithmetic, laws in double precision);
nothing is recomputed here.

Run: python3 plot_bias.py
"""
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "experiments.csv")
OUT = os.path.join(HERE, "bias_tv.pdf")

FAMILIES = [
    ("dense", "dense (nonreversible)", "o", "#0072B2"),
    ("graph", "graph (reversible)", "s", "#D55E00"),
    ("drift", "drift (one forest law, 30 chains)", "^", "#009E73"),
]
OFFSETS = {"dense": -0.22, "graph": 0.0, "drift": 0.22}


def main():
    tv = defaultdict(list)
    with open(SRC) as fh:
        for row in csv.DictReader(fh):
            tv[(row["family"], int(row["b"]))].append(float(row["tv_fixed"]))
    b_sizes = sorted({b for (_, b) in tv})

    plt.rcParams.update({"font.size": 9, "font.family": "serif",
                         "font.serif": ["cmr10"],
                         "mathtext.fontset": "cm",
                         "axes.formatter.use_mathtext": True,
                         "axes.unicode_minus": False,
                         "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    for fam, label, marker, color in FAMILIES:
        for b in b_sizes:
            vals = np.array(tv[(fam, b)])
            if fam == "drift":
                vals = vals[:1]  # the 30 drift chains share one forest law
            x = np.full(vals.shape, b + OFFSETS[fam])
            ax.scatter(x, vals, s=12, marker=marker, color=color, alpha=0.75,
                       edgecolors="none",
                       label=label if b == b_sizes[0] else None, zorder=3)
            ax.hlines(np.median(vals), b + OFFSETS[fam] - 0.1,
                      b + OFFSETS[fam] + 0.1, color=color, lw=1.6, zorder=4)
            ax.hlines(np.median(vals), b + OFFSETS[fam] - 0.1,
                      b + OFFSETS[fam] + 0.1, color="k", lw=0.5, zorder=5)
    ax.set_xlabel(r"number of roots $|B|$")
    ax.set_ylabel("total variation distance")
    ax.set_xticks(b_sizes)
    ax.set_xlim(0.4, b_sizes[-1] + 0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="0.88", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, loc="lower center", ncol=3, markerscale=1.8,
              bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
