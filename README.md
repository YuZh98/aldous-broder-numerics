# aldous-broder-numerics

Scripts and stored outputs for the numerical illustration (Appendix F) of

> Hugh Zheng, *The Order Mass Identity of Rooted Forests, with a Direct
> Proof of the Extended Aldous–Broder Theorem*.

The Lean formalization of the paper is a separate repository:
<https://github.com/YuZh98/aldous-broder-lean>.

## Files

| Script | Purpose | Output |
|---|---|---|
| `make_experiments.py` | the three chain families, exact forest laws, total variation and lost mass, simulation check | `experiments.csv`, `experiments.json` |
| `plot_bias.py` | the total-variation figure, drawn from `experiments.csv` without recomputation | `bias_tv.pdf` |
| `robustness.py` | sensitivity to root placement, drift strength and connectivity; calibration of the simulation check | `robustness.json` |
| `estimator_demo.py` | inverse-completion weighting for one estimator | printed values |
| `zero_check.py` | zero completion weights: combinatorial test against the numerical threshold | printed values |

The stored outputs are the ones reported in the paper. Each script is run
as `python3 <script>.py` from this directory; it requires Python 3 with
NumPy and Matplotlib. The random seed is fixed in `make_experiments.py`.

## License

[Apache License 2.0](LICENSE).
