# Comparing Wald test, Rao test, GLRT for adaptive detection with limited training data

In many practical detection systems — most notably radar, sonar and array-based sensing
generally — the interference covariance matrix itself is an additional unknown nuisance
parameter, and it must be estimated from auxiliary training (secondary) data: snapshots taken
from nearby resolution cells that are assumed to be free of the target signal but share the same
interference statistics as the cell under test. Consider GLRT, Wald test and Rao test. While all 3
detectors are asymptotically equivalent, diverge in performance when constrained to a limited
number of training snapshots which is to be studied from a classical perspective and verified
using simulations. Next, study how does diagonal loading (regularization) affect the
performance and verify it.

## Repository layout

```
adaptive-detection-limited-data/
├── shared-functions/        functions used by every workload (written once, never copied)
│   ├── steer.m              ULA steering vector a(θ), θ in degrees
│   ├── make_cov.m           interference covariance R = σ²I + Σ Pⱼ a(θⱼ)a(θⱼ)ᴴ
│   ├── gen_data.m           M trials of test vector x and K training snapshots X
│   ├── detect.m             AMF, Kelly and Rao statistics from x, S, p
│   └── calib.m              Monte Carlo threshold for a given Pfa (optional transform of S)
│
├── workload-a/              harness validation and CFAR
├── workload-b/              Task 1: finite-K divergence of the three tests
├── workload-c/              Task 2: diagonal loading
│   ├── c1_sinr_loss.m
│   ├── c2_....m
│   └── results/
│       ├── c1_sinr_loss.png
│       └── c1_sinr_loss.mat
├── workload-d/              robustness: steering mismatch, contaminated training, shrinkage
│
└── toy-example/             mid-evaluation code (sdetmid.m) and slides
```

### Conventions

- **Shared functions live only in `shared-functions/`.** Workload scripts add it to the path instead of copying the files:
  ```matlab
  addpath('../shared-functions');
  ```
  so scripts are run from inside their own workload folder. If a shared function needs a change, change it there and tell the team.
- **One folder per workload**: `workload-a/` to `workload-d/`.
- **One script per sub-workload**, named `<workload letter><step number>_<short_name>.m`, all lowercase, e.g. `c1_sinr_loss.m`, `c2_pd_vs_loading.m`, `a1_mf_check.m`.
- **Outputs go in that workload's `results/` folder**, with the same base name as the script that made them: `results/c1_sinr_loss.png` (figure) and `results/c1_sinr_loss.mat` (data). Every script can be rerun to regenerate its results.

### Shared scenario

Used by every workload unless an experiment deliberately changes it:

| Parameter | Value |
|---|---|
| Array | N = 16 elements, half-wavelength ULA, σ² = 1 |
| Target | θₜ = 0° (broadside) |
| Jammers | −40°, 10°, 20°, JNR = 35 dB each |
| False-alarm rate | Pfa = 10⁻³ |
| SNR | clairvoyant output SINR, \|α\|² pᴴR⁻¹p |
| Diagonal loading | R̂ = S/K + λI, passed to `detect`/`calib` as `S + K*λ*I`; LNR = λ/σ² in dB |

### Plot and `.mat` format (temporary, remove once the project is done)

**Plots**

- Plain `figure`, `subplot`, `plot` with MATLAB's **default figure settings and default colours**. Don't set colours, themes or figure sizes.
- Tell series apart with **marker shapes** (`'o'`, `'s'`, `'^'`, `'d'`, ...), **line styles** (`'-'`, `'--'`, `':'`) and `'LineWidth'`.
- Keep the same plotting order for the same quantities across panels so the default colours match. To give a second set of curves (e.g. unloaded, dashed) the same colours, use `set(gca, 'ColorOrderIndex', 1)` before plotting them.
- Reference lines (nominal Pfa, clairvoyant Pd, 3 dB, 2J, ...) with `yline` / `xline`, no colour given.
- Every panel gets a title and axis labels with units; label series with `'DisplayName'` and add a `legend`. Use a log y-axis for Pfa, with decade ticks.
- Save with `saveas(gcf, 'results/<script name>.png')`.
- The saved PNG follows each person's MATLAB theme (dark or light). For the final deck, regenerate all figures with the same theme so the slides match.

**`.mat` files**

- One per script: `save('results/<script name>.mat', ...)`, saving **plain named variables** (not one big struct), so `load` gives them directly.
- Include:
  - the scenario parameters used (`N`, `theta_t`, `theta_j`, `jnr_dB`, `Pfa`, ... as relevant),
  - the swept values (e.g. K grid, `LNR_dB`, SNR grid),
  - the trial counts (`M`, `M0`, `M1`, ...),
  - `tests = {'AMF', 'Kelly', 'Rao'}` when results are per test,
  - the results themselves (thresholds, Pd, Pfa, losses, ...).
- Result arrays have one dimension per swept variable, in the order they are swept, with the **test as the last dimension**: e.g. `Pd(K, LNR, test)`, `pfa(scenario, K, LNR, test)`.
- Quantities in dB carry a `_dB` suffix (`LNR_dB`, `SNR_dB`, `jnr_dB`). SNR always means the clairvoyant output SINR, \|α\|² pᴴR⁻¹p.
- Set `rng(...)` in the script so results can be regenerated exactly.
