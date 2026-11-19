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
