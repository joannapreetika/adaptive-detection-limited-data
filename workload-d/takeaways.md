# Workload D: robustness (steering mismatch, contaminated training, shrinkage)

## How to run

From inside `workload-d/` (the scripts add `../shared-functions` themselves):

```matlab
d1_steering_mismatch        % ~25 s
d2_contaminated_training    % ~1.5 min
d3_contamination_loading    % ~2.5 min
d4_shrinkage_vs_loading     % ~6 min
d5_summary                  % prints every number quoted below
results = runtests('tests'); disp(table(results))   % unit tests
```

Each script writes `results/<script name>.png` and `.mat`. Set the environment
variable `SDET_QUICK=1` to run with 1/20 of the trials (outputs go to `results/quick/`).
In batch mode, run `set(groot, "DefaultFigureVisible", "off")` first to stop figure
windows from popping up.

## Setup

- N = 16, half-wavelength ULA, sigma2 = 1, look direction 0 deg, JNR = 35 dB per jammer,
  Pfa = 1e-3.
- **S1** is the shared scenario, with jammers at -40/10/20 deg. **S2** is S1 plus a
  mainlobe jammer at 4 deg. S2 is drawn dashed in every figure.
- SNR is the clairvoyant output SINR along the direction the signal actually comes from.
- Monte Carlo: 1e5 H0 trials per threshold and per actual-Pfa point, 5e3 trials per Pd
  point, chunks of 1e4 trials. Seeds are stored in each .mat.
- S stays at sum scale everywhere. Every loaded or shrunk detector has its own threshold,
  calibrated with the same transform and the same R.
- Operating SNR:
  - D1 uses 16.0 dB, the smallest value on a 0.5 dB grid where all three tests reach
    Pd >= 0.9 at K = 32 with matched steering.
  - D3 and D4 use 10.8 dB, where the clairvoyant matched filter has Pd = 0.9 (the same
    value as Workload C). At 16 dB every loaded method saturates at Pd ~ 1.
- Shared-code change: `shared-functions/calib.m` now also accepts a two-argument
  transform `@(S, X)`, which Ledoit-Wolf needs. Existing `@(S)` calls are unaffected and
  give bit-identical thresholds.

## D1: steering mismatch (`d1_steering_mismatch`)

- **The expected ordering holds in S1.** At K = 32, Pd drops below 0.5 at
  dtheta = 4.23 deg for AMF, 3.20 deg for Kelly and 2.04 deg for Rao. At dtheta = 2 deg
  (cos^2 = 0.780), Pd is 0.920 / 0.847 / 0.519 (AMF / Kelly / Rao). AMF is the most
  tolerant, Rao the most selective, Kelly in between.
- **The ordering holds at every K and shifts outward as K grows.** The 0.5 crossings are
  4.88 / 4.01 / 3.11 deg at K = 48 and 2.35 / 1.88 / 0.02 deg at K = 24. At K = 24, Rao
  is already near Pd = 0.5 when matched.
- **In S2 the mainlobe jammer dominates.** All three tests fall below 0.5 at about
  3.7 deg (3.75 / 3.74 / 3.69 at K = 32), and at 4 deg (cos^2 = 0.000) Pd is
  0.036 / 0.001 / 0.000. Away from that null, cos^2 stays high (0.931 at 3 deg), so the
  tests differ less: Pd at 3 deg is 0.975 / 0.963 / 0.833.

## D2: contaminated training (`d2_contaminated_training`)

- **One strong outlier costs about 6 dB, two cost about 8 dB, and no test is more robust
  than the others.** At K = 32 (S1), one 20 dB outlier raises the SNR needed for
  Pd = 0.9 by 5.80 / 5.76 / 5.91 dB (AMF / Kelly / Rao). Two outliers cost 8.24 / 8.28 dB
  for AMF / Kelly, and Rao never reaches 0.9 within the 24 dB grid. S2 gives the same
  picture (5.81 / 5.79 / 5.98 dB).
- **Contamination makes the clean thresholds conservative.** With one 20 dB outlier
  (K = 32, S1), the actual Pfa is 0.12 / 0.03 / 0.01 x 1e-3, against 0.95 / 0.96 / 0.92
  x 1e-3 with clean training.
- **Weak outliers are harmless and more training dilutes strong ones.** A 0 dB outlier
  costs <= 0.3 dB. The one-outlier, 20 dB loss shrinks from about 5.8 dB at K = 32 to
  4.32 / 4.35 / 4.33 dB at K = 48.

## D3: light loading under contamination (`d3_contamination_loading`, K = 32, SNR = 10.8 dB)

- **Up to 10 dB outliers, 10 dB of loading recovers almost the whole loss.** With one
  10 dB outlier (S1), Pd goes from 0.177 / 0.220 / 0.200 unloaded to
  0.838 / 0.840 / 0.827. The clean, equally loaded detectors reach 0.853 / 0.856 / 0.841.
  The actual Pfa stays close to nominal: 1.09 / 1.01 / 0.91 x 1e-3.
- **20 dB outliers are only partly recovered, and Pfa then rises above nominal.**
  - One outlier: Pd 0.708 / 0.711 / 0.696 vs 0.853 / 0.856 / 0.841 clean, with Pfa
    2.12 / 1.78 / 1.29 x 1e-3.
  - Two outliers: Pd 0.576 / 0.587 / 0.576, with Pfa 3.28 / 2.37 / 1.58 x 1e-3.
  - S2 with two outliers: Pfa up to 5.87 x 1e-3 (AMF).
  - Rao's Pfa inflates the least in every 20 dB case.
- Caveat: the best LNR is 10 dB, the edge of the 0-10 dB "light loading" grid, in almost
  every case. Heavier loading may recover more; in D4, clean training is best at
  LNR = 8-18 dB.

## D4: shrinkage vs diagonal loading (`d4_shrinkage_vs_loading`, SNR = 10.8 dB)

- **Ledoit-Wolf does not reach the tuned optimum, and is far below it.**
  - Its Pd shortfall vs the best tuned method is 0.215-0.755 in S1, and up to 0.845 in
    S2 (AMF, K = 48).
  - In S2 its Pd is <= 0.004 at every K.
  - At K = 48 (S1) it only matches plain S: 0.572 vs 0.569 for AMF.
- **Why:** LW shrinks toward (trace(Rhat)/N) I, and with 35 dB jammers trace(R)/N is
  about 1e4. LW's mean rho (0.068 at K = 48 up to 0.348 at K = 8) therefore amounts to
  roughly 30 dB of loading or more; rho = 0.1 is 30.2 dB in S1. That much loading wipes
  out the jammer nulls, and in S2 the mainlobe one.
- **Fixed shrinkage works once rho is small enough to act as light loading.** The best
  rho is 5e-4 to 5e-3, equivalent to 8-17 dB of loading. With it, Pd is within 0.035 of
  the best loading for K >= 24 (S1 AMF at K = 32: 0.833 vs 0.856). The gap is larger at
  K = 8 (S1 AMF: 0.246 vs 0.423).
- **Tuned loading is the most reliable fix, and it matters most for K <= N.** In S1:
  - At K = 8, the best loading (LNR 14-16 dB) gives Pd 0.423 / 0.623 / 0.547
    (AMF / Kelly / Rao), and plain S is unusable.
  - At K = 48, loading lifts Pd from 0.569 / 0.611 / 0.571 to 0.871 / 0.870 / 0.867.
  - With few snapshots and loading, Kelly is the best of the three; S2 at K = 8 gives
    0.164 / 0.509 / 0.428.
- Caveat: "best" is the maximum over 11 (loading) or 20 (shrinkage) noisy Pd estimates,
  so it is biased upward by about one standard error (~0.006).
