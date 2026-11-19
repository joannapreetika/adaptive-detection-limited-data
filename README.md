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