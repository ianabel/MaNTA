# Noisy fluxes that report an error bar — research notes

**Branch:** `gp-toy-model` (work in progress; not for `main`)
**Written:** 2026-10-17, against `b7c7352`.

## What this document is

The question: if a noisy flux model hands MaNTA a maximum-likelihood estimate
*and* an error measure at every point, as GX does through QUENDS, can a steady
solve do better than when it only has the noisy value and must fight it?

| Document | What it holds |
|---|---|
| `python-examples/toy-model/README.md`, last two sections | the measured tables and how to reproduce them |
| `python-examples/toy-model/reporting.py` | the box that reports an error bar and the filter that uses it; the docstring has the mechanisms |
| `docs/superpowers/specs/2026-10-11-steady-state-termination-design.md`, "Noise: phase 2" | the planned solver-side work: `NoiseLimited`, iterate averaging, a noise hook |
| `refs/TransportWithNoise.pdf` | Parker, LoDestro & Campos (2018), summarised below |

These notes record what was found, how it connects to the paper and to the
architecture the branch is aiming at, and what is still open.

---

## The noise model: GX through QUENDS

GX's fluxes reach a transport solver as time averages, processed by QUENDS
(https://github.com/sandialabs/quends). Read from its source, not its README:

* `RobustWorkflow.process_data_stream` finds the start of the statistically
  steady state (`sss_start`), trims the transient, and takes the mean over the
  rest.
* `mean_uncertainty` is the **standard error of autotuned, independent block
  means** (`compute_statistics`, `autotune_blocks`). Block size is tuned until a
  Ljung-Box test passes. If there are too few blocks for that test, the standard
  error falls back to a Geyer effective sample size, `n / tau_int` with
  `tau_int` from positive-pair truncation of the autocorrelation function.
* With no steady segment the result is **ad hoc**: the mean of the last 34% of
  the series, with `mean_uncertainty` equal to 100% of the mean
  (`_IRREGULAR_REL_UNCERTAINTY = 1.0`), flagged `NoStatSteadyState`. With
  `operate_safe` it is NaN instead.

GX is a flux-tube code, so every radial point is its own run with its own QUENDS
analysis. The error therefore varies independently by point *and* by call, and
occasionally one sample is unreliable and says so.

`QuendsFlux` (`reporting.py`) reproduces three features: an effective sample
size log-uniform on [4, 64], so the error varies 4x; a reported error that is
itself estimated from about that many blocks (a chi factor); and a fraction of
irregular runs, biased by up to +-50% and reported at 100%. It does **not**
reproduce four others, each listed under *Open* below: spatially correlated
noise, log-normal bursty fluctuations, error that depends on the state, and
error in the error bar beyond the chi factor.

## The filter

`FilteredCase` keeps one scalar Kalman filter at each physics node. The state is
`est`, the estimate of the true flux at the solver's current state at that
point, and `var`, its variance P. On each value call:

1. **Predict.** The solver has moved the point from `(u0, q0)` to `(u, q)`, so
   `est += dGamma`, where `dGamma = 1/2 (J0 + J) . (du, dq)`, and
   `P += (kappa dGamma)^2`. The process noise grows with the extrapolation, so
   old samples are trusted only while the state has stopped moving.
2. **Update** with the sample `y` of variance `R`: `K = P / (P + R)`,
   `est += K (y - est)`, `P *= 1 - K`.

MaNTA is handed `est`. Far from convergence the state moves a lot, P is large,
`K ~ 1`, and the filter returns the sample: no lag. Near convergence P keeps
shrinking and, for equal R, `K ~ 1/n` and `est` is the running mean. That is
phase 4's iterate averaging, done on the flux and correct for samples taken at
slightly different states. Unequal R makes it the inverse-variance mean, and a
sample reported at 100% gets `K ~ 0`.

The arms differ only in R: none ("raw", today's behaviour); one global relative
level, `(eps |est|)^2` ("uniform", the best possible without an error bar, given
the true level); or the reported error squared ("weighted").

**It is a GP in miniature.** A Kalman filter is GP regression done one sample
at a time, for a model that is a random walk along a single dimension. Here
that dimension is the distance the state has moved, the walk is the kappa term,
and the extrapolation uses J. What it lacks is any sharing between points, so it
cannot predict where it has not sampled. That is the capability the target
design below needs.

## Findings

48 cells, k = 4, median relative sample error `eps` = 1e-2, eight seeds. Errors
are against the noiseless discrete solution, in units of `eps`. The full tables
are in the README.

**With the box's exact derivatives** (the control), at 100 calls:

| arm | flat | spread (4x) | irregular (2% of runs) |
|---|---|---|---|
| raw | 0.052 | 0.065 | 0.15 |
| uniform | 0.014 | 0.008 | 0.023 |
| weighted | 0.016 | 0.006 | 0.006 |

* Filtering gains 4-10x on raw, whatever it is told.
* The error bar gains nothing on flat noise and about 1.3x on a 4x spread,
  which is the `sqrt(E[s^2] E[1/s^2])` the spread predicts.
* It gains 3-4x when runs fail, because a 100% error bar gives the bad sample
  almost no weight.
* Only the relative weights matter: error bars misreported by 2x, or a global
  level off by 3x, change almost nothing.
* kappa has a plateau from 0.03 to 0.3. At 0.01 the filter lags; at 1 it hardly
  averages.

**Without derivatives** (MaNTA and the filter both from forward differences at
`dz = sqrt(eps)`, three calls a build), at 800 calls:

| arm | guard | flat | spread | irregular |
|---|---|---|---|---|
| raw | -- | 0.063 | diverges in at least half | diverges |
| uniform | -- | 0.015 | 0.014 | diverges |
| weighted | -- | 0.011 | 0.012 | diverges |
| uniform | 0.5 | 0.013 | 0.010 | diverges |
| weighted | 0.5 | 0.010 | 0.014 | 0.009 |

* The filter still gains 4-6x, but needs about 4x the calls of the control for
  the same accuracy.
* **A difference must not share a sample with the filter.** Differencing from
  the last value call's sample drifts the filtered solve to 0.074 `eps`.
  Systematic J errors of 11-20% and independent random ones of 30% were both
  tested and cost nothing. The cause is the correlation: the base sample's noise
  enters J~ divided by h, and the filter's prediction carries it back.
  `fresh_base` takes a sample of its own, at one call per build.
* **One flagged run in a difference wrecks the Jacobian, and only the error bar
  can see it.** A biased sample divided by h makes a wild entry, Newton follows
  it, and u goes negative. The guard keeps the previous derivative where
  `sqrt(s_base^2 + s_pert^2) / h` exceeds half the difference. With the reported
  error that rejects the flagged differences; with a global level it cannot.

**The error bar's value is concentrated where the noise misbehaves.** It is
worth almost nothing on well-behaved noise and is decisive when runs fail. That
is the GX regime: QUENDS's no-steady-state fallback exists because it happens.

### Stopping

The NewtonStep test passes on a filtered flux at rtol 1e-6 when the real error
is about 7e-5. With kappa = 0.01 it passes on every seed at 0.05 `eps`. The
filter makes F change slowly between calls, so successive corrections shrink
like 1/n whether or not the noise has gone. Raw, the same test never passes
below the floor. Neither the correction nor the residual measures the remaining
noise. The filter's posterior P, carried to u through `J^-1`, does, so a
stopping test has to read it. That is the phase 4 hook, fed by the surrogate's
variance rather than by the box's.

## Parker, LoDestro & Campos (2018)

*Investigation of a multiple-timescale turbulence-transport coupling method in
the presence of random fluctuations*, Plasma **1**, 126-143,
doi:10.3390/plasma1010012. LLNL; the solver is Tango (LoDestro's method, as in
Shestakov et al. 2003), the same family as `refs/LoDestroMethod.pdf`.

**What they do.** They take Tango's Picard iteration, which builds a
diffusive/convective split from the lagged flux and relaxes both flux and
profile with a fixed `alpha`, and add noise to an analytic flux. Their test is
Shestakov's `D = (p'/p)^2` problem with S0 = 1, a = 0.1, P0 = 0.01. The noise is
of two kinds:

* spatially correlated (a Langevin process with correlation length 0.2), white
  in time;
* spatially uniform, correlated in time: an ARMA(3,2) fitted to the *log* of a
  GENE heat-flux trace, which is bursty and right-skewed with `tau ~ 11.7`
  samples, exponentiated with a convexity correction (their eq. 28), then
  averaged over `T` = a few `tau`.

**What they find.**

* In the fixed-point model `delta_{n+1} = lambda delta_n + alpha eps_n`, at the
  noise floor the rms **error scales as `alpha^(1/2) sigma` and the residual as
  `alpha sigma`**. Both scale as `T^(-1/2)`. Cost scales as `alpha^-1 T`, so
  neither knob is better on scaling alone.
* **The residual reaches its floor before the error does**, and keeps flat while
  the iterates still improve. The residual is a misleading proxy for the error,
  which is their main conclusion.
* Averaging the profile over the final iterates gives an answer visually
  indistinguishable from the exact one (their Fig. 9). They suggest averaging
  the residual over iterates to make it a better proxy, but quantify neither.
* At T = 5 tau and alpha = 0.1 they predict a few percent error at the floor,
  which their coupled GENE runs bore out at alpha = 0.3.
* Left open: near-marginal turbulence that is intermittent, with long quiescent
  periods and dominant bursts.

**How it relates.**

* **Their relaxation is a fixed-gain version of this filter.**
  `q_bar = alpha q[p] + (1 - alpha) q_bar_prev` is the update step with `K =
  alpha` for ever and no prediction step. A fixed gain is what sets their floor
  `~ alpha^(1/2)`. The filter's gain starts near 1, for speed while the state
  moves, and falls towards 1/n, for averaging once it stops. So it escapes their
  speed-against-floor trade, with kappa setting how far it falls. The prediction
  through J is what lets it average without lagging, which a pure relaxation
  cannot do. Their small `alpha` is also what stabilises Picard; Newton does not
  need it.
* **Their residual-versus-error result is ours from the other side.** The
  residual floors at 5.5 `eps` (`noise_study.py`) while the frozen-noise error
  is 0.07 `eps`, and a filtered NewtonStep test passes short of the floor.
  Neither quantity sees the noise left in the answer. A noise-aware test must
  estimate it directly.
* **Their noise is spatially correlated; QuendsFlux's is not.** Independent
  per-point noise is right for a flux-tube code like GX, where each radius is its
  own run. It is wrong for a global code like GENE-global, which Parker et al.
  coupled, and for T3D's particle flux (one draw per call, see
  `python-examples/t3d-relu/README.md`). Correlated noise averages less through
  the integral that makes u, so floors are higher, and the GP's noise model needs
  a spatial covariance.
* **Their test problem is in this tree.** `python-examples/shestakov-nonlinear`
  is Shestakov's case. Their `P0 = 0.01` keeps the edge value nonzero, which
  avoids the `n_b = 0` degeneracy that example's README discusses. Their noise
  models are a ready second benchmark.

## The target: a `NoisyTransportSystem`

What the branch is aiming at, as the user described it on 2026-10-17:

* MaNTA talks only to a `NoisyTransportSystem`. Everything below is hidden
  behind `ComputePhysics` and `ComputePhysicsDerivatives`.
* Behind it a **GP proxy** is fitted to every GX sample, with the reported error
  as the per-sample noise variance. It returns its own mean, without calling GX,
  wherever its **predicted error is small enough**.
* GX is evaluated at **information-maximising points**, not at the points MaNTA
  asks about.

What these results say about that design:

* **Choosing points is a necessity, not an optimisation.** MaNTA asks for 240
  points a call here, while T3D runs GX on about 9-65 radii. The proxy must
  interpolate in x regardless, and `EvaluationPlan` tells the case MaNTA's point
  set in advance, so the design can target exactly those points.
* **"Small enough" should be what the solver needs now.** Early iterates need
  little accuracy and the last ones the most, as the forcing term does in
  inexact Newton. The natural measure is the proxy's variance carried to u,
  compared with the current Newton correction. That is a request from solver to
  case, which no interface carries today.
* **The GP's derivatives replace differencing.** They cost no GX calls, and they
  avoid both traps above, because a GP learns J jointly from every sample,
  weighted by its error, rather than from pairs. That covers the branch's second
  aim, a black box with no derivatives.
* **The error bar is the GP's noise term.** Without it the GP must fit a noise
  level, poorly identified from few samples and harder still when it varies by
  point. With it, a flagged run is down-weighted automatically, the decisive
  case above.
* **What MaNTA itself must gain is small:** a way for the case to report its
  predicted variance at the evaluated points, so the steady test stops at the
  proxy's uncertainty, and possibly the accuracy request above. A by-product: for
  any objective G, `Var(G) ~ lambda^T Sigma_F lambda` with lambda the adjoint
  MaNTA already solves for. That gives an error bar on a result for one adjoint
  solve. It is untested.

## Open

* **Cost in flux-tube runs, not calls.** Every call here evaluates every point,
  so skipping calls and choosing points cannot show up yet.
* **The GP itself**, over (x, u, q), or over (x, q/u) as gyro-Bohm scalings
  suggest, on a fixed number of tubes per call. Compare it with the per-point
  filter at equal tube counts.
* **`NewtonJacobianReuse`** to recover the 4x that differencing costs: near
  convergence J barely changes. Not measured.
* **Richer noise:** spatially correlated (Parker et al.'s Langevin), log-normal
  ARMA with bursts, error that depends on the state, intermittency.
* **Shestakov with Parker et al.'s noise**, as a second benchmark beside the toy.
* **A stopping test on the posterior**, and whether it is calibrated: does the
  reported uncertainty cover the true error over seeds?
* **One channel only.** The guard and the filter are written for `nVars = 1`;
  several channels need the `d sigma_i / d q_j` block kept definite, as
  `blackbox.py` notes.

## Reproducing

    cd python-examples/toy-model
    python known_noise_study.py                                    # exact derivatives
    python known_noise_study.py --scenario spread irregular --kappa 0.01 0.03 0.3 1.0
    python known_noise_study.py --arms raw uniform weighted --jacobians full/build \
        --fresh-base --significance 0 0.5 --budget 800 --budgets 50 100 200 400 800

Each takes 10-30 s on six workers.
