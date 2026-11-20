# Error-controlled transport with noisy flux models

**Status as of 2026-11-19.** Branch `feat/noisy-transport-system`, after the
merge of `main` (`1b7f5d0`).

This document explains what we are building and why. It sets out the problem
of coupling a transport solver to turbulence simulations, the approach MaNTA
takes, how our test noise compares with the noise models in the literature,
what the measurements so far show, and where the work stands. The last section
points developers to the code and to the detailed records.

## Summary

The aim is to predict plasma profiles by coupling MaNTA directly to turbulence
simulations such as GX, rather than to reduced models. Their fluxes are time
averages of chaotic signals, so every evaluation is noisy and expensive, and
the solver can never drive its residual to zero. We reframe "converged" in
terms of what the user actually wants: **an output, such as the stored
energy, to a stated accuracy, at the least cost.** MaNTA estimates the error in
that output in three parts (discretisation, incomplete convergence, and noise)
using the adjoint of the output. It splits the tolerance between those parts,
and tells the flux model how accurate each evaluation needs to be.

The deterministic half of this (phases 1-4) is built and on `main`, and it is
useful for every existing case. The noisy half (phases 5-7) is designed and
backed by two benchmarks, but not yet built.

## Why we want this

### Coupling to turbulence simulations, and the noise that comes with it

A transport solver evolves 1-D profiles on slow timescales. The turbulent
fluxes that drive them come from much faster, smaller-scale physics. Most
integrated modelling gets those fluxes from surrogates: analytic models,
quasilinear models such as TGLF or QuaLiKiz, or neural-network fits to those.
They are fast, deterministic, and of uncertain accuracy outside the regimes
they were built for [1]. The alternative is to compute the flux by direct
simulation of the turbulence, as Trinity [3], TGYRO [4], Tango [16] and T3D
[11] do. This is now affordable for a 1-D profile on modern machines, at the
cost of thousands of turbulence runs per prediction.

Direct simulation brings a problem surrogates do not have. **A turbulence
simulation does not return a single flux.** It returns a fluctuating time
series, and the transport solver gets a time average over a finite window.
Every evaluation therefore carries a statistical error, which can be large:
heat-flux bursts several times the mean are routine [1]. Averaging longer
reduces the error as `T^(-1/2)`, at proportional cost. The solver's residual
has a floor set by the noise, and below that floor it carries no information.

Two facts make this a design problem rather than a nuisance:

* **The error can be measured, not just tolerated.** QUENDS [10], the analysis
  layer we use for GX, reports a standard error with every time-averaged flux.
  It estimates that error from block means of the steady part of the series,
  and it flags runs that never reached a steady state. A flux model that comes
  with an error bar can be told how accurate to be.
* **Not every evaluation needs the same accuracy.** Early iterates are far from
  the answer and need little accuracy. The output the user cares about is
  sensitive to some radii and some channels much more than to others. A fixed
  averaging time everywhere spends effort where it does not matter.

### How existing approaches handle the noise

Existing coupled codes manage noise with fixed knobs. Tango, following
Shestakov et al. [2], relaxes the flux and the profile with a fixed factor
`alpha` and averages over a fixed time `T`. Parker, LoDestro & Campos [1]
analysed exactly that scheme under synthetic noise, and their results are the
starting point here:

* At the noise floor the error scales as `alpha^(1/2) T^(-1/2)` and the
  residual as `alpha T^(-1/2)`, while the cost scales as `alpha^(-1) T`. "In
  terms of scaling alone, neither parameter is a clearly superior control
  knob" [1, Sec. 8], and `alpha` must also be small enough to stabilise the
  Picard iteration.
* **The residual reaches its noise floor before the error does**, and stays
  flat while the iterates are still improving. "The residual may not be a good
  proxy for the error of the solution" [1, Sec. 8].
* Averaging the profile over the last iterates gives an answer visually
  indistinguishable from the exact one [1, Fig. 9].

Both lessons hold in our own measurements (below). Three consequences shape
the design:

1. A stopping test on the residual cannot tell "converged to within the noise"
   from "not converged yet". The test needs a direct estimate of the noise left
   in the answer.
2. A fixed gain trades speed against the noise floor. A gain that starts near 1
   and falls as the iterates settle avoids that trade, which is what a Kalman
   filter does.
3. The accuracy asked of the flux model should vary: with the stage of the
   solve, and with how much each evaluation matters to the output.

### What a user actually wants: an output to an accuracy

A user of a transport code rarely wants the whole profile to a uniform
tolerance. They want the fusion power, the stored energy or a confinement
time, to a stated accuracy. Tolerances on the whole solution say nothing about
which output matters, and they fail in a specific way. In T3D's ReLU
benchmark [11], ported to MaNTA, the steady-state test weighed every row of
the residual equally. The density's rows contribute almost nothing at steady
state, so a solve stopped at `||F|| < 3` left the density 2.6% wrong, and
still 1.9% wrong at `||F|| = 0.03`. Under noise the residual floored near 2.5,
250 times above the level at which the density converges.

Asking for an output to an accuracy fixes the stopping problem, and it also
gives the noise a target. The adjoint of the output shows how each flux
evaluation affects it, so the statistical error in the output can be computed,
and an accuracy budget can be divided among the evaluations.

### Why the deterministic half is worth having on its own

Everything except the noise is useful for deterministic cases, which is every
case MaNTA runs today:

* stopping a steady solve when the outputs that matter are converged, rather
  than when a norm of the whole residual is small;
* an estimate of the discretisation error in each output, with no closed form
  needed;
* adapting the polynomial degree and the mesh to an output, rather than to the
  solution as a whole;
* access from Python to the adjoint and to the output's sensitivity to the
  physics at each point. That sensitivity says where a model's uncertainty
  matters, which is useful whether or not the model is noisy.

That is why this half was built and merged first, on its own branch. It could
also be checked against closed forms, without noise or GX.

## The approach

### Three parts of the error in an output

For an output `G = int g dx`, the value MaNTA reports differs from the true one
in three parts. The user's tolerance `eps_G` is divided between them
(`ObjectiveBudget`).

| Part | Cause | Estimate |
|---|---|---|
| discretisation | finite mesh and degree | `eta_disc`: the residual of the discrete solution in a richer scheme, weighted by that scheme's adjoint |
| algebraic | the solve stopped before `F = 0` | `z^T F = (dG/dy) . J^-1 F`, read off the Newton correction |
| statistical | noise in the flux and source evaluations | `sqrt(sum_m w_m^T Sigma_m w_m)` |

This is the dual-weighted residual method of Becker & Rannacher [12], and the
target-based adaptation of Woopen et al. [8], applied to MaNTA's HDG scheme and
extended with a statistical term. `z` is the adjoint of `G`. The vector `w_m`
is the sensitivity of `G` to the physics outputs at evaluation point `m`, and
`Sigma_m` is their covariance there.

**The sensitivity `w` is the link between the solver and the flux model.** The
residual depends linearly on each flux value through the interpolation onto
the cell's basis. So `w_m = -z^T dF/dGamma_m` comes from the adjoint for no
extra cost, and the variance of `G` follows from the flux model's reported
covariances.

### Asking the flux model for the accuracy it needs

If an evaluation at point `m` with standard error `s_m` costs `c_m / s_m^2`
(a time average: `s ~ T^(-1/2)`), the cheapest set of targets that keeps
`Var(G) <= eps_noise^2` is

```
s_m^2 = eps_noise^2 sqrt(c_m) / (|w_m| sum_j sqrt(c_j) |w_j|)
total cost = (sum_j sqrt(c_j) |w_j|)^2 / eps_noise^2
```

This is a constrained minimisation with one Lagrange multiplier. Points the
output is sensitive to get tight targets, and the rest get loose ones. **The
targets also tighten as the solve converges**, following the forcing term in
inexact Newton methods [13]: the share used at a call is
`max(eps_noise, eta |z^T F|)`. Early iterates can be served by short runs, or by
a surrogate's prediction.

Compare this with the fixed knobs above. Parker et al.'s `T` is one run length
everywhere, for the whole solve. Here the run length is chosen per point and
per iteration, from what the output needs.

### Stopping under noise

A solve stops when both conditions hold: `|z^T F|` is below its share, and
`sqrt(w^T Sigma w)` is below the noise share. `z^T F` is itself noisy, with
the statistical error as its floor, so its share must not be set below that
floor. A target the flux model reports it cannot meet ends the solve with an
outcome that says so, rather than hanging.

### Architecture

```
 MaNTA core (C++)                        error estimates, tolerance, budget,
   |                                     stopping, adaptation, accuracy requests
   |  interface 1: NoisyTransportSystem  (reports nothing: today's TransportSystem)
   v
 a flux system (Python)                  direct model | filter | GP surrogate | ...
   |
   |  interface 2: expensive model       (used by surrogates only)
   v
 GX + QUENDS, or a test box
```

**MaNTA's core holds everything that needs the Jacobian, the adjoint or the
discretisation, and nothing about how a flux model meets a request.** The
reasons:

* Two implementations of interface 1 exist already as prototypes: the toy
  model's filter, and a direct model wrapper. They test the interface before
  any surrogate exists.
* Surrogates change faster than the solver core. Kernels and sampling rules
  belong in Python (JAX or GPyTorch), outside the C++ core and its CI
  requirements.
* The core can be verified without GX, and much of it without noise.

On each call, MaNTA sends the flux system the points and states, the accuracy
targets `s_m`, and the functional (`w`, `eps_noise`). A direct model uses the
targets to choose a run length per flux tube. A surrogate uses the functional
to decide where to run the model. The system returns means, covariances,
derivatives when asked, a flag per point (QUENDS's "no steady state"), and the
cost. **A system that reports nothing is today's `TransportSystem`**, so every
existing case is untouched.

### Why a Gaussian-process surrogate is the end point

A turbulence code is too expensive to run at every point MaNTA evaluates: 240
points per call in our toy problem, against the 9-65 radii at which T3D runs
GX. A surrogate is needed to interpolate in any case. A Gaussian process (GP)
fitted to every sample, with QUENDS's error bar as each sample's noise
variance, provides three things:

* **Predictions with an error bar.** Where its predicted error already meets
  the target, it answers without running the model.
* **Derivatives without differencing.** It learns the Jacobian from all the
  samples together, weighted by their error bars. That avoids the two traps
  differencing falls into under noise (see "What we have measured").
* **A rule for where to run next:** wherever a run most reduces
  `Var(w^T Gamma)`, the variance in the output.

The filter we have built is the first step. A Kalman filter is GP regression
carried out one sample at a time, for a random walk along one dimension
[14, 15]. The GP adds sharing between points, which lets it predict where it
has not sampled.

## Our noise models compared with Parker et al.'s

The comparison matters because the noise model decides what a benchmark can
show. Parker et al. [1] used two synthetic models on Shestakov's problem [2].
We use two of our own, plus T3D's. Everything about Parker et al.'s models
below is taken from the paper (arXiv:1807.04779).

| | Parker et al., Sec. 6.1/7.2 | Parker et al., Sec. 6.2/7.3 | `QuendsFlux` (toy model) | `blackbox.py` (toy model) | T3D ReLU benchmark |
|---|---|---|---|---|---|
| What it stands in for | a global code | GENE ion heat flux | GX + QUENDS, flux tubes | a generic noisy model | T3D's regression test |
| Applied to | flux, multiplicative: `q(1 + W z)` (eq. 37) | flux, log-normal: `q exp(W n - W^2 sigma^2/2)` (eq. 38) | flux, additive with a relative scale | flux, multiplicative: `sigma(1 + eps xi)` | flux, multiplicative: `(1 + 0.2 U)` |
| Distribution | Gaussian | log-normal, right-skewed, bursty | Gaussian mean; per-point error from a log-uniform effective sample size | Gaussian | uniform on ±1 |
| In space | correlated: discrete Langevin (AR(1)), length 0.2 | perfectly correlated | independent per point | independent per point | heat flux: independent per point. Particle flux: one draw for the whole profile |
| In time (between solver calls) | white | ARMA(3,2) fitted to the log of a GENE trace, `tau ~ 11.7` samples; averaged over `T = 3-20 tau` | white | white | white |
| Typical size | log variance not stated; `sigma^2` = 1e-4 to 1e-2 in Fig. 6 | `sigma^2` = 0.05 to 0.21 for the log; before averaging, 0.21 is about 48% relative (our arithmetic) | median 1e-2, varying 4x between points | 1e-4 to 1e-2 | ±20% (standard deviation 12%) |
| Error bar given to the solver | no | no | yes, itself estimated (chi factor) | no | no |
| Failed runs | none | none | a fraction biased by up to ±50%, reported at 100% | none | none |
| Boundary treatment | taper `W(x)` to zero at the edges | taper `W(x)` | none | none | none |

**What we share with Parker et al.** Their test problem, Shestakov's
`D = (p'/p)^2` diffusion with `S0 = 1`, `a = 0.1`, `P0 = 0.01` (eq. 29-34), is
in this tree as `python-examples/shestakov-nonlinear`. Their `P0 = 0.01` keeps
the edge value nonzero, which avoids the degeneracy at `n_b = 0` that the
example's README describes. Their noise is also relative to the flux, as ours
is.

**Where the models differ, and why it matters:**

* **Spatial correlation is the biggest difference, and it follows from the
  code being coupled.** GX is a flux-tube code: each radius is a separate run,
  so noise is independent across radius. Parker et al. modelled a global code,
  GENE-global, where neighbouring radii come from one simulation and are
  correlated. They also report that their two models have "either finite
  spatial or temporal correlation, but not both" [1, Sec. 6.2]. This matters
  for the floor. The solution integrates the flux over radius, so independent
  noise averages down through that integral and correlated noise does not. A
  benchmark with correlated noise will show higher floors, and a GP for a
  global code would need a spatial covariance. T3D's particle-flux noise, one
  draw for the whole profile, is an artefact of its unvectorised test model,
  not a property of GX.
* **Correlation between channels at one radius.** In a flux-tube run the heat
  and particle fluxes are averages of the same turbulent time series, so their
  errors are correlated. Parker et al. had one channel, so the question did
  not arise. Our boxes do not model it yet, and QUENDS reports a standard error
  per channel without the cross-covariance.
* **Burstiness and skew.** Parker et al.'s log-normal ARMA model is the more
  realistic description of the signal *before* averaging. QUENDS averages
  until its blocks pass a Ljung-Box test, so what reaches MaNTA is closer to
  Gaussian, as the central limit theorem argument in [1, Sec. 4] predicts.
  The skew survives where runs are short, and it is one of the open noise
  features below.
* **Correlation between solver calls.** Their ARMA model makes successive
  samples correlated when `T` is not much larger than `tau`. Our boxes draw
  independently at every call. That matches a flux-tube run restarted per
  call, but not a run continued from the previous state, which is what a
  surrogate sampling one point repeatedly would see.
* **The error bar.** This is the difference that matters most for the design.
  Parker et al.'s solver is given only the noisy value. Our `QuendsFlux` gives
  the error bar too, including its own uncertainty and the failed-run flag, so
  that we can measure what knowing the error is worth. Their analysis has no
  counterpart to the failed run, and our measurements show that this is where
  the error bar matters most.
* **Size.** Their GENE-like model, after averaging over `T = 5 tau`, is roughly
  20% per sample (our estimate from `sigma^2 = 0.21`), the same order as T3D's
  ±20%. Our toy studies go to 1e-2, an order of magnitude smaller. The toy
  results should be read as statements about mechanisms, and the T3D
  benchmark as the realistic noise level.

**Planned:** Shestakov's problem with both of Parker et al.'s noise models, as
a second benchmark beside the toy. It is the only way to compare the filter and
the GP with their relaxation under identical conditions. It would also show
how the design behaves with spatially correlated noise, which our boxes never
produce.

## What we have measured so far

Error in units of the median sample error `eps`. Toy model: 48 cells, `k = 4`,
eight seeds.

**The residual and the Newton step both mislead under noise.** With raw noise
the residual floors at 5.5 `eps`, while the error the noise actually causes is
0.07 `eps`. A tolerance below the floor halves the pseudo-time step at every
rejection until it reaches about 1e-57. With a filtered flux, the Newton-step
test passes at `rtol` 1e-6 when the real error is about 7e-5: the filter makes
the flux change slowly between calls, so the corrections shrink like `1/n`
whether or not the noise has gone. This is Parker et al.'s finding [1] from the
other side. Neither quantity measures the noise left in the answer.

**A filter beats a fixed relaxation, and the error bar matters when runs
fail.** A Kalman filter at each evaluation point predicts the flux's change as
the state moves, using the Jacobian, and then averages new samples by their
variance. At 100 calls with exact derivatives:

| Arm | Flat noise | 4x spread | 2% failed runs |
|---|---|---|---|
| raw (no filter) | 0.052 | 0.065 | 0.15 |
| filter, one assumed noise level | 0.014 | 0.008 | 0.023 |
| filter, weighted by the error bar | 0.016 | 0.006 | 0.006 |

* The filter gains 4-10x over raw noise, whatever weighting it is given.
* The error bar gains nothing on uniform noise, about 1.3x when the error
  varies between points, and 3-4x when runs fail. A failed run reported at
  100% gets almost no weight. That is the GX regime: QUENDS has a fallback for
  runs with no steady state because they happen.
* Only the relative weights matter. Error bars wrong by a factor of 2 change
  almost nothing.

Parker et al.'s relaxation is the filter's update with a fixed gain `alpha` and
no prediction. The fixed gain sets their error floor of `alpha^(1/2)`. The
filter's gain starts near 1 while the state moves fast, and falls to about
`1/n` as it settles. The prediction through the Jacobian is what lets it
average without lagging.

**Differencing under noise has two traps, and only the error bar catches the
second.** Without derivatives the filter still gains 4-6x, but it needs about
four times the calls.

* A finite difference must not reuse a sample the filter has seen. The noise
  enters the derivative divided by the step size, and the filter's prediction
  feeds it back.
* One failed run inside a difference produces a wild Jacobian entry. Newton
  follows it, and the solution goes negative. Only the error bar shows the
  difference is not significant.

**On a realistic benchmark, the stopping test was the limit, not the noise.**
T3D's ReLU benchmark (three channels, W7-X geometry, ±20% noise) was ported to
MaNTA and checked against T3D: without noise, T3D converges to MaNTA's solution
as its grid is refined. MaNTA reaches T3D's N = 33 accuracy in `T_e` in 104 box
calls, where T3D needs 1352. Under noise:

* The residual test left the density 2.6% wrong (above). Replacing it with
  IDA's corrector test in per-channel error weights
  (`SteadyStateTest = "NewtonStep"`, now `main`'s default) fixed that.
  Continuation follows Kelley & Keyes [5] and Coffey, Kelley & Keyes [6].
* At a tolerance of 1e-2, every seed converges, for a fifth of T3D's 68 calls
  with exact derivatives, or about as many with T3D's own Jacobian.
* At 3e-3 with exact derivatives, only about half the seeds converge: the
  test asks for a correction smaller than the noise produces. This is the floor
  that phase 6's stopping test is designed for.

## Design decisions that were settled by measurement

Several choices in the original design were tested and changed. They are
recorded here because each says something about the problem.

* **The convergence estimate is read off the Newton step, not from a separate
  adjoint solve.** An adjoint solve at every iteration cost 36-55% more time.
  `z^T F` equals `(dG/dy) . J^-1 F`, and `J^-1 F` is the step the iteration has
  just computed, so the two agree to 1e-15 at no extra cost. On Shestakov's
  problem, the default `NewtonStep` test leaves the particle inventory 6.5e-4
  to 1.8e-3 out. The output tolerance at 1e-3 spends two more iterations and
  returns 6.4e-8 to 3.0e-6.
* **The discretisation estimate uses a richer adjoint, not a postprocessed
  one.** The postprocessed adjoint would have cost no flux evaluations, which
  matters a great deal with an expensive model. But on closed-form problems its
  effectivity (estimate divided by true error) ranged from -1.27 to 5.28.
  Solving the adjoint two degrees higher gives a median of 1.000, with 95% of
  cases within 10%. The richer adjoint is two degrees up, not one, because for
  even `k` the output already converges as fast as at `k + 1`.
* **The discretisation estimate has a known blind spot.** An error that decays
  only algebraically in the degree is underestimated, and nothing flags it. On
  Shestakov's problem the effectivity falls from 0.75 to 0.43 for `k = 1..5`,
  because the richer scheme makes the same boundary error.
* **The output decides *whether* to refine the mesh; the solution decides
  *where*.** The per-cell contributions to the estimate cancel: their absolute
  sum reached 10^6 times the total. As an indicator of where to grade the
  mesh, they gave false positives at 41 of 1217 smooth ends. In 560 adaptation
  runs, 408 met their tolerance, 151 stopped at the degree ceiling with a
  warning, and one finished outside its tolerance (on a singular wall cell).
  Degree adaptation keeps Giorgiani et al.'s rule [7].
* **No adjoint solve may evaluate the physics.** With GX, a flux evaluation is
  the whole cost. The adjoint is assembled from the derivatives of the last
  Jacobian build, and a test counts evaluations and requires zero. The
  discretisation estimate is the one exception: one evaluation at the richer
  degree per estimate. With an expensive model it will have to be made rarely,
  or taken from a surrogate.
* **The sensitivity `w` is checked point by point.** An operator error that
  sums to zero over a cell is invisible in every total. A defect of exactly
  that kind went unnoticed in the adjoint code for months, so `w` is checked by
  finite differences at each point.

## Where it stands

| Phase | What | Status |
|---|---|---|
| 1 | Adjoint and sensitivity `w` exposed to Python, with no flux evaluations | **Built, on `main`** |
| 2 | Tolerance on an output: convergence part (`SteadyStateTest = "Objective"`) | **Built, on `main`** |
| 3 | Discretisation error estimate for an output | **Built, on `main`** |
| 4 | Degree and mesh adaptation to an output | **Built, on `main`** |
| 5 | `NoisyTransportSystem`, reporting only: MaNTA reports `G ± sqrt(w^T Sigma w)` | Not started |
| 6 | Error control under noise: accuracy requests, tightening, the stopping test | Not started |
| 7 | GP surrogate, measured against the filter at equal model cost | Not started; the filter is its prototype |

The three noise measures planned before this design (an outcome for a solve
that has hit the noise floor, averaging over iterates, and a noise-level hook)
are not implemented. They map onto phases 5-6: the hook becomes interface 1's
covariance; the outcome comes from a flux model's report that it cannot meet
a target, or from `Stalled`, which exists now; and averaging over iterates
remains the cheapest fallback for a model with no error bar.

### Next steps

1. Re-run this branch's examples against `main`. They are not in CI, and the
   merge brought solver changes.
2. Decide where interface 1's hooks live (below), then build phase 5 with a
   direct model and the filter on the toy box. The test is calibration: across
   seeds, does the reported variance cover the true error?
3. Add Shestakov's problem with Parker et al.'s two noise models as a second
   benchmark.
4. Phase 6: requests and the stopping test. The toy box gains a run-length
   control (`s ~ n^(-1/2)`), with cost counted in effective samples.
5. Phase 7: the GP, against the filter at equal model cost, on the toy model,
   on Shestakov with Parker et al.'s noise, and on T3D's benchmark.

## Open questions

* **Where the hooks live.** Virtual functions on `TransportSystem` with no-op
  defaults keep one physics interface. A `NoisyTransportSystem` subclass keeps
  uncertainty optional in the type as well. Either way, a model that reports
  nothing must behave exactly as now, bit for bit.
* **The cost of the discretisation estimate under noise.** It is the one
  estimate that evaluates the physics. Should it be made rarely, taken from the
  surrogate, or simply paid for?
* **Uncertainty in the derivatives.** Newton needs the Jacobian only roughly,
  but guarding a difference against a failed run needs the derivative's error,
  and a GP can supply it. Is that part of interface 1?
* **Noise we do not model yet:** correlation between channels, log-normal
  bursts, noise that depends on the state, intermittency near marginal
  stability (which Parker et al. also leave open), and spatial correlation for
  global codes.
* **More than one channel.** The filter and the difference guard are written
  for one variable. Several channels need the flux's gradient Jacobian kept
  positive definite.
* **Time-dependent runs** are out of scope. A transient output needs an adjoint
  solved backwards in time, which MaNTA does not have.
* **Field models and scalars.** The adjoint now handles global scalars, and the
  output tolerance includes their share. The discretisation estimate still
  refuses scalars and field models.

## For developers

| Path | What it holds |
|---|---|
| `docs/adjoints.rst`, `docs/running.rst` | user documentation for phases 1-4 |
| `docs/dev/adjoints.md` | implementation notes and traps for phases 1-4 |
| `git show c0dcbce^:docs/superpowers/specs/2026-10-17-objective-error-control-design.md` | the as-built record of phases 1-4, with every measurement; removed from `main` as finished |
| `docs/superpowers/specs/2026-10-17-noisy-transport-system-design.md` | the original design draft; its "What exists" section predates phases 1-4 |
| `docs/superpowers/notes/2026-10-17-noisy-flux-error-bars-notes.md` | the filter study and the first reading of Parker et al. |
| `git show 308079d:docs/superpowers/specs/2026-10-11-steady-state-termination-design.md` | the `NewtonStep` design, and the earlier noise plan ("Noise: phase 2") |
| `python-examples/toy-model/` | the toy problem, `blackbox.py`, `noise_study.py`, `reporting.py` (`QuendsFlux`, `FilteredCase`), `known_noise_study.py` |
| `python-examples/t3d-relu/` | T3D's benchmark, the check against T3D, `noise_benchmark.py` |
| `python-examples/shestakov-nonlinear/` | Parker et al.'s test problem, without noise so far |

None of the examples run in CI (`pytest.ini` collects `python/Tests` only).

## References

1. J. B. Parker, L. L. LoDestro & A. Campos, *Investigation of a
   multiple-timescale turbulence-transport coupling method in the presence of
   random fluctuations*, Plasma **1** (2018) 126-143.
   https://doi.org/10.3390/plasma1010012 (arXiv:1807.04779)
2. A. I. Shestakov et al., *Self-consistent modeling of turbulence and
   transport*, J. Comput. Phys. **185** (2003) 399-426.
   https://doi.org/10.1016/S0021-9991(02)00063-3
3. M. Barnes et al., *Direct multiscale coupling of a transport code to
   gyrokinetic turbulence codes* (Trinity), Phys. Plasmas **17** (2010)
   056109. https://doi.org/10.1063/1.3323082
4. J. Candy et al., *Tokamak profile prediction using direct gyrokinetic and
   neoclassical simulation* (TGYRO), Phys. Plasmas **16** (2009) 060704.
   https://doi.org/10.1063/1.3167820
5. C. T. Kelley & D. E. Keyes, *Convergence analysis of pseudo-transient
   continuation*, SIAM J. Numer. Anal. **35** (1998) 508-523.
   https://doi.org/10.1137/S0036142996304796
6. T. S. Coffey, C. T. Kelley & D. E. Keyes, *Pseudotransient continuation and
   differential-algebraic equations*, SIAM J. Sci. Comput. **25** (2003)
   553-569. https://doi.org/10.1137/S106482750241044X
7. G. Giorgiani, S. Fernández-Méndez & A. Huerta, degree adaptivity for HDG,
   Comput. Fluids **98** (2014) 196-208.
   https://doi.org/10.1016/j.compfluid.2014.01.011
8. M. Woopen, A. Balan, G. May & J. Schütz, target-based hp-adaptation for
   hybridized DG, Comput. Fluids **98** (2014) 3-16.
   https://doi.org/10.1016/j.compfluid.2014.03.023
9. Superconvergence algorithm for interpolatory HDG methods, J. Sci. Comput.
   **81** (2019) 2188-2212. https://doi.org/10.1007/s10915-019-01081-3
10. QUENDS, Sandia National Laboratories. https://github.com/sandialabs/quends
11. T3D (Trinity3D). https://bitbucket.org/gyrokinetics/t3d
12. R. Becker & R. Rannacher, *An optimal control approach to a posteriori
    error estimation in finite element methods*, Acta Numerica **10** (2001)
    1-102. https://doi.org/10.1017/S0962492901000010
13. S. C. Eisenstat & H. F. Walker, *Choosing the forcing terms in an inexact
    Newton method*, SIAM J. Sci. Comput. **17** (1996) 16-32.
    https://doi.org/10.1137/0917003
14. C. E. Rasmussen & C. K. I. Williams, *Gaussian Processes for Machine
    Learning*, MIT Press (2006).
15. J. Hartikainen & S. Särkkä, *Kalman filtering and smoothing solutions to
    temporal Gaussian process regression models*, IEEE MLSP (2010).
16. Tango, LLNL. https://github.com/LLNL/tango. Coupled to GENE in
    J. B. Parker et al., *Bringing global gyrokinetic turbulence simulations to
    the transport timescale using a multiscale approach*, Nucl. Fusion **58**
    (2018) 054004.

[1]-[11] are indexed in `refs/Refs.md` (the PDFs are gitignored). [12]-[16]
are not, and their details are given from memory except [16], whose citation
is taken from [1]'s reference list.
