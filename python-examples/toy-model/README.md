# Toy model

Nonlinear diffusion with a localised source and a closed-form steady state,
written vectorised as the test problem for a surrogate flux. In the form MaNTA
integrates, `d_t u - d_x[sigma_hat] = S`,

    sigma_hat = A u^(-3/2) du/dx,         S(x) = D exp( -(x - C)^2 / B )

    u'(0) = 0,   u(1) = U1,   u(x, 0) = U1

with `A = 6`, `B = 0.02`, `C = 0.3`, `D = 50`, `U1 = 0.3`. The diffusivity
falls as `u` rises, so the source heats the core against a conductivity that
weakens where it is hottest.

## Running it

    pip install .            # from the repository root, once
    cd python-examples/toy-model
    manta run.conf           # time march to t = 0.5: 12 cells, k = 3
    python toy_model.py      # steady solves against the exact steady state

`run.conf` writes `run.nc` and `run.restart.nc`. By `t = 0.25` the profile is
within the discretisation error of the steady state.

## The steady state

With `w = u^(-1/2)`, the steady equation integrates twice in closed form:

    u(x)^(-1/2) = U1^(-1/2) - G(x),   G(x) = (1/2A) int_x^1 int_0^s S(r) dr ds

`ExactSteadyState` in `toy_model.py` evaluates `G` with error functions, and
agrees with nested quadrature to 5e-16. `python toy_model.py` solves for the
steady state at k = 3 on refining meshes and prints the error against it:

| cells | relative L_inf error |
|---|---|
| 6 | 3.5e-2 |
| 12 | 2.7e-3 |
| 24 | 1.7e-4 |
| 48 | 1.1e-5 |

That is order 4 = k + 1. The six-cell row is under-resolved rather than
wrong: those cells are 0.17 wide against a source of width `sqrt(B)` = 0.14,
which is why `run.conf` uses twelve.

Every degree from 1 to 5 reaches k + 1 once the source is resolved. The L2
error is by 12-point Gauss quadrature in each cell, the L_inf error as above,
both relative, at tolerances of 1e-10 (they move nothing in the third figure
from 1e-8 to 1e-12):

| k | L2 at 192 cells | L2 order, 48 to 96 to 192 | L_inf order, same |
|---|---|---|---|
| 1 | 3.1e-4 | 2.00, 2.00 | 2.01, 2.00 |
| 2 | 1.6e-6 | 3.02, 3.01 | 3.01, 3.00 |
| 3 | 8.7e-9 | 4.03, 4.01 | 4.00, 4.01 |
| 4 | 4.1e-11 | 5.00, 5.01 | 5.00, 4.99 |
| 5 | 1.7e-13 | 5.99, 5.98 | 5.99, 5.97 |

**Measure the L_inf error inside every cell, up to both faces.** `u_h` is
discontinuous at a face, the error is largest there, and the two one-sided
limits differ -- at 12 cells, k = 3, 2.7e-3 from the left, 2.6e-3 from the
right, 2.1e-3 in the interiors. A sample point placed *on* a face measures
whichever cell rounding puts it in, and one fixed set of points for every mesh
misses the maximum on the coarse ones; between them, measured that way, the
orders here scattered from 2.8 to 5.4 at k = 2 and k = 4.

## The interface

The case supplies `ComputePhysics` and `ComputePhysicsDerivatives`, which take
every point of a batch in one call, rather than the pointwise `SigmaFn` and
`Sources`. That is the shape a surrogate wants: a model fitted to the flux is
evaluated a batch at a time, so this is the place to swap one in.

## A noisy black box: `blackbox.py` and `noise_study.py`

`blackbox.py` presents the same flux the way a turbulent transport model would:
one batched call returns sigma and S at every point, sigma carries stochastic
relative noise `eps` (a fresh draw per point per call), and there are no
derivatives. `BlackBoxCase` builds MaNTA's Jacobian from it by forward
differences in u and q (`full`) or in q alone (`gradients`, as TGYRO does),
and the box's exact derivatives are kept as a control (`analytic`). The
module docstring says why every build keeps `d sigma/dq` positive.

    python noise_study.py            # 528 steady solves, ~20 s on 6 workers

solves at 48 cells, k = 4 (discretisation error 2.2e-7), for `eps` = 1e-4,
1e-3 and 1e-2, eight seeds each, steps `dz` = 0.1, 1 and 10 times `sqrt(eps)`
and `SteadyStateTolerance` = 10, 20 and 50 `eps`. What it shows, with cost in
box calls:

* **The residual has a floor at `||F|| = 5.5 eps`** (4.6-5.7 over every seed
  and level). A tolerance below it is never met: the solve runs to
  `MaxRejectedSteps`, about 1000 calls -- each rejection halves the
  pseudo-time step, which ends near 1e-57 -- and stops at a noise error of
  `0.1 eps` (at most `0.2 eps`). That is what the noise allows: one realisation
  frozen at every point and solved to 1e-13 is out by 0.06-0.07 `eps`, since u
  integrates sigma over many points and averages independent draws. It is the
  number every other row should be read against.
* **A tolerance above the floor stops after 8-38 calls, but usually short of
  that accuracy.** Its noise error is anywhere from `0.1 eps` to `1.5 eps` at
  10-20 `eps`, and up to `10 eps` at 50 `eps`. The stopping test cannot tell "within the noise"
  from "not yet converged", so the error columns largely measure *where the
  test happened to stop*: a differenced Jacobian that takes more iterations
  can end *closer* than the control (1.1e-4 against 8.8e-4 at `eps` = 1e-3,
  10 `eps`). Compare Jacobians by calls, not by these errors.
* **`dz = sqrt(eps)` is the step to use.** It converged in every case, and
  `full` cost 1.7-2.6x the control's calls (15-26 against 8-14). At
  `0.1 sqrt(eps)` the solve fails from `eps` = 1e-3 up: the guard fires at
  thousands of points and the error is 30-57%. At `10 sqrt(eps)` it converges
  at similar cost, with generally larger errors.
* **The guard firing means the step is too small for the noise.** It never
  fired at `dz >= sqrt(eps)`. When it fires a solve can still report success:
  at `eps` = 1e-2 and 50 `eps`, `full` at `0.1 sqrt(eps)` *converged* in all
  eight seeds with a 35% error.
* **`gradients` against `full` is roughly even on this problem.** Dropping
  `d sigma/du` costs Newton iterations (3-11 against 3-7) and saves a call per
  build, which nets out at between 3 fewer and 7 more calls at `sqrt(eps)`. The toy's
  `u^(-3/2)` makes `d sigma/du` large; a gyro-Bohm flux may come out
  differently.

## A box that reports its error bar: `reporting.py` and `known_noise_study.py`

GX's fluxes come from QUENDS (https://github.com/sandialabs/quends), which
returns a mean and a standard error for every flux-tube run. The error varies
from run to run with how much statistically steady data the run had. A run with
no steady segment comes back with its tail mean and a 100% error. `QuendsFlux`
reproduces that on the toy flux. `FilteredCase` keeps a scalar Kalman filter at
each point: the estimate is carried between states by the flux's derivatives,
gets process noise `kappa` times the predicted change, and is updated with each
new sample. What differs between the arms is the variance assigned to a sample:
none (raw, as before), one global level (uniform), or the reported error
(weighted). `reporting.py`'s docstring has the details.
`docs/superpowers/notes/2026-10-17-noisy-flux-error-bars-notes.md` interprets
the results and relates them to the literature and to the surrogate design.

    python known_noise_study.py      # 168 solves, ~15 s on 6 workers

The study runs 48 cells at k = 4, with a median sample error of `eps` = 1e-2,
eight seeds and kappa = 0.1. Each entry is the noise error in units of `eps`,
median over seeds, at a budget of 100 calls (worst seed in brackets):

| arm | flat | spread (4x) | irregular (2% of runs) |
|---|---|---|---|
| raw | 0.052 (0.18) | 0.065 (0.25) | 0.15 (0.65) |
| uniform, true level | 0.014 (0.025) | 0.008 (0.031) | 0.023 (0.049) |
| uniform, level 3x off | 0.014-0.018 | 0.008 | 0.024-0.034 |
| weighted | 0.016 (0.027) | 0.006 (0.018) | 0.006 (0.025) |
| weighted, error bar 2x off | 0.016-0.017 | 0.006 | 0.006 |

* **Filtering is what pays: 4-10x on raw, whatever it is told.** Raw never gets
  below about 0.05 `eps`, because the last iterate carries one call's noise in
  full. Every filtered arm reaches 0.006-0.02 `eps` by 100 calls.
* **The error bar adds little when the noise is well behaved.** The arms tie
  when every run has the same error. When the error varies 4x, weighting gains
  about 1.3x, which is the `sqrt(E[s^2] E[1/s^2])` the spread predicts.
* **The error bar earns its keep on flagged runs.** When 2% of runs fail to
  reach a steady state, raw is wrecked (0.15-0.34 `eps`) and uniform weighting
  is 3-4x worse than weighted. The weighted arm is unaffected, because a 100%
  error gives the bad sample almost no weight.
* **Only the relative weights matter.** Misreporting the error bar by 2x either
  way, or assuming a global level 3x off, moves almost nothing.
* **kappa has a wide plateau.** Anything from 0.03 to 0.3 gives these numbers.
  At 0.01 the filter trusts old samples too long and lags (0.1-0.4 `eps` at
  25-100 calls); at 1 it hardly averages.

**The NewtonStep test cannot judge a filtered flux.** It passes at rtol 1e-6 in
most filtered runs, while the real error is about 7e-5. The filter makes F
change slowly between calls, so successive Newton corrections become small
whether or not the noise has been averaged away. The clearest case is kappa =
0.01: the test passes on every seed at 0.05 `eps`. A stopping test for this
setup has to read the filter's own posterior variance.

### Without derivatives

The table above gives MaNTA and the filter the box's exact derivatives. A real
box has none, so `--jacobians full/build` differences instead: forward
differences in u and q at `dz = sqrt(eps)` for Newton, with the filter reusing
each build's derivatives. Each build costs three calls. The entries below are
the noise error in units of `eps` at 800 calls, median (max), for eight seeds:

    python known_noise_study.py --arms raw uniform weighted --jacobians full/build \
        --fresh-base --significance 0 0.5 --budget 800 --budgets 50 100 200 400 800

| arm | guard | flat | spread | irregular |
|---|---|---|---|---|
| raw | -- | 0.063 (0.10) | diverges in at least half | diverges |
| uniform | -- | 0.015 (0.024) | 0.014 (0.031) | diverges |
| weighted | -- | 0.011 (0.023) | 0.012 (0.030) | diverges |
| uniform | 0.5 | 0.013 (0.024) | 0.010 (0.038) | diverges |
| weighted | 0.5 | 0.010 (0.023) | 0.014 (0.018) | 0.009 (0.024) |

Diverging here means errors of 1e3 `eps` and up, with u driven negative.

* **The filter still gains 4-6x on raw.** Differencing makes it slower: the
  error reached at 200 calls with exact derivatives takes about 800 here.
* **Difference from a fresh sample.** With the default base, the last value
  call's sample, the filtered solve drifts back up to 0.074 `eps` by 800 calls
  in the flat case. Neither a systematic error in J (11-20% in d sigma/du) nor
  independent random errors of 30% do this; only the correlation between the
  base sample's noise and the filter's estimate does. `reporting.py` has the
  mechanism.
* **One flagged run in a difference wrecks the Jacobian, and only the error bar
  sees it.** A biased sample divided by `h` makes a wild entry. Newton follows
  it and the solve diverges, filtered or not. The significance guard keeps the
  previous derivative wherever the difference's standard error exceeds half its
  value. With the reported error that rejects the flagged differences. With one
  global level it cannot, and the solve diverges as before.
