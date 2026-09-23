# Performance criteria

MaNTA is designed to be used with expensive flux and source functions.

MaNTA's performance should be determined by the number of calls to the flux/source
functions required to achieve a given accuracy.

MaNTA should be tuned to work at low-to-moderate (4-10 cells, polynomial order 2-5)
spatial resolution. Tradeoffs that increase the accuracy at high resolution but
increase the number of calls into a `TransportSystem` at low spatial resolution
(4 cells, polynomial order 2-3) should be viewed skeptically and generally exist
only as options, not defaults.

MaNTA's performance should be compared to the algorithms described in
[`refs/`](refs/Refs.md). The three benchmarks under
[`python-examples/`](python-examples/) do that, each reporting evaluations of the
physics per point.

Where only the steady state is wanted, `SteadyStateSolver` chooses how it is
reached, and the choice is worth an order of magnitude -- but only once
`NewtonJacobianReuse` is out of the way, which on these fluxes matters more than
the choice of method does. Visits per collocation point, measured on
2026-09-22, for answers identical in every printed digit:

| benchmark | `TimeMarch` | `PseudoTransient` 10 / 1 | `Newton` 10 / 1 |
|---|---|---|---|
| `park-convergence` | 120 | 10 / 10 | 5 / **5** |
| `jardin-critical-gradient` | 183 | 137 / 22 | 161 / **15** |
| `shestakov-nonlinear` | 257 | 621 / 44 | 646 / **37** |

The two numbers in each of the last two columns are `NewtonJacobianReuse = 10`,
which is KINSOL's default, and `= 1`, a fresh Jacobian factorisation at every
inner iteration. Park on 4 cells, the other two on 10, all at `k = 3` -- not
each example's own `run.conf` resolution, which differs in `k`.

`TimeMarch` sizes every step from a local error estimate on a transient that is
then discarded, which is where the factor goes; Park's own solver reaches the
same state in 9-15 iterations on the nonlinear problems he reports, which is
where `Newton` lands.

**What a steady solve spends, sweep by sweep**, since that is the budget the
table is made of. Two sweeps are fixed -- one building `sigma` from the initial
condition, one evaluating the residual to test whether the initial state is
already converged -- and then each continuation step costs KINSOL's residual at
each end of it plus one Jacobian. A step at finite `dt` costs one more, because
KINSOL drives the *damped* residual to zero and the SER schedule needs the
steady one, which has to be evaluated separately; at `dt = inf` the two are the
same function at the same state and the second evaluation is served from what
KINSOL already computed. So `Newton` is `2 + 3n` visits for `n` steps and
`PseudoTransient` is `2 + 4n`, and `park-convergence`'s 5 is `2 + 3*1`.

**Three of those five are irreducible for a Newton method and one is not
overhead at all.** A scheme that does not know the problem is linear has to
evaluate the residual to form a right-hand side and again to learn the
correction was exact; that is the floor, and it is why Park's direct solve of
the same linear system costs one pass where this costs three. The
already-converged test is the fifth, and on the workload this solver is built
for -- a parameter sweep resuming from a neighbouring answer -- it is frequently
the only sweep a solve pays.

**At the default reuse, `shestakov-nonlinear` looks like a counter-example, and
it is not.** Continuation appears to cost 2.4x what time marching does there; at
reuse 1 it costs a sixth, and both residual-driven methods beat time marching on
all three benchmarks. Reuse is not a trade on a flux like `D0 q^3/u^2` or a
`chi(q)` with a square root switching on at a critical gradient: the stale
Jacobian points somewhere useless, and the extra inner iterations it buys push
the *total* assembly count up as well, so it loses on both axes at once. Jardin
under `Newton` costs 5680 flux and 760 derivative calls at reuse 10 against 360
and 240 at reuse 1. **If a steady solve is slow or will not converge,
`NewtonJacobianReuse = 1` is the first thing to try.** The default of 10 is
retained for the opposite case, a model whose Jacobian is finite-differenced at
`(1 + n_p)` residuals per assembly, which genuinely would rather have the
iterations.

## Warm-starting a cold solve by climbing the degree

Measured 2026-09-22, `SteadyStateSolver = Newton`, `NewtonJacobianReuse = 1`,
Park on 4 cells and the other two on 10. "Nested" means solving at `k = 1` and
then at each degree up to 5, each level resuming from the previous one's answer
through a restart file; "direct" is a single cold solve at `k = 5`. Both reach
the same answer. Model calls, and the ratio against direct:

| benchmark | initial condition | direct `k = 5` | nested 1..5 | |
|---|---|---|---|---|
| `park-convergence` | either | **120** | 400 | 3.3x worse |
| `jardin-critical-gradient` | the case's own | 900 | **660** | 1.4x better |
| `jardin-critical-gradient` | deliberately poor | 1860 | **940** | 2.0x better |
| `shestakov-nonlinear` | the case's own | 2460 | **2040** | 1.2x better |
| `shestakov-nonlinear` | deliberately poor | 6420 | **3830** | 1.7x better |

**The mechanism is the target level, and it is worth a factor of five to
seventeen there.** Jardin from a poor start takes 14 Newton iterations at
`k = 5` cold and **zero** warm -- it trips the already-converged early return,
at two sweeps -- because its exact steady state is linear, so a converged
`k = 1` answer is already the answer. Shestakov goes from 51 iterations to 3.
What is paid for that is the coarse levels, which is why the ratios are 1.2-2.0
and not 5-17.

**On a linear problem it is a 3.3x loss, every time, and that is structural.**
Newton is exact in one step from any initial guess, so there is nothing for a
warm start to save and every coarse level is pure overhead. A degree ladder can
therefore never be a default; it is a bet that the flux is nonlinear enough to
repay it.

**Sequencing the tolerance across levels -- solving the coarse ones loosely --
is not reliably a win.** Three ramps were measured against solving every level
to the final `1e-11`, and the best choice differs by problem, with a spread up
to 1.9x:

| benchmark | IC | full `1e-11` | ramp from `1e-2` | ramp from `1e-6` |
|---|---|---|---|---|
| `jardin` | own | **660** | 920 | 710 |
| `jardin` | poor | **940** | 1200 | 990 |
| `shestakov` | own | 2540 | **2040** | 2300 |
| `shestakov` | poor | 5380 | 7240 | **3830** |

Two mechanisms make it backfire, and they pull in opposite directions. On
Jardin, a *fully* converged coarse level is what makes every level above it exit
at zero Newton iterations; loosening it destroys that, and each level then pays
one iteration and four sweeps instead of two sweeps. On Shestakov from a poor
start, a loosely converged answer projected up is a *worse* initial guess than a
well-converged one: `k = 2` took 46 iterations at full tolerance, 81 with three
rejected steps at a `1e-2` ramp, and 25 at `1e-6`. So the refinement is the win
and the tolerance sequencing is not.

**Refining the mesh pays more than raising the degree, and doing both pays
most.** Same runs as above, target 10 cells at `k = 5` for the two nonlinear
cases and 4 cells at `k = 5` for Park:

| benchmark | IC | direct | k-ladder | h-ladder | h+k ladder |
|---|---|---|---|---|---|
| `park-convergence` | either | **120** | 400, 0.30x | 210, 0.57x | 170, 0.71x |
| `jardin` | own | 900 | 660, 1.36x | 420, 2.14x | **252, 3.57x** |
| `jardin` | poor | 1860 | 940, 1.98x | 588, 3.16x | **260, 7.15x** |
| `shestakov` | own | 2460 | 2540, 0.97x | **1668, 1.47x** | 1784, 1.38x |
| `shestakov` | poor | 6420 | 5380, 1.19x | 2208, 2.91x | **1692, 3.79x** |

The `h` ladders are `2, 4, 6, 10` cells and the combined ones
`(2,k1) (4,k2) (6,k3) (10,k5)`. Coarsening the mesh is the better lever because
it cuts the points per sweep linearly in the cell count where dropping `k` from
5 to 1 only cuts them threefold, and the coarse mesh still carries the
nonlinearity that the ladder exists to work through. The combined ladder wins on
three of the four nonlinear rows and is the only one that makes Park's loss
tolerable rather than severe.

Two things to keep in mind when building one. **A rung costs a prologue and a
Newton solve whatever it achieves**, so the fewest rungs that keep each solve
convergent beats a rung per refinement -- measured separately on `(1 + u^2) q`
at `k = 3` to 16 cells from a poor start, a two-rung ladder `4, 16` cost 816
model calls against 936 for `2, 4, 8, 16` and 3456 direct. And **the exact
steady state must not be a low-degree polynomial or the measurement is
vacuous**: a converged coarse solution is then exactly the fine-mesh solution as
well, every rung above the first exits at zero Newton iterations, and the ladder
looks free. That is why `jardin` shows `0N` at every rung above the first -- its
exact steady state is linear -- and it is the more representative `shestakov`
numbers that should be read as typical.

**`DegreeAdaptation` is not the vehicle for this, though it does the same
transfer.** It implies `Superconvergent`, which costs `(k+2)/(k+1)` points per
sweep at every level: driven as a fixed ladder to `k = 5` it is 2.05x *better*
than direct on Jardin's own initial condition -- the best number here -- and
2.9x worse on Shestakov from a poor one, where the extra points are spent on
levels that were not the bottleneck. It also cannot start at all on Jardin from
a poor initial condition, for the reason below. A ladder to a fixed target
degree needs no error estimate and so needs no superconvergence, and that is
what `DegreeLadder`/`GridLadder` are (`docs/running.rst`).

**Do not loosen the early rungs.** Measured on the h+k ladder, against solving
every rung to the final `1e-11`:

| benchmark | IC | all `1e-11` | ramp `1e-2` | ramp `1e-4` | ramp `1e-6` |
|---|---|---|---|---|---|
| `jardin` | own | **252** | 516 | 316 | 280 |
| `jardin` | poor | **260** | 524 | 324 | 288 |
| `shestakov` | own | 1784 | 1736 | 1740 | 1776 |
| `shestakov` | poor | 1692 | 1776 | 1784 | **1684** |

Jardin loses a clean factor of two at the loosest ramp and loses at every ramp;
Shestakov is a wash, a few per cent either way with no pattern. The mechanism is
visible in the rung trace: at full tolerance Jardin's rungs above the first exit
at **zero** Newton iterations, two sweeps apiece, through the already-converged
test -- and a *fully* converged coarse rung is what earns that. Loosen it and
each of those rungs pays an iteration instead, which costs more than was saved
below.

The general point is that the classical nested-iteration advice -- solve each
level only to its own discretisation error -- does not transfer. It assumes the
coarse levels are where the work is. Here a good ladder makes them cheap by
construction (two cells at `k = 1` is a few per cent of the budget), so there is
almost nothing to save by under-solving them, and the thing they buy above them
is worth much more. Hence no per-rung tolerance key.

## Superconvergence narrows the Newton basin

`Superconvergent = true` cannot reach a steady state on
`jardin-critical-gradient` from a perturbed initial condition -- ten runs out of
ten across `k = 1..5` and both residual-driven solvers, with `dt` damping to
1e-107 while the residual stalls near 4e-2 -- where the plain method converges
in all ten. Diagnosed 2026-09-22, and it is **not** a wrong Jacobian block:

* **The linearisation is right where the solve fails.** Differencing the
  residual at that exact state, on that flux, gives `||J dy - g|| / ||g||` of
  5.7e-8 to 1.8e-7 with the flag on -- matching the plain method to the last
  digit, and marginally better -- at `cj = 0` as well as `cj != 0`, at `k = 2`
  and `3`. `the_superconvergent_jacobian_is_right_where_its_own_solve_fails`
  pins it.
* **It is the curvature of the flux in `q`, not the size of its derivative.**
  Jardin's `chi = chi0 + kappa(|q| - qc)^alpha` fails with the flag on at
  `alpha <= 0.6` and converges from `0.7` up; capping `dchi/dq` with a
  *continuous* regularisation puts the threshold between 158 and 50. Yet
  `alpha = 1` reaches `|d sigmahat/dq| = 474` and is perfectly happy, because
  there the second derivative is zero. What matters is how fast the tangent
  diffusivity varies, which is what bounds a Newton basin -- not how large it
  gets.
* **Damping fixes it, and costs nothing.** Pseudo-transient continuation at
  `PseudoTransientInitialStep = 1e-4` converges with the flag on in 5090
  transport-model calls against 5280 with it off.

Two practical consequences. The default `PseudoTransientInitialStep = 0` means
"use `delta_t`", which on these configurations is 1e4 and so is no damping at
all -- which is why `PseudoTransient` appeared to fail here too, and why naming
a genuinely small first step is the first thing to try on a stiff flux from a
poor guess. And since `DegreeAdaptation` implies `Superconvergent`, an adaptive
run can fail where a fixed-degree run of the same problem succeeds; the remedy
is the same.

A caution on measuring this, since the first attempt got it backwards. A
regularisation written as `chi0 + kappa(t + eps)^alpha` above the threshold is
not a regularisation: it makes `chi` *jump* by `kappa eps^alpha` at the
threshold rather than smoothing it, and it left the failure in place at every
`eps`, which read as evidence that the derivative was not the cause. Subtracting
`kappa eps^alpha` restores continuity, and the threshold behaviour above is what
that shows.

`TimeMarch` stays available for a reason that is not cost: a problem with more
than one steady state selects a branch by following the physics, where the other
two select whichever branch the initial guess lies in the basin of. It also
remains `shestakov-nonlinear/run.conf`'s setting, since that case exists to
document a boundary rather than to be fast. See `docs/running.rst`.

The Jardin row was remeasured when that case moved its axis condition from a
Neumann gradient to a mixed zero-flux one, which is cheaper across all three
modes while giving the same answer to round-off. A boundary condition is not
usually a performance knob; it is here because the gradient form imposed a
constraint the problem does not have, and the solver spent iterations
reconciling it.

The numbers above are reproduced by `python-examples/paper-figures/`, which also
holds the scripts behind the figures in `paper/`.
