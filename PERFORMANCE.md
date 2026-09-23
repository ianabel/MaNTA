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

**`DegreeAdaptation` is not the vehicle for this, though it does the same
transfer.** It implies `Superconvergent`, which costs `(k+2)/(k+1)` points per
sweep at every level: driven as a fixed ladder to `k = 5` it is 2.05x *better*
than direct on Jardin's own initial condition -- the best number here -- and
2.9x worse on Shestakov from a poor one, where the extra points are spent on
levels that were not the bottleneck. It also cannot start at all on Jardin from
a poor initial condition; see `TODO`. A ladder to a fixed target degree needs no
error estimate and so needs no superconvergence, and that is the thing to build
if this is built.

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
