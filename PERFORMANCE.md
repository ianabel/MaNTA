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
2026-09-21, for answers identical in every printed digit:

| benchmark | `TimeMarch` | `PseudoTransient` 10 / 1 | `Newton` 10 / 1 |
|---|---|---|---|
| `park-convergence` | 120 | 11 / 11 | 7 / **7** |
| `jardin-critical-gradient` | 183 | 138 / 23 | 163 / **17** |
| `shestakov-nonlinear` | 257 | 622 / 45 | 648 / **39** |

The two numbers in each of the last two columns are `NewtonJacobianReuse = 10`,
which is KINSOL's default, and `= 1`, a fresh Jacobian factorisation at every
inner iteration. Park on 4 cells, the other two on 10, all at `k = 3` -- not
each example's own `run.conf` resolution, which differs in `k`.

`TimeMarch` sizes every step from a local error estimate on a transient that is
then discarded, which is where the factor goes; Park's own solver reaches the
same state in 9-15 iterations, which is where `Newton` lands.

**At the default reuse, `shestakov-nonlinear` looks like a counter-example, and
it is not.** Continuation appears to cost 2.4x what time marching does there; at
reuse 1 it costs a sixth, and both residual-driven methods beat time marching on
all three benchmarks. Reuse is not a trade on a flux like `D0 q^3/u^2` or a
`chi(q)` with a square root switching on at a critical gradient: the stale
Jacobian points somewhere useless, and the extra inner iterations it buys push
the *total* assembly count up as well, so it loses on both axes at once. Jardin
under `Newton` costs 5760 flux and 760 derivative calls at reuse 10 against 440
and 240 at reuse 1. **If a steady solve is slow or will not converge,
`NewtonJacobianReuse = 1` is the first thing to try.** The default of 10 is
retained for the opposite case, a model whose Jacobian is finite-differenced at
`(1 + n_p)` residuals per assembly, which genuinely would rather have the
iterations.

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
