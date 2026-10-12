# T3D's ReLU noise benchmark, through MaNTA

[T3D](https://bitbucket.org/gyrokinetics/t3d) (Trinity3D) is a transport solver
that uses GX as its flux model. Its regression test `test-w7x-relu-ke.in`
replaces GX with an analytic `ReLU` model with stochastic noise, which makes it
a benchmark both codes can run. This directory poses it for MaNTA, checks the
port against T3D, and measures what the noise costs each.

| File | What it is |
|---|---|
| `relu_ke.py` | the case: equations, geometry, sources, the ReLU box, three Jacobians |
| `t3d_relu_ke.npz` | T3D's own coefficients for the benchmark, written by `extract_t3d.py` |
| `extract_t3d.py` | runs T3D's initialisation and dumps them; needs T3D |
| `run_t3d.py` | runs T3D, optionally with the two corrections below; needs T3D |
| `noise_benchmark.py` | the noisy benchmark over several seeds, in MaNTA |

Only `extract_t3d.py` and `run_t3d.py` need T3D. Everything else runs from the
`.npz`.

## The problem

Three channels, n_e, p_H and p_e, in W7-X VMEC geometry over rho in [0, 0.8]:
zero flux on the axis, fixed values at rho = 0.8, 5 MW of electron heating in a
Gaussian at rho = 0.1 of width 0.05, collisional exchange, and no particle
source. The fluxes are gyro-Bohm prefactors times T3D's ReLU model in the
gradient scale lengths kappa = -d ln / d rho, with +-20% uniform noise drawn
afresh on every call. `relu_ke.py` has the equations in full; every coefficient
comes from T3D's own code.

## Three things about the benchmark itself

**As written, its steady state is not unique near the axis.** T3D's `ReLU` is
mirror-symmetric in kappa_T: zero only for |kappa_T| < a, and m (kappa_T + a)^2
below -a. That branch is anti-diffusive, and it supplies a second kappa_T for
every heat flux. Where the enclosed heating is small -- inside the heating peak
-- both branches carry it. Noiseless and run to steady state, T3D's N = 17 and
N = 65 solutions sit on the negative branch near the axis and N = 9 and 33 on
the positive one; MaNTA's land on either depending on resolution; and the
central T_e wanders by up to 25% under refinement in both codes. The port's
default is the one-sided ReLU (no turbulent flux for kappa_T < a), and
`ReluFluxes(one_sided=False)` gives T3D's.

**T3D's geometry is wrong inside VMEC's first surface.** VMEC stores
<|grad rho|> = 0 on the axis surface; T3D interpolates linearly from it, so
<|grad rho|>, and with it the volume element, is wrong everywhere inside
rho = 0.071 at every resolution, by 34% at N = 9's innermost point. The heating
is normalised to 5 MW over that volume, so power moves inward and the outer
profiles cool. `run_t3d.py --fix-axis-grho` re-interpolates without the axis
surface.

**The benchmark's iteration settings never iterate.** With
`newton_threshold = newton_tolerance = 5e-2`, every T3D time step accepts its
first linearised iterate, so the final state is not a converged steady state:
against MaNTA it is off by 0.85% in n at every resolution and by up to 15% in
T_e at N = 65. With thresholds of 1e-5, T3D's Newton converges -- linearly, at
about 12% per iteration, the signature of its approximate Jacobian.

## The port, checked against T3D

Noiseless, one-sided, axis corrected and with converged iterations, T3D
converges to MaNTA's solution (64 cells, k = 4) as N grows:

| T3D N | iterates | n | T_H | T_e |
|---|---|---|---|---|
| 9 | 211 | 1.4e-5 | 2.0e-3 | 1.8e-2 |
| 17 | 276 | 4.1e-4 | 4.7e-4 | 6.1e-3 |
| 33 | 338 | 9.7e-5 | 1.5e-4 | 1.8e-3 |
| 65 | 268 | 6.7e-5 | 5.7e-5 | 5.7e-4 |

(maximum relative differences at T3D's grid points). MaNTA's own density
matches an independent solution of the zero-flux ODE to 5e-12. Each T3D iterate
is four box calls -- the base and one perturbation per channel -- so T3D's
1.8e-3 in T_e cost 1352 calls; MaNTA at 16 cells, k = 4 is at 1.7e-3 for 104.

## The noisy benchmark

    python noise_benchmark.py         # MaNTA: 168 steady solves, a few minutes on 6 workers

16 cells, k = 4 (discretisation error 1.7e-3 in p_e), eight seeds. The
Jacobians are "control" (exact derivatives of the noiseless fluxes), "full"
(forward differences in all six box inputs) and "t3d" (T3D's scheme: three
kappa perturbations of max(f |kappa|, f), the box's explicit u-dependence and
the prefactor's density derivative dropped, as T3D's psi matrices drop them).

**What limits the accuracy here is the stopping test, not the noise.** MaNTA's
steady solve stops on one mesh-weighted L2 norm of the whole residual. The
density's rows hardly contribute to it -- at steady state Gamma_e is zero -- so
the density converges invisibly: noiseless, a solve stopped at ||F|| < 3 is out
by 2.6% in n and 2.2% in p_H, still 1.9% in n at ||F|| = 0.03, and below 1e-3
only at ||F|| = 0.01. With noise, ||F|| floors at 2.46 (2.36-2.58 over eight
seeds), set by the heat-flux rows, 250 times above where n converges. Every
noisy solve therefore ends with n where the stopping test left it:

* **A tolerance just above the floor**, 3, stops in 33 calls (control), 84
  (t3d, f = 0.5) or 144 (full), eight seeds of eight, at 2.4-2.7% in n,
  2.0-2.5% in p_H and 1.4-2.0% in p_e -- the noiseless solve's errors at the
  same tolerance. Looser tolerances leave p_e unconverged too: 5% at 5, 11% at
  10-20.
* **A tolerance below the floor** is never met: near it every continuation
  step's inner Newton solve fails, each rejection halves the pseudo-time step,
  which falls to ~1e-59, and after ~1000 calls the last accepted state is
  returned, with n still out by 2.5%.
* `SteadyStateSolver = "Newton"` behaves the same, and so does `NewtonScaling
  = "ErrorWeights"`, which reweights KINSOL's inner solve but not the loop's
  test.

**`SteadyStateTest = "NewtonStep"` removes that limit**: it stops on IDA's
corrector test, `ss * ||J0^-1 F|| <= 0.33` in the error weights
`1/(rtol |u| + atol)`, so each channel is measured on its own scale and the
density counts. Noiseless it reaches the answer the residual test cannot see:

| Jacobian | rtol = atol | box calls | error n | p_H | p_e |
|---|---|---|---|---|---|
| control | 1e-2 | 11 | 2.2e-3 | 4.5e-2 | 8.1e-3 |
| control | 3e-3 | 29 | 1.5e-3 | 1.5e-3 | 1.9e-4 |
| control | 1e-3 | 39 | 1.8e-11 | 1.9e-11 | 2.5e-12 |
| t3d, f = 0.5 | 1e-2 | 37 | 7.2e-3 | 4.9e-3 | 3.3e-2 |
| t3d, f = 0.5 | 3e-3 | 73 | 1.7e-2 | 1.2e-2 | 5.6e-3 |
| t3d, f = 0.5 | 1e-3 | 309 | 1.4e-3 | 1.7e-3 | 2.5e-4 |

(errors against the converged solution at this discretisation, which the
`Residual` test at `||F|| < 1e-10` reaches in 104 calls). The test's norm is an
RMS over every row, IDA's, so the worst point can sit a few tolerance units
out -- up to 2.4 here. Measured in its own norm the test is conservative with
exact derivatives (true error 0.14 where it reports 0.19 at 1e-2, 0.08 against
0.22 at 3e-3) and over-optimistic by up to 6x with T3D's Jacobian, whose
corrections are not `J^-1 F` (0.81 against 0.14 at 3e-3).

With noise, `SuppressAlgebraicError = true` is required, so that only u is
tested: sigma is an algebraic unknown equal to the flux, its correction carries
the +-20% noise at full strength, and no step damps it. With it:

| Jacobian | rtol = atol | converged | box calls, median (max) | noise error n | p_H | p_e |
|---|---|---|---|---|---|---|
| control | 1e-2 | 8/8 | 11 (20) | 4.9e-3 (8.5e-3) | 3.8e-2 (9.1e-2) | 1.3e-2 (2.2e-2) |
| control | 3e-3 | 0/8 | 1530 (1542) | 2.6e-2 (3.4e-2) | 2.6e-2 (3.1e-2) | 6.4e-3 (1.8e-2) |
| t3d, f = 0.5 | 1e-2 | 8/8 | 65 (148) | 1.6e-2 (1.9e-2) | 1.2e-2 (1.4e-2) | 1.9e-2 (2.8e-2) |
| t3d, f = 0.5 | 3e-3 | 7/8 | 83 (6205) | 1.7e-2 (2.4e-2) | 1.3e-2 (2.5e-2) | 7.1e-3 (1.1e-2) |

So at 1e-2 the noisy solve converges every seed for a third of T3D's 68 calls
(exact derivatives) or about as many (T3D's own Jacobian), at errors of the
order of T3D's below. At 3e-3 with exact derivatives the test asks for a
correction below what the noise itself produces, and every seed runs out of
steps -- the noise floor the design's phase 4 (a `NoiseLimited` outcome and
iterate averaging) is for. T3D's Jacobian "converges" there only because its
corrections understate the distance, as the noiseless row at 3e-3 shows.

**The error the noise itself causes is much smaller.** Frozen -- the same draw on
every call -- a +-20% particle-flux error moves n by 3-4e-3, and a constant
+-20% heat-flux error moves p_H by 5-6% and p_e by 6-8%; live noise, drawn
independently at each point, averages below that.

Two more results hold whatever the stopping test:

* **T3D's step, f = 0.5, is the right one for this noise; f = 0.2 is not.** At
  0.2 only 2-4 seeds of 8 converge, the definiteness guard fires ~1900 times a
  solve, and p_e is out by up to 38%.
* **T3D's reduced Jacobian beats "full"**: the same Newton iterations (18 at
  tolerance 3) for three box calls a build instead of six.

**T3D itself**, at N = 17 with the benchmark's own 5% thresholds, one-sided and
axis-corrected, eight seeds run four at a time (one T3D process peaks at 3.6 GB
reading VMEC): every seed takes 17 iterates, 68 box calls, with no reset, and
ends 0.87% from the converged answer in n, 1.2% in p_H and 0.93% in p_e
(worst seed 0.91%, 1.7%, 2.3%). Against the same run without noise the noise
accounts for 6e-4 in n and 5e-3 in p_H and p_e; the rest is the 5% thresholds,
which leave the noiseless run 8.8e-3 from converged. T3D's test is the relative
change of each profile, which weighs the channels separately, and that is why
its density converges where MaNTA's does not.

T3D's particle-flux noise is one draw per call for the whole profile --
`Linear` is not vectorised the way `ReLU` is -- so it is fully correlated across
radius; the heat-flux noise is drawn per point. The port reproduces both.
