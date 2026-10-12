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

## The noisy benchmark in MaNTA

    python noise_benchmark.py         # 136 steady solves, ~1 min on 6 workers

16 cells, k = 4 (discretisation error 1.7e-3 in p_e), eight seeds. The
Jacobians are "control" (exact derivatives of the noiseless fluxes), "full"
(forward differences in all six box inputs) and "t3d" (T3D's scheme: three
kappa perturbations of max(f |kappa|, f), the box's explicit u-dependence and
the prefactor's density derivative dropped, as T3D's psi matrices drop them).
Errors are against the noiseless solution at the same discretisation.

* **The residual floors at ||F|| = 2.46** (2.36-2.58). Reaching it costs ~1000
  calls and leaves errors of 2.5% in n, 2.9% in p_H and 0.8% in p_e: what one
  noise realisation allows.
* **A tolerance just above it, 3, gets nearly all of that** in 33 calls
  (control), 84 (t3d, f = 0.5) or 144 (full): 2.4-2.7% in n, 1.4-2.0% in p_e,
  eight seeds of eight. Looser tolerances leave p_e unconverged -- 5% at 5, 11%
  at 10-20 -- while n and p_H barely move, being noise-limited already.
* **T3D's step, f = 0.5, is the right one for this noise; f = 0.2 is not.** At
  0.2 only 2-4 seeds of 8 converge, the definiteness guard fires ~1900 times a
  solve, and p_e is out by up to 38%.
* **T3D's reduced Jacobian beats "full"**: the same Newton iterations (18 at
  tolerance 3) for three box calls a build instead of six.
* **The density is the noisiest channel, and that is T3D's noise model.**
  `Linear` is not vectorised, so its noise is one draw per call for the whole
  profile: fully correlated across radius, with nothing for the solve to
  average. The heat-flux noise is drawn per point.
