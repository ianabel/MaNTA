The papers behind MaNTA's numerics, and the prior art on stiff transport solvers
that motivates them. The PDFs are gitignored (publisher material, several MB
each); these tables are the tracked part, so fetch each one from its doi and save
it under the file name given.

## The HDG discretisation

| Reference | URL (doi or arxiv) | Short Description | File Name |
| --- | --- | --- | --- |
| Advances in Mathematical Physics 2017, article 9736818 | https://doi.org/10.1155/2017/9736818 | Error estimates for HDG on parabolic equations with a nonlinear coefficient — MaNTA's problem class. Open access | HDG-Transport.pdf |
| Journal of Scientific Computing (2019) 81:2188–2212 | https://doi.org/10.1007/s10915-019-01081-3 | Superconvergence algorithm for interpolatory HDG methods | SuperconvergentHDG-I.pdf |
| Communications on Applied Mathematics and Computation (2022) 4:477–499 | https://doi.org/10.1007/s42967-021-00128-3 | Superconvergence algorithm without postprocessing | SuperconvergentHDG-II.pdf |

## Boundary conditions in HDG

Gathered while planning mixed/Robin boundary conditions (`FEATURES.md`). The
useful finding is a negative one worth recording so nobody repeats the search:
**none of these treats a general mixed condition `a u + b q + d sigma = c` for a
diffusion problem with `q` carried as an unknown**, which is MaNTA's formulation.
The Robin literature is almost entirely Helmholtz, where the condition is an
impedance/absorbing one and the coefficient is `i kappa` rather than free.

What they do settle is the *structure*, and it is the one MaNTA already has. A
boundary condition is imposed on the **numerical flux** `q_hat . n = q . n + tau
(u - lambda)`, as a linear relation between that and the **trace unknown**
`lambda` — not between it and the interior `u`. So the `a` coefficient belongs on
the `H` diagonal (the lambda column), which is where `SystemSolver.cpp` keeps
`-tau`. Cui & Zhang is the clearest statement: their eq. (2.3) defines the flux and
the impedance condition relates it to `u_hat` with the datum on the right.

| Reference | URL (doi or arxiv) | Short Description | File Name |
| --- | --- | --- | --- |
| IMA Journal of Numerical Analysis (2014) 34:279–295 | https://doi.org/10.1093/imanum/drt005 | Cui & Zhang, HDG for Helmholtz with a first-order absorbing (Robin) boundary condition. **The closest thing to a house reference for the mixed row**: the condition is imposed on the numerical flux against the trace unknown, and `tau` stays in the row. An author copy is free from polyu.edu.hk | HDG-Helmholtz-Robin.pdf |
| Journal of Computational Physics 228 (2009) 3232–3254 | https://doi.org/10.1016/j.jcp.2009.01.030 | Nguyen, Peraire & Cockburn on HDG for linear convection–diffusion. The canonical statement of imposing boundary conditions through the numerical flux, for exactly MaNTA's `(u, q, sigma_hat)` triple. Paywalled | HDG-ConvDiff-BCs.pdf |
| SIAM Journal on Numerical Analysis (2009) 47:1319–1365 | https://doi.org/10.1137/070706616 | Cockburn, Gopalakrishnan & Lazarov, the unified hybridization framework — where the trace equation and `tau` come from. Paywalled; a free copy is on PDXScholar | HDG-UnifiedHybridization.pdf |
| arXiv:1811.00737 | https://arxiv.org/abs/1811.00737 | Oikawa, a flux-based HDG method: hybridizes the *flux* trace rather than the solution trace, so the local problem carries the Neumann condition, and studies the `tau -> infinity` limit. Structurally the dual of MaNTA's `d sigma` term. Dirichlet only | HDG-FluxBased.pdf |
| arXiv:2212.11529 | https://arxiv.org/abs/2212.11529 | Modave & Chaumont-Frelet, HDG with characteristic variables for Helmholtz — hybridizes *Robin traces*. Its Remark 2.10 is worth reading before trusting a pure flux condition: local problems with Robin data are always well posed where Dirichlet ones need not be | HDG-CharacteristicVariables.pdf |
| arXiv:2503.19684 | https://arxiv.org/abs/2503.19684 | Ellmenreich, Lederer, Giacomini & Huerta, characteristic (non-reflecting) boundary conditions for HDG, in a framework where the common HDG conditions are special cases. Compressible Euler rather than diffusion, so the machinery transfers and the conditions do not | HDG-CharacteristicBCs.pdf |

## Mesh adaptivity

`FEATURES.md`'s first item, and no longer speculative: **two of these four are
the sources of code that ships.** `DegreeAdaptation` picks the global polynomial
degree by solving, estimating the error and re-solving, and it is assembled out
of this section — Capasso *et al.* eq. (15) is the per-cell estimate, which
`Postprocessing.hpp` cites by name, and Giorgiani's rule is the increment that
estimate feeds. `docs/running.rst` has the keys and a worked four-solve run.

The estimator came for free. The gap between `u_h` and its own postprocessing
`u*` is what `Postprocessing.cpp` already built for every run with `k >= 1`,
whether or not `Superconvergent` was set; what the adaptation added was the loop
around it, not the measurement.

The finding worth carrying forward is a **negative** one, and it is why `p` went
first and `h` has not gone at all: raising `k` beat adaptive `h` by seven orders
on every benchmark in the tree, and `h`-adaptivity did not pay on any of them
once the cost of the extra solves was counted. Read the `h` and `hp` papers below
as the case for revisiting that, not as a plan of record. Still open, in
`FEATURES.md`'s order: `h`-adaptivity, which needs a mesh-to-mesh transfer and is
*not* the problem degree transfer already solved — the projections stop
composing; per-cell degrees; and adapting inside the real time march, still
blocked on carrying a BDF history across a level change.

| Reference | URL (doi or arxiv) | Short Description | File Name |
| --- | --- | --- | --- |
| International Journal for Numerical Methods in Engineering (2025) 126:e70107 | https://doi.org/10.1002/nme.70107 | Capasso, Kudashev, Schwander & Serre, `h`-adaptivity in the SolEdge-HDG code. **The source of MaNTA's error estimate**: eq. (15), the squared `L2` norm of `u* - u_h` over a cell divided by that cell's measure, is what `Postprocessing.hpp` implements and what `DegreeAdaptation` decides on. The paper spends it on refining and coarsening a 2-D drift-Braginskii mesh, with a cheaper oscillation indicator to find the under-resolved regions first; MaNTA took the indicator and none of the mesh machinery. Also the nearest of these to MaNTA's problem class — an HDG solver for tokamak fluid transport. Open access (CC BY) | HDG-hAdaptivity.pdf |
| Computers & Fluids 98 (2014) 196–208 | https://doi.org/10.1016/j.compfluid.2014.01.011 | Giorgiani, Fernández-Méndez & Huerta, degree adaptivity for HDG on incompressible Navier–Stokes. **The source of `MaxDegreeIncrement`'s rule**, `dk = ceil(log_b(E / eps))` with `b` = `DegreeAdaptationBase`. What earns it its place is that it assumes no convergence *order* — only that one more degree buys roughly a factor of `b` — which is what makes it safe against `u*`'s undependable observed rate. What MaNTA has *not* taken is what the paper actually adapts: the degree on **elements and faces separately**, i.e. per-cell degrees and a trace `lambda` free not to follow the cells. Paywalled | HDG-pAdaptivity.pdf |
| Computers & Fluids 98 (2014) 3–16 | https://doi.org/10.1016/j.compfluid.2014.03.023 | Woopen, Balan, May & Schütz, target-based (adjoint-driven) `hp`-adaptation, and a like-for-like efficiency comparison of hybridized against standard DG. Nothing here implements this: it is the way in to refining on an *objective* rather than on a solution indicator, and it would reuse the adjoint MaNTA already solves for `G = ∫ g dx` rather than needing new machinery. Compressible flow. Paywalled | HDG-hpAdaptivity.pdf |
| Computers and Fluids 301 (2025) 106792 | https://doi.org/10.1016/j.compfluid.2025.106792 | Levý & May, time-accurate anisotropic adaptation for HDG — and the reason it is here is the second half, a **solution transfer between meshes that preserves local minima and maxima**. Two of the three open items above need exactly that: `h`-adaptivity, where the mesh-to-mesh projections stop composing, and adapting inside the time march, where IDA's BDF history has to survive the level change and an interpolation that overshoots corrupts the history rather than merely the current step. DIRK there against BDF here, so it is the transfer operator that carries over, not the scheme. Paywalled | HDG-UnsteadyAdaptivity.pdf |

## Stiff transport solvers

The problem MaNTA exists to solve: a 1-D transport equation whose diffusivity
depends strongly and non-analytically on the gradient, which defeats a plain
implicit time step. These are the established ways of coping.

| Reference | URL (doi or arxiv) | Short Description | File Name |
| --- | --- | --- | --- |
| Journal of Computational Physics 185 (2003) 399–426 | https://doi.org/10.1016/S0021-9991(02)00063-3 | Shestakov/LoDestro iteration: couples a turbulence code to an implicit transport equation, stable at arbitrarily large time steps | LoDestroMethod.pdf |
| Journal of Computational Physics 186 (2003) 360 | https://doi.org/10.1016/S0021-9991(03)00047-0 | Corrigendum to the above — a short-wavelength instability of the fully implicit scheme, and two corrected equations | ShestakovCorrection.pdf |
| Journal of Computational Physics 227 (2008) 8769–8775 | https://doi.org/10.1016/j.jcp.2008.06.032 | Jardin et al. on gradient-dependent diffusivity: why Crank–Nicolson and backward Euler oscillate, and the correction that cures it. Despite the file name, this is not about PTRANSP | PTRANSP.pdf |
| Physics of Plasmas 17, 056109 (2010) | https://doi.org/10.1063/1.3323082 | Trinity: multiscale coupling of a 1-D transport solve to gyrokinetic flux calculations, with the fluxes evaluated by a separate expensive code | TrinityAlgorithm.pdf |
| Computer Physics Communications 214 (2017) 1–5 | https://doi.org/10.1016/j.cpc.2016.12.018 | FASTRAN: 4th-order Interpolated Differential Operator scheme plus a root-finding nonlinear iteration; solves for the gradient as an independent unknown, as MaNTA's `q` is | ParkEfficientSolver.pdf |
| Physics of Plasmas 16, 060704 (2009) | https://doi.org/10.1063/1.3167820 | TGYRO (Candy, Holland, Waltz, Fahey & Belli). The closest algorithmic analogue to `SteadyStateSolver = Newton` in the literature, and the best cost comparison MaNTA has. It solves the *steady* problem directly by flux matching, `Q_hat = Q_hat^T`, and its unknowns are the **logarithmic gradients** `z_a = -(1/T_a) dT_a/dr` rather than the profiles, which are recovered by integrating from a fixed pedestal-top value (their eq. (7)). Newton with a per-equation relaxation parameter `eta`, halved on any step that increases the residual, plus a cap `dz_max` on the correction -- where MaNTA puts the same job on the pseudo-time term. Its Jacobian is **block diagonal by assumption** (fluxes depend only on local gradients, which they note "affects only the rate of convergence, not the accuracy of the root") and **forward-differenced**, so it pays the `(1 + n_p)` multiplier `PERFORMANCE.md` measures MaNTA against. **Its cost is quoted in exactly our currency**: 4 calls per radius per iteration (3 for the Jacobian, 1 for the relaxation correction), 8 radii, ~12 iterations = 384 calls to GYRO/NEO, i.e. **48 visits per point**. Compare `tab:steady`, where MaNTA is 7-45 at `NewtonJacobianReuse = 1`. In wall-clock that is **58 h on 1536 cores** of a Cray XT4 for one DIII-D L-mode discharge, at 6.6 GYRO simulations per hour. Note also the two tuned parameters the scheme carries -- the finite-difference step `Dz = 0.3/a` and the correction cap `dz_max = 0.4/a` -- and that their good values differ between gyrokinetic and reduced-model fluxes, because the large `Dz` is there to see past the statistical variance in a time-averaged turbulent flux | TGYRO.pdf |
| Plasma Physics and Controlled Fusion 68 (2026) 065024 | https://doi.org/10.1088/1361-6587/ae7640 | ASTRA-8 (Tardini *et al.*), the 2026 rewrite of the 40-year-old ASTRA framework. Open access. Three things in it bear on MaNTA. **Sec 3.1**: one solver routine for every transport channel, up to 15 equations, the user choosing which to evolve and which to prescribe -- the same separation MaNTA makes with `SystemSpec`, arrived at independently. **Sec 3.3**: a boundary condition may be set *on the profile or on the corresponding flux*, at any radius, independently per equation. That is MaNTA's Dirichlet-versus-mixed-`delta` distinction, and it is independent corroboration of the Jardin finding in the paper's Sec 6.3 -- a production code has had the flux form for decades. **Sec 5.3**: the expensive models (TGLF, QuaLiKiZ, NEO) are parallelised over *radial points* -- shared memory segments and semaphores rather than MPI -- scaling as `1/n_cores` down to one point per core, where 64 radial points on 64 cores is about 5 s per TGLF time step. The cost model is the one `PERFORMANCE.md` counts in: total work is flux evaluations, and the parallelism is over the grid rather than inside the solve. **Sec 5.4**: see the section below; it is the reason this reference matters most | ASTRA-8.pdf |
| IPP Report 5/98, February 2002 | https://pure.mpg.de/pubman/item/item_2138238 | Pereverzev & Yushmanov, the 147-page ASTRA manual, and reference [18] of ASTRA-8 -- which cites its **eq. (59)** for the generic transport equation every channel is reduced to. Also worth knowing: ASTRA splits each source into a part *linear in the unknown*, treated implicitly, and a remainder treated explicitly (`SN` against `SNN`), which is the device MaNTA replaces with a full Newton through `dSources_du`. **Table 4.15 is where the comparison bites**: ASTRA controls the nonlinearity with the *time step*, not with a Newton iteration. `ITEREX`, the number of outer iterations per step, defaults to **1**, and accuracy comes from `TAU` being cut whenever the maximum relative change in any evolved profile exceeds `DELVAR` and grown by `TAUINC = 1.1` otherwise -- a rule Sec 4.9.8 lets a case replace outright. That is the cost MaNTA's Newton and pseudo-transient modes are measured against: a stiff model forces ASTRA to a time step far below `tau_E`, which is exactly what ASTRA-8 Sec 5.4's stabiliser exists to avoid paying. A scanned original, OCR'd, so the text layer is imperfect and the equations are images -- read the figures, do not trust a copied formula | Astra_ocr.pdf |
| SIAM Journal on Scientific Computing 25 (2003) 553–569 | https://doi.org/10.1137/S106482750241044X | Coffey, Kelley & Keyes, **the house reference for `SteadyStateSolver = PseudoTransient`**, and the one that actually covers MaNTA: it extends the global convergence result for pseudo-transient continuation from the semidiscretised ODE case to semi-explicit index-1 DAEs. That distinction is the whole point of citing this rather than the better-known ODE paper — MaNTA is an index-1 DAE, with the `sigma`, `q`, `lambda` and `phi` rows algebraic, so the ODE-only theory says nothing about the system being solved. The SER step-size rule `docs/running.rst` describes is the one analysed here (it is due to Mulder & van Leer; this paper is where it meets the DAE case) | PseudoTransientDAE.pdf |


## Sources that carry a time derivative

The motivation for `FieldSpec::sourceReadsTimeDerivatives` and `State::udot`
(`docs/superpowers/specs/2026-09-21-time-derivative-sources-design.md`). One
paper asks for it, and it is the derivation MaNTA's own equations come from. A
second looks as though it does and, read properly, asks for something else --
which is worth keeping here, because the distinction is the interface boundary
this feature sits on.

**Abel *et al.*, Rep. Prog. Phys. 76 (2013) 116201**, eq. (201) -- the
potential-exchange heating `P^pot` on the right-hand side of the heat transport
equation (194) -- carries `dn_s/dt` outright, plus `d/dt` of a metric coefficient
and of the flux-surface label. That is the derivation MaNTA's equations come
from, so this is not an extension so much as a gap being closed. Not in `refs/`;
it is `~/Papers/RoPP-final.pdf`.

**ASTRA's stiffness stabiliser is the nearest thing in production use, and the
primary source says what it actually is.** `ASTRA-8.pdf` Sec 5.4 describes the
oscillation a stiff turbulent model produces -- zero gradient gives no transport,
so the gradient grows, so the transport overshoots and flattens it again -- and
calls the cure *"an additional artificial diffusion, proportional to the time
derivative of the relevant kinetic profile"*, one that *"has also been crucial
for almost every predictive transport modelling study which was carried out with
the ASTRA code in the last two decades"*. `PereverzevCorrigan.pdf` is that cure,
and it is a **discretisation device rather than a source term**: a large
diffusive flux `Dbar u_x` is added *implicitly* and an exactly compensating
convective flux `Vbar u`, with `Vbar = Dbar u_x / u`, is subtracted *explicitly*.
The two cancel identically in the differential equation. Only the difference
scheme sees anything, and what it sees is their eq. (16),

```
q~ = -tau * Dbar * u_xt + O(h^2) + O(tau^2)
```

a mixed space-time derivative carrying the time step in front of it.

**So it is not a `du/dt` source, and phase 1 does not reach it.** The term
differentiates the *gradient*: in MaNTA's variables that is `dq/dt`, and it
belongs in `SigmaFn` rather than in `Sources`, while `State::udot` is
deliberately the only time derivative a case is given. Note what that decision
does and does not rest on. The *value* is there -- IDA supplies `y'` for the
algebraic components too, and `residual()` is handed all of it -- so a flux
carrying `-tau Dbar q_dot` is not blocked by anything being unavailable. It is
blocked by `IDASetId`, which declares this system's only differential rows to be
`u`: a residual that reads `q_dot` puts `alpha`-weighted entries in the `q`
columns of `dF/dy'` and makes that declaration false. Anyone porting the
stabiliser starts there, not at the interface. What the device *is*, in
MaNTA's terms, is pseudo-transient continuation -- a term proportional to the
step that vanishes at the fixed point and exists to make the Newton step
survivable. So this narrows the second motivation for phase 1 rather than
removing it. `RoPP` eq. (201) remains the one that asks for `udot` inside a
source, and it asks alone.

**It is a steady-state device, and ASTRA says so.** The term speeds convergence
*"for steady-state simulations, i.e. in absence of a significant
time-dependence"*, and where the conditions are strongly time dependent --
heating modulation, a pellet -- it *"has to be set to zero, or at least strongly
reduced, and its impact has to be assessed"*. So it buys convergence on a steady
problem at the price of a modified transient, and a case carrying one is making
that trade whether or not it says so.

Note what that means for MaNTA's own machinery. ASTRA writes the damping into
the *difference scheme*, where pseudo-transient continuation writes it into the
*solver* and drives it to zero at the fixed point. The two are alternatives, not
complements: a case that reimplemented the stabiliser as a physics term on top of
a `PseudoTransient` run would be damping twice. A physics term of that shape
would also modify the effective mass matrix `X - dS/d(udot)` that
`checkEffectiveMassMatrix` guards, which is exactly the operator such a
stabiliser is designed to change.

| Reference | URL (doi or arxiv) | Short Description | File Name |
| --- | --- | --- | --- |
| Computer Physics Communications 179 (2008) 579–585 | https://doi.org/10.1016/j.cpc.2008.05.006 | Pereverzev & Corrigan, *Stable numeric scheme for diffusion equation with a stiff transport* -- the scheme ASTRA-8 Sec 5.4 cites as its [34], and the source for everything above. **What the coefficient should be**: `Dbar > q_eta = dq/d(eta)` pointwise -- the *slope* of the flux-gradient curve, not the diffusivity `D_eff = q/eta`. In a stiff model that distinction is the whole point: in their ITER inductive / GLF23 example `q_eta` runs 20-100 m^2/s while `D_eff` never exceeds 1, and it takes `Dbar >= 50 m^2/s`, fifty times the diffusivity. Below that the instability appears locally wherever `Dbar < q_eta` and then spreads over the grid. They prescribe `Dbar` constant in space and call `Dbar > q_eta` "rather a rough estimate", since the flux depends on more than the gradient. **The error is monitored, not bounded**: the leftover difference source `Sbar_i = (qbar_{i+1/2} - qbar_{i-1/2})/h = O(tau)` is computed each step and compared against the physical sources -- it is a short-scale dipole, so its integral over any few cells is essentially zero and the energy balance is untouched, but where it grows comparable to `S` locally, `Dbar` or `tau` has to come down, or the term is subtracted and iterated away within the step. Their instability sensor is `max_i |Dhat_an - D_an| / D_an <= eps_tol`, with `eps_tol` at 5-10% and no sense in going below 5%. **What it buys**: several orders of magnitude on the time step at steady state -- a factor 1e3 does not yet bring accuracy into play -- and less in a fast transient, where other limits bind first. Transport barriers are the known weak spot, their two variants moving the barrier in opposite directions. **Note also their objection to the alternative**, Kinsey, Staebler & Waltz (Phys. Plasmas 9 (2002) 1678), who take `D_an = dq/d(eta)` exactly: that quantity is discontinuous in space, incomplete, and "requires derivation of the numerically defined flux `q` that is usually the most expensive part of the simulation". MaNTA's answer to the last is that the HDG Jacobian derives it anyway, and never assembles it | PereverzevCorrigan.pdf |

## Coupling to a magnetic field solver

For `FEATURES.md`'s third item. A self-consistent field is, algorithmically, a
large set of algebraic constraints — which IDA and KINSOL already handle — so what
these are for is the *Jacobian* question: the coupled system has the block form

```
( HDG Jacobian | A1              )
( A2           | B^{GS} Jacobian )
```

with `N_magnetics >> N_HDG`, and MaNTA's static condensation only solves the top
left. The two ends of the design space are represented here: what a free-boundary
transport code has traditionally done, and what a modern differentiable
Grad–Shafranov solver can now provide.

| Reference | URL (doi or arxiv) | Short Description | File Name |
| --- | --- | --- | --- |
| ENEA report RT/TIB/88/5 (1988) | (no doi; scanned report) | Cenacchi & Taroni, **JETTO** in its original free-boundary form. The prior art for exactly this coupling, and worth reading for what it does *not* attempt: transport and equilibrium are advanced separately rather than solved as one system, which is the cheap end of the design space and the fallback if the coupled Jacobian proves too expensive | JETTO.pdf |
| SIAM J. Sci. Comput. (2025) S364–S385 | https://doi.org/10.1137/24M1674108 | Serino, Tang, Tang, Kolev & Lipnikov, an adaptive Newton-based free-boundary Grad–Shafranov solver. **The paper closest to what the roadmap entry assumes exists**: Newton on the full nonlinear free-boundary problem, with the free-boundary contribution to the Jacobian obtained by shape calculus, and the linear system solved by block factorization with AMG on the elliptic subblocks. That block factorization is the same structural question MaNTA would face | NewtonGSMFEM.pdf |
| arXiv:2406.06718 | https://arxiv.org/abs/2406.06718 | Citrin et al., **TORAX** — a differentiable tokamak transport simulator in JAX that solves ion and electron heat, particle transport *and current diffusion* as one coupled system. The closest existing thing to MaNTA's ambitions taken one step further, and the demonstration that automatic differentiation through the whole solve is practical rather than aspirational. Relevant to `manta.jax` and the adjoints as much as to the field coupling | TORAX.pdf |
