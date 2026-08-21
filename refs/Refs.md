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
| SIAM Journal on Scientific Computing 25 (2003) 553–569 | https://doi.org/10.1137/S106482750241044X | Coffey, Kelley & Keyes, **the house reference for `SteadyStateSolver = PseudoTransient`**, and the one that actually covers MaNTA: it extends the global convergence result for pseudo-transient continuation from the semidiscretised ODE case to semi-explicit index-1 DAEs. That distinction is the whole point of citing this rather than the better-known ODE paper — MaNTA is an index-1 DAE, with the `sigma`, `q`, `lambda` and `phi` rows algebraic, so the ODE-only theory says nothing about the system being solved. The SER step-size rule `docs/running.rst` describes is the one analysed here (it is due to Mulder & van Leer; this paper is where it meets the DAE case) | PseudoTransientDAE.pdf |


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
