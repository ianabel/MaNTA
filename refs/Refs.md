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


## Reaching a steady state without integrating to one

The theory behind `SteadyState.cpp`. That file attributes pseudo-transient
continuation to "Kelley & Keyes", which is the 1998 paper below — and the useful
thing to know is that **the 1998 result does not cover MaNTA**. Its global
convergence proof is for `u' = -V^{-1} F(u)`, an ODE or a method-of-lines
semidiscretisation of one. MaNTA is an index-1 DAE: `sigma`, `q`, `phi` and
`lambda` are algebraic, which is the whole reason IDA is the integrator. The 2003
paper is the extension of that result to index-1 DAEs, so it is the one that
actually applies, and it is the house reference for the mode.

Two further things in it are load-bearing here. Its eq. (1.3) is the SER schedule
`SteadyState.cpp:15` implements, `dt <- dt ||F_prev||/||F_now||`, which it
attributes to Mulder & van Leer (its ref. [23], J. Comput. Phys. 59 (1985)
232–246). And its §1 advocates **mesh sequencing** — solve on a coarse mesh,
interpolate to the next, run Ψtc on each level — as the primary strategy for a
highly resolved nonlinear problem, calling that combination "particularly
effective". `FEATURES.md`'s first entry proposes adaptive mesh refinement against
the PTC solve specifically, so this is prior art for that entry and not only for
the steady-state mode; its ref. [29] (Smooke & Mattheij, Appl. Numer. Math. 1
(1985) 463–487) is the mesh-sequencing citation to follow next.

| Reference | URL (doi or arxiv) | Short Description | File Name |
| --- | --- | --- | --- |
| SIAM J. Sci. Comput. 25 (2003) 553–569 | https://doi.org/10.1137/S106482750241044X | Coffey, Kelley & Keyes, pseudo-transient continuation for **index-1 DAEs** — global convergence where the 1998 result assumes an ODE, so this is the one covering MaNTA's formulation. Also the source of the SER rule the code uses and of the mesh-sequencing argument. Paywalled, but free as CRSC tech report TR02-18 from crsc.ncsu.edu | PseudoTransientDAE.pdf |
| SIAM J. Numer. Anal. 35 (1998) 508–523 | https://doi.org/10.1137/S0036142996304796 | Kelley & Keyes, the original convergence analysis of Ψtc, and what `SteadyState.cpp:10` names. Reference [16] of the above, which reviews its results in §1.1 and states its hypotheses — those are ODE ones, so read it alongside the 2003 paper rather than instead of it. Paywalled | PseudoTransientConvergence.pdf |

## Mesh adaptivity

For `FEATURES.md`'s first item. The section above has one half of the prior art —
Ψtc run on each level of a mesh-sequenced solve — and this is the other: an error
indicator that says *where* to refine.

**The reason to read Capasso first is that MaNTA already computes what its
indicator is made of.** The strategy is built on the elementwise difference
between the solution at order `p` and the post-processed solution at order
`p + 1`, with the latter taken as the reference `u_ref` in `||u_ref - u_h||` over
each element. That post-processed field is `u*`, and `Postprocessor::computeUStar`
already produces it on *every* run with `k >= 1` — `SystemSolver.cpp:219` builds
the postprocessor regardless of the `Superconvergent` flag, and `NetCDFIO.cpp:174`
writes `u_star` into the output. So the expensive ingredient of the indicator is
present and tested; what is missing is the mesh machinery that would act on it.

Two caveats. It is a 2D unstructured code and MaNTA is 1D on an interval, so the
indicator transfers and the remeshing does not — refining a 1-D `Grid` is a much
smaller problem than theirs, and none of their geometry handling applies.

The second is sharper, and is the interpolatory question the HDG section is
already about. The indicator's worth rests on `u*` being a genuinely better
reference than `u_h`, which is the `k+2` superconvergence the paper cites. MaNTA
builds `u*` by the same local Neumann problem they do (`Postprocessing.hpp`, paper
I eqs. (6)–(7)), but **paper I exists because the plain interpolatory HDG scheme
loses that superconvergence**, and recovering it is exactly what
`Superconvergent = true` switches on. So with the flag off, `u*` is still computed
and still written out, and `||u* - u_h||` is still a reasonable smoothness
sensor — but it is not the `p`-against-`p+1` gap the paper's error analysis
assumes. Anything calibrated rather than used as a relative ranking should be
built on a superconvergent run.

| Reference | URL (doi or arxiv) | Short Description | File Name |
| --- | --- | --- | --- |
| Int. J. Numer. Methods Eng. 126 (2025) e70107 | https://doi.org/10.1002/nme.70107 | Capasso, Kudashev, Schwander & Serre, h-adaptivity for HDG applied to **fluid transport in a tokamak** — MaNTA's problem class, in SolEdge-HDG's 2D fluid-drift Braginskii setting. The indicator is the order-`p` against post-processed order-`p+1` difference, i.e. `u*`, which MaNTA already builds; it drives coarsening as well as refinement, and their §1 surveys the alternatives (residual-based indicators, a jump sensor on the trace). Open access, CC-BY | HDG-hAdaptivity.pdf |

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
