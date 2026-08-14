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
