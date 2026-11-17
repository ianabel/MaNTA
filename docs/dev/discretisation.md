# Superconvergence and stabilisation

The superconvergent scheme (`Superconvergent = true`) and the stabilisation `tau` (`tauScaling`).

## Superconvergence (`Superconvergent = true`)

MaNTA *is* an interpolatory HDG method: `residual` evaluates `SigmaFn`, `Sources`
and `AuxG` at the `k+1` nodes of the degree-`k` nodal basis and applies
`InterpolateOntoBasis`, i.e. `I_h F(u_h)` with `I_h` mapping into `W_h = P_k`.
`Matrices.cpp` cites arXiv:1811.09667 for the Jacobian form, which is exactly the
paper that method comes from. Its two sequels (`refs/SuperconvergentHDG-I/II.pdf`, indexed with their dois in
`refs/Refs.md`; the PDFs are gitignored, so fetch them if you need them)
exist because that scheme loses the `k+2` superconvergence of the postprocessed
solution, and paper I's fix is what the flag implements.

`Postprocessing.{hpp,cpp}` reconstructs `u* ∈ P_{k+1}` per cell from `(u_h, q_h)`
by the local Neumann problem of paper I eq. (7) — sign-flipped, because MaNTA
carries `q = +d_x u` and the paper `q = -grad u`. Eliminating the Lagrange
multiplier gives `gamma = B11 alpha_q + B12 beta_u`, so the per-cell operators are
built once in `initialiseMatrices` and reused. Two more come with them:
`V = [phi_j(x_m)]` samples a `P_k` field at the `k+2` star nodes, and
`A9 = [(chi_m, phi_i)]` projects a `P_{k+1}` interpolant onto the `P_k` test
space, replacing the mass matrix `InterpolateOntoBasis` applies.

With the flag on, the physics is evaluated on the star nodes with `u*` in place of
`u_h`, and every Jacobian block gains a chain factor
(`SystemSolver::accumulateStarBlocks`):

```
d/d(u coeffs)     = A9 diag(dX/du) B12
d/d(q coeffs)     = A9 [ diag(dX/dq) V + diag(dX/du) B11 ]
d/d(sigma, phi)   = A9 diag(dX/dZ) V
```

`diag(dX/du) B11` in the `q` column is the only new coupling: `u*` depends on the
cell's `q` as well as its `u`. Every block stays `(k+1)x(k+1)` and `u*` is
cell-local, so **the DOF layout, `MX`, `solveHDGJac`, `solveJacEq`, the restart
format and the pybind11 casters are all untouched.** `ComputePhysics` loops over
`states.size()`, so no physics case — C++, Python or JAX — needs changing to be
evaluated at `k+2` points instead of `k+1`.

The reconstruction is built for every run with `k >= 1` regardless of the flag, so
`u_star` is always in the netCDF output and the `.dat` files; the flag controls
only whether the *method* uses it. `Tests/README.md` has the measured orders and
the list of what is not covered.

## Stabilisation (`tauScaling`)

`tau` enters five places — the `D`, `E`, `G`, `H` blocks and the `tau g_D`
Dirichlet term in `RF_cellwise` — and **all of them are written by
`assembleTauBlocks`**, reached through `applyResidualTau` (the residual's copies)
and `applyJacobianTau` (the `MBlocks`/`CEBlocks`/`CG_cellwise` parts and
`H_jac_cellwise`). `initialiseMatrices` calls both with the constant, with the
arithmetic in its original order: a `Constant` run is **byte-identical** to the
tree before the split, checked by `cmp` over every regression config's `.nc` and
`.restart.nc`. Re-run that check after touching `assembleTauBlocks`.

Under `Diffusive`, `faceTau` evaluates `tau * (kappa/h + floor * max kappa/h)`
per face, one-sided. `TauKappa` says where `kappa` comes from:

* **`Nodal` (the default)** — `nodalKappaOverH` takes the cell's `d sigma_hat /
  d q` at the physics nodes and extrapolates each cell's interpolant of it to its
  two faces; where the line goes non-positive and every node is positive it
  extrapolates `log kappa` instead, and where a node is non-positive the face
  gets the floor alone. It evaluates the case nowhere the residual does not.
* **`Face`** — `faceStates` builds the state at each face from the trace (the
  Dirichlet datum at a Dirichlet end) and the cell's own one-sided `q`, and
  `faceKappaOverH` makes one batched `ComputePhysicsDerivatives` on the `2 nCells`
  faces.

The residual applies tau at *its* state, the Jacobian build at `yJac` — which is
why `H` is split into `H_cellwise` and `H_jac_cellwise`: it is the one tau block
both read directly, and `solveHDGJac` reads it at *solve* time, long after the
residuals of a Newton iteration have overwritten the residual's copy.

Under `tauUpdate = Residual` the Jacobian carries `d tau / dy` in the blocks
`applyJacobianTau` rewrites, by a forward difference per component of the state
kappa is read from, every point perturbed in one batched call. `faceTauJacobian`
differences the face state, so `1 + 3 nVars + nAux` face calls per build, and
reaches `u` through `lambda`. `nodalTauJacobian` differences the nodal state —
only `3 nVars + nAux` calls, since kappa itself comes off the build's own
derivatives — and chains it onto the cell's coefficients, so nodal tau has no
`lambda` column at all; under `Superconvergent` the `u` component goes through
`B12` *and* `B11`, `u*` reading `q`. What both leave out is the floor's
grid-maximum term, and `LocalTauTests.cpp` shows that is *all* they leave out —
the mismatch against a finite-differenced residual falls from 3e-7 to 1e-9 when
the floor is taken to 1e-9. `the_nodal_dtau_dy_is_chained_through_every_component`
covers `u`/`q`, aux and the log branch with and without `Superconvergent`;
dropping the `B11` path moves it from 1e-8 to 3e-3, dropping the aux path to 5e-2.

Under `ContinuationStep` the continuation loop sets `tau` (`freezeTauAt`), the
residual and the Jacobian leave it alone, and `steadyNorm` refreshes it before
measuring convergence; a transient run is refused in `initialize()`. Under
`Face` the refresh evaluates the faces at the current state. Under `Nodal` it
reads `jacobianKappaOverH`, which every Jacobian build fills from the derivatives
it has just taken — so a held nodal tau costs no evaluation at all, at the price
of lagging the state by however many Newton iterations ago that build was. Before
a run's first build there is none, and `freezeTauAt` evaluates once; the flag is
cleared in the unconditional part of `initialize()`, which is what keeps a reused
solver from starting on the previous run's kappa (the `RF_cellwise` trap again).
`JacobianBuild` is `ContinuationStep` plus a refresh in each Jacobian build
(`tauRefreshedPerJacobian`), so tau is fixed across the Newton iterations sharing
a Jacobian; KINSOL evaluated the residual before the rebuild with the old tau, so
that step is lagged and costs Newton iterations.

Measured cost on the wall layer, against a constant tau, `Nodal` / `Face`:
`Residual` 2.6-2.8x / 1.6-1.7x — nodal pays `N(k+1)` points per residual against
`2N`, though the same number of *calls*; `ContinuationStep` 1.01-1.13x /
1.06-1.15x; `JacobianBuild` 1.04-1.31x / 1.27-1.49x. Answers agree to 2% in `u`.
On Shestakov's degenerate flux `Face` under `Residual` converged to a wrong steady
state in three runs of twelve and `Nodal` in none, and at a zero Dirichlet value
the face evaluation is a 0/0 that `Nodal` never makes. `docs/formulation.rst`
has the tables. `ContinuationStep` is a fixed-point iteration on tau that
pseudo-time cannot damp (tau reads `q`, and under `Face` `lambda`, which are
algebraic); it stalled once, under the default Jacobian reuse, and not with
`NewtonJacobianReuse = 1`. The adjoint would carry neither, so
`applySolverConfig` refuses `Diffusive` with `solveAdjoint`.

**`printSources` reads the source cache through a basis of the residual's
order.** With `Superconvergent = true` the cache holds `k+2` values per cell
rather than `k+1`, so `SystemSolver::print` picks its basis and stride from the
flag; hardcoding `k+1` there reads across cell boundaries.
