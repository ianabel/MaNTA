
Lists upcoming features that are in development or being thought about

- Explore adaptive mesh-refinement schemes; first to the PTC steady-state algorithm or possibly the direct newton solve as well; then work out if it's possible to add to the real TimeMarch (requires interpolating history to allow IDA's BDF methods to work, and requires the hooks for that to be in SUNDIALS)

  **Partly done, and not in the shape this entry expected.** `DegreeAdaptation`
  adapts the global *polynomial degree* between steady solves — see
  `docs/running.rst`. It went first because that is where the win looked to be,
  and it covers both steady modes at once, since `SteadyMode::Newton` is the same
  code with `ptcStep = infinity`.

  **The reason it went first has since been half retracted**, and the retraction
  is the more useful finding. "Raising `k` beat adaptive `h` by seven orders on
  every benchmark, and h-adaptivity did not pay on any of them" holds only for the
  *indicator-driven equidistribution loop* that was tried, and only on the smooth
  problems. A mesh **graded geometrically towards a singularity**, at an unchanged
  cell count and unchanged DOF, is worth **14900×** on Shestakov — and the reason
  the earlier loop found 1.5× is that the accuracy indicator cannot see that
  error, which `MESH-REFINEMENT.md` §3 had already measured without drawing the
  conclusion. Grading is now a configuration option (`GradedGridBoundary`).

  What the measurements settled, all in `MESH-REFINEMENT.md` §8–§10:

  * the error on a graded mesh is `0.0487 h0` in the width of the cell touching
    the singularity and in *nothing else* — not the cell count — so the useful
    move is redistribution at a fixed budget rather than refinement;
  * the modal decay rate decides whether to grade and at which end, reliably from
    `k >= 3`, and at `k = 2` the decision is *inverted* rather than merely noisy;
  * so the order of an hp scheme is forced: **p, then h, then p**;
  * per-cell degrees are **gated no** — 217 `(k+1)` sites for a few percent of DOF,
    because the smooth region's degrees were measured to buy nothing.

  Still open: the driver that runs that sequence; and TimeMarch, which is blocked
  on exactly what this entry says.

- ~~Scope out coupling to a magnetic field solver.~~ **Done, in 1-D.** The
coupling described here is implemented: a `FieldModel` supplies `nFieldDOF`
unknowns `psi` and their residual rows, MaNTA appends them to the solution
vector after the global scalars, and the geometry the model derives from them
reaches the transport physics through `State::geom`. The Jacobian really does
have the block form this entry predicted,

    ( HDG Jacobian | A1 )
    ( A2           | B  )

with A1 assembled by the chain rule through the physics case's three
`d*_dGeometry` hooks and the model's `dGeometry_dpsi`. Both routes this entry
proposed are there and are selected by `FieldSolve`: the exact block elimination
(a Schur complement onto `psi`, `nField + 1` transport solves per Jacobian
solve) and the iterate-assuming-weak-coupling one (block Gauss-Seidel with
Irons-Tuck acceleration, one transport solve per sweep). The guess that the
exact route "would almost certainly be expensive" turned out to be the wrong way
round for every problem in the tree so far: the break-even is
`#sweeps < nField + 1`, and nothing here is on the winning side of it. The
iterative path is a bet on `N_magnetics >> N_HDG`, which is the regime this
entry was written for and which nothing yet exercises. See
`docs/field_coupling.rst`.

  What remains, and is not in scope of that work:

  * **A field model with real physics in it.** Nothing is registered in the tree
    — the two that exist are unregistered test fixtures — so `FieldModel` has
    nothing to name in the shipped binary. A 2-D Grad-Shafranov solve is the
    obvious first one; a DESC stellarator equilibrium is the other. Both plug in
    by overriding `updateFieldJacobian`, `applyB`/`applyBTranspose` and
    `solveB`/`solveBTranspose` so MaNTA never forms the model's Jacobian.
  * **`nScalars > 0` alongside a field model**, refused today for a reason
    recorded at `setFieldModel`.
  * **A Python or JAX field model.** `FieldModel` has no pybind11 class.
  * Two structural gaps in the coupled adjoint — an objective that reads
    geometry, and a field residual that depends on an adjoint parameter — both
    in `TODO`.
