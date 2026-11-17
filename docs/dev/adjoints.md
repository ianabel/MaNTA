# Adjoints

The adjoint solve, its gradients, the discretisation estimate and objective-driven adaptation.

`AdjointProblem` defines an objective `G = ∫ g dx` and its parameter
derivatives. `computeAdjointGradients` fills `G_p`: `adjointDerivativesAt` →
`assembleAdjointOperator`, once for every objective → per objective
`assembleObjectiveGradient` → `solveAdjointState` → `accumulateAdjointGradients`.
`np` splits into `np_boundary` trailing boundary parameters and the rest
internal; several defaults loop only to `getNpInternal()`.

`assembleAdjointOperator` assembles `M^T` per cell from the same derivative
blocks `assembleCellMatrix` assembles `M` from, so **the two must be kept in step
block for block** — the adjoint operator *is* that transpose. This is where the
two differ in consequence: a block missing from the forward Jacobian only slows
Newton down, but a block missing from the adjoint matrix produces a silently
wrong gradient with a perfectly good `G`. The `dSigma/dPhi` block was absent
here for exactly that reason, and cost nothing visible until
`python/Tests/test_adjoint_aux.py` was written. If you add a coupling to one
function, add it to the other.

**An adjoint solve makes no physics call.** Every `updateMatricesForJacSolve`
moves the derivatives it took into `jacobianDerivatives`, with the state, time
and `alpha` they were taken at, and the adjoint assembles from those at
`alpha = 0` into its own factorisations (`adjointMXSolvers`), leaving the
forward Jacobian's as it found them. What is reused is the *derivatives*, not
the factorisation: the forward one is of `MX` at the build's `alpha`. So:

* **On demand** -- `solveAdjointAt`, `Runner.solve_adjoint` -- it is solved from
  the last build whatever state that was at, and says so
  (`operator_at_state`). `adjoint_state`, `objective_gradient` and
  `objective_sensitivity` (`w = dG/dGamma` per node, for the flux and the
  sources) read it, `adjoint_state` and `objective_gradient` with the scalars'
  block beside the cells'. `tauScaling = "Diffusive"` is refused: the operator
  carries no `d tau / dy`.
* **At the end of a run** the last build is reused only when it was taken at
  exactly the final state -- `memcmp` on the state, the same time and `psi'` --
  which in practice means after `estimateObjective`'s build; otherwise
  `adjointDerivativesAt` evaluates once (the `Adjoint` plan site,
  `PerAdjointSolve`). A case whose sources read `du/dt` is always evaluated
  afresh. Note the final state is `y`, and the last build is usually an
  iterate behind it, so "reuse the build" without the bitwise check would move
  every gradient.
* `AdjointOnDemandTests.cpp` counts physics calls and requires zero, and pins the
  on-demand adjoint at the final state to the end-of-run one bit for bit. Every
  existing `z`, `G_p` and objective estimate hashed identical across the
  refactor.

**`w` is checked node by node, against data that separates `diag(w)` from the
mass matrix.** `the_sensitivity_matches_finite_differences_node_by_node`
finite-differences G under a single-node perturbation of the flux and of the
source, and carries a guard that the wrong operator misses by orders of magnitude
-- the lesson of the `DerivativeSubVector` trap below.

**The discretisation estimate** (`EstimateDiscretisationError`) is dual-weighted
against an adjoint solved at degree `k + 2` (`k + 3` under `Superconvergent`), on
`enrichedSolver`: a second `SystemSolver` on the same case and grid that **never
runs** -- no SUNDIALS objects, no `initialize()` -- and whose
`initialiseMatrices` skips `aFn` (`massFromHook`), so building it asks the case
for nothing. It shares the residual's body, `assembleTransportResidual`, so a
change to `residual()` reaches it; keep that shared. It costs exactly one
`ComputePhysics` and one `ComputePhysicsDerivatives` at the enriched nodes per
estimate, the `DiscretisationEstimate` plan site at `PerObjectiveEstimate`
cadence. One degree richer was measured and is not enough: the plain scheme's
`G` converges at `h^(k+2)` for even `k`, and `Superconvergent` at `k` reproduces
the plain scheme at `k + 1`. Two limits, both documented in `docs/adjoints.rst`:
an error that decays only algebraically in `k` is **underestimated with nothing
to flag it** (Shestakov, effectivity 0.43-0.80), and **the per-cell `eta_K`
cancel** -- `sum |eta_K| / |eta|` reaches 1e6 -- so they are not a localisation.

**`AdaptationTarget = "Objective"`** adapts `DegreeAdaptation` and
`MeshAdaptation` to `|eta| / (eps_disc |G|)` instead of `||u* - u_h||`. Because
of the cancellation above, **the objective decides *whether* to grade and the
modal decay still decides *where*** -- grading on the share of `eta_K` was built,
measured, and lost (false positives at 41 of 1217 smooth ends off uniform
meshes). It turns the estimate on and makes `"Objective"` the default steady
test, so both shares of `ObjectiveBudget` are enforced. The degree ceiling is
held so the enriched batch, `N(k+3)` or more, fits `MaxPhysicsBatch`
(`discretisationEstimateFits`); `ParallelFill` itself does not know about it.

Anything indexed per auxiliary variable is sized `nAux`, not `nVars`. Those
coincide in every fixture except `test_adjoint_aux.py`, so a confusion between
them is invisible in the rest of the suite; `dGdaux_Vec` had two.

**With `spatialParameters = true` each parameter takes its own value at every
*physics* node, and `G_p` is `(ng * nCells * nPhys, np)` — one row per
(objective, node), one column per parameter *field*.** `nPhys` is
`physicsNodesPerCell()`: `k+1` basis nodes, or `k+2` star nodes with
`Superconvergent = true`, because that is where the case evaluates the field.
`manta.getPhysicsNodes` returns the same points in the same order. Interpolating
a basis-node field onto the star nodes instead would make the parameter count
scheme-independent, and is the wrong trade: a coefficient like `kappa(x)` in
`kappa(x) q` is data at the points the flux is evaluated, and an interpolant of
it is a different coefficient.

Every entry is the **discrete** derivative `dG/dp_m`, and the two halves of
`accumulateAdjointGradients` have to agree on that, not merely on the shape:

* the adjoint term is `-z^T F_p` with column `m` of `F_p` the projection of node
  `m`'s value alone (`projectOntoTestSpace`: `A9` or the mass matrix), written to
  `G_p(gIndex * nPoints + i * nPhys + m, pIndex)`;
* the explicit term is `w_m dg/dp_m` (`PyAdjointProblem::explicitParameterDerivative`),
  `w` being the quadrature `GFn` integrates `g` with. `dgFndp` reports
  `(np, nPoints)`, so it is weighted and transposed on the way in;
  `checkDgFndpShape` refuses the other orientation by name, because
  `checkShapeAndSet` is a plain assignment outside `DEBUG` and a square mismatch
  would transpose silently.

An unweighted explicit term is a pointwise derivative beside a discrete one,
off by `1/w_m` — 20 to 100 at `k = 4` — and nothing that zeroes
`dgFndp` sees it, which is every value test that differences the *physics*.
`test_a_spatial_field_in_the_objective_is_weighted_by_the_quadrature` puts the
field in `g` alone, so `Runner.G()` on the one converged state differences it
exactly at every node. And note the sum test's blind spot: summing over nodes
cannot see a row-stride error in the adjoint term, which preserves the total —
only the node-by-node finite difference catches that.

**With `Superconvergent = true` the explicit term is taken at the star nodes, at
`u*`, with the star weights `b1`** — `AdjointProblem::dGFndp(gIndex, y, pp)`,
the derivative of `GFn(gIndex, y, pp)`, for spatial and scalar parameters alike.
Taken at `u_h` with the basis weights it is a derivative of a different
objective, 2.3% out at `k = 1`. That is invisible to any `g` *linear* in `u`,
because `u*` has the cell means of `u_h` by construction (paper I's local
Neumann problem constrains them), so the check needs a `g` at least quadratic in
`u` and a `k` at which `u*` differs from `u_h`. A C++ case's `dGFndp(g, p, y)`
integrates for itself and leaves no nodal values to weight, so the C++ base
falls back to it; such a case differentiates the `u*` objective by overriding
the `pp` overload, as `AutodiffAdjointProblem` does. `Runner.G()` reports the
`u*` objective too, as `getAdjointGradients()` does.

**`GFn(g, y)` is the nodal sum `Σ_j w_j g(Z_j)` by default, and an override
must keep it that.** That sum is the functional `dg` differentiates
(`assembleObjectiveGradient` weights the nodal `dg/dZ` by `diag(w)`), so
integrating `g(u_h)` any other way -- exactly, say -- reports a different
objective from the one the adjoint differentiates. The two differ only at the
level of the quadrature error, which a finite-difference gradient check at a
loose tolerance cannot see; `the_autodiff_objective_is_the_functional_dg_differentiates`
checks the pair at round-off, against a guard that the exact integral is
visibly different.

Only Python can set the flag (`Python.cpp` exposes it on `PyAdjointProblem`;
the C++ base defaults false and `AdjointProblemTests.cpp` asserts as much).

The spatial gradient is exact per node as well as in total, on both node sets,
pinned in `test_adjoint.py`. Getting there turned up the more general trap below.

**Global scalars border the adjoint as they border the forward solve.**
`solveTransportJac` solves `[[A, -V], [W^T, N]]`, `v` being the projected
`dSources_dScalars`, `w` the scalar rows' `dG_s/dy` and `N` their `dG_s/dmu`;
`solveBorderedAdjoint` solves its transpose. The coupling is cached like the
field's: `evaluateScalarCoupling` is the physics half (`dSources_dScalars`,
`ScalarGPrime`), each build keeps its result in `jacobianDerivatives.scalar`
beside the transport derivatives (and `stateDot`, since `ScalarGPrime` reads
it), and `assembleAdjointOperator` applies it at `alpha = 0` into storage of the
adjoint's own -- `v`, `w` and `N_global` keep the forward solve's alpha -- so an
adjoint on demand still makes no physics call. A record that lacks it is refused
by name rather than indexed. The gradient gains `-z_mu^T dG_s/dp`, from `AdjointProblem::dScalarG_dp`, which has
no default: an adjoint problem must say a constraint does not read the
parameters by returning zeros. The objective's `dG/dmu` is `dg`'s `Scalars`
entry, already integrated. Without the border the adjoint treats the scalars as
constants, and the gradient misses the share that comes through them — 2-6% on
a one-scalar diffusion problem, `mu - int u dx = 0` feeding the source, with `G`
itself exact.

`test_adjoint_combinations.py` checks every combination of aux, scalars, spatial
parameters and the superconvergent scheme against finite differences, and its
fixture is shaped by one trap: **a transposition in the border is invisible
unless the problem is asymmetric enough to see it.** With one scalar every block
is 1x1 and a transposed `N` or Schur complement is exact. With two, it was
*still* exact while the source read both with constant weights (so `V` had rank
one) and the second constraint weighted `u` by `x`: `u` is symmetric about
`x = 1/2`, so `int x u = int u / 2`, the same ratio as the source's, and
`V^T A^-T W` came out symmetric. Hence `mu + x nu` in the source and `x^2` in the
constraint. Each of the six ways to break the border — no border, `N` for `N^T`,
swapped Schur indices, no `G_mu`, no `dScalarG_dp`, no `G_mu . delta_mu` in
`estimateObjective` — fails it, and the on-demand test fails on a build that
keeps no scalar coupling or an operator that assembles none. The `"Objective"`
steady test takes the same `G_mu . delta_mu`; the discretisation estimate, whose
enriched solver records derivatives of its own without the scalars, still
refuses them.

**The adjoint's `dG/dZ` is `diag(w)`, not the mass matrix, and the difference is
invisible in every aggregate.** `GFn` reports `G = Σ_m w_m g(Z_m)` — the
interpolatory quadrature, `∫ I_h[g] dx` — so the exact derivative in the nodal
coefficient `Z_i` is `w_i dg/dZ|_i`. `DerivativeSubVector` used to apply
`InterpolateOntoBasis`, i.e. the cell mass matrix `M`. Since **`M·1 = w`
exactly** — a mass matrix's row sums *are* the quadrature weights — the two
agree whenever `dg/dZ` is constant across a cell and otherwise differ by
`(M − diag(w))·dg/dZ`, an operator that annihilates constants.

That last property is the whole reason it survived: the error summed to zero
over every cell, so `G_p.sum()`, the scalar-parameter gradient (exact to 7e-16
against a closed form) and every finite-difference check in the suite were
untouched. It was reachable *only* per node — i.e. only through spatial adjoint
parameters — where it appeared as an error set purely by the intra-cell node
index, symmetric and alternating in sign, decaying as O(h^4). That decay is a
convergence rate of an inconsistency, not a discretisation order: refining the
mesh hid it and nothing removed it. The generalisation worth keeping is that
**two quadratures agreeing is not evidence either is right** — the fixture that
should have caught this instead pinned the defect, because it differenced an
exactly-integrated `GFn` that no real case reports.

The pointwise `DerivativeSubVector` overload and the `dGdu_Vec`/`dGdq_Vec`/
`dGdsigma_Vec` wrappers over it are gone — they computed `∫ dg/dZ φ_i dx`, the
derivative of `∫ g dx`, and no solve ever called them. `dGdaux_Vec` was the last
one left and is now the same operator over `nAux` blocks: it takes the nodal
`dg/dphi` from the batched `dg` and weights it. A C++ case's `dgFn_dphi` still
reaches it, through `AdjointProblem::dg`'s default, which samples the hook at the
nodes; a Python case supplies `dg` and `PyAdjointProblem::dgFn_dphi` raises.

**Beware how nearly `diag(w)` and `M` agree — it is far more than the constants
the argument above needs.** `(M v)_i = ∫ φ_i v` and `(diag(w) v)_i = v_i ∫ φ_i`,
so they coincide whenever the interpolatory rule integrates `φ_i v` exactly; on
`k+1` Chebyshev points of the first kind that rule is symmetric, hence exact to
degree `k+1` for even `k`. At `k = 2` — which is what the adjoint fixtures use —
that covers every *affine* `dg/dZ`, and the mocks' hooks are affine in `x`. Both
`the_derivative_sub_vector_weights_dg_by_the_integration_weights` and its aux
sibling therefore passed with the mass matrix reinstated, by 3e-16 and 5e-16,
until each was given a second half driven by a synthetic degree-`k` `dg/dZ` and a
guard that the two operators still differ on it. Those two guards are the whole of
the coverage, so keep them: a reference built "straight from the weights" pins the
formula, not the operator, if the data cannot tell them apart.
