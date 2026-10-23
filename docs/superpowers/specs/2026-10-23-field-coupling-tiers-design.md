# Field coupling in tiers: from a black-box equilibrium to the coupled DAE

## Why

`FieldModel` (the 2026-08-16 design, and the theta pinch built on it) asks a
field code for everything at once: unknowns `psi` in MaNTA's state vector, a
residual `R(psi, p)`, its Jacobian blocks `dR/dpsi` and `dR/dp`, the geometry
`g(psi, x)` and `dg/dpsi`, and the transposes of all of it for the adjoint.
That is the right contract for a field with dynamics of its own and for
gradients through the field. It is the wrong *entry* contract for the codes this
is for. An equilibrium code such as VMEC or DESC, driven as a black box, can
answer "given this pressure, what is the equilibrium" and very little else: it
has no residual in MaNTA's form, its unknowns are not a vector on MaNTA's grid,
and it exposes no derivatives.

What the transport needs from the field is a single map, from the pressure at
the physics nodes to the geometry at them,

```
g = G(p)        p: the field's input at the physics nodes
                g: the geometry slots at the physics nodes
```

and the tiers below are that map with progressively more information about it.
A model supplies what it can, the solver runs the best mode the information
supports, and a model grows into the coupled DAE by adding hooks, not by being
rewritten.

### What the modes cost, measured

The deciding metric is calls into the transport system, counted per physics
point: 1 for residual values, 2 for a derivative evaluation. Linear solves and
field-model work are free by comparison -- assembling the transport matrices
is the step that takes tens of minutes on GPUs for a GX-class flux. Measured on
`ThetaPinchDiffusion` at k = 4 with an experimental build of tier 0
(`FieldSolve = "consistent"`, the `EXPERIMENT`-marked code on
`feat/theta-pinch-field`, kept until tier 0 replaces it and can be compared
against it; `theta_pinch_mms_tests/compare_field_solves`, disabled by default,
is the harness),
values + 2 x derivatives, with the coupled modes' `dflux/dgeometry` calls shown
separately as `+2g`:

| 5 cells | transient | PTC | Newton |
|---|---|---|---|
| coupled, T = 0.3 | 7,950 (+1,200) | 775 (+500) | 325 (+200) |
| tier 0, T = 0.3 | 33,325 | 775 | 400 |
| coupled, T = 0.9 | 8,975 (+1,100) | 775 (+500) | 325 (+200) |
| tier 0, T = 0.9 | 31,850 | 850 | 550 |

At 32 cells the transient ratio runs 2.3x (T = 0.3) to 5.6x (T = 0.9), and at
1e-9 / 1e-11 tier 0 stalled IDA at order one (30000+ steps against ~210).

So:

* **Transients want the coupled Jacobian.** IDA reuses each Jacobian across many
  steps, so the cost is almost all residual values, and tier 0's linearly
  converging Newton costs IDA 3-5x the steps.
* **Steady solves are decided by what `dflux/dgeometry` costs.** Free (analytic
  prefactors, as in `ThetaPinchDiffusion`), coupling ties PTC and wins Newton by
  1.2-1.7x; priced like a derivative evaluation, tier 0 wins PTC and ties
  Newton; priced by differencing the flux in each geometry slot, tier 0 wins
  both.
* **The binding constraint is on the transport side, not the field side.**
  Every field-side derivative can be had by differencing the equilibrium, which
  is free on this metric. `dflux/dgeometry` cannot.

## The tiers

### Tier -1: no field, and nothing to see

A run with no field model, or a case that reads no geometry, sees none of
this. Today that holds by construction for the first -- `nGeom = 0`,
`evaluateGeometry` returns early, the layout has no `psi`, and every
regression config is byte-identical to the uncoupled build (CLAUDE.md, "The
whole thing is inert when unused") -- and holds only by accident for the
second.

New: **`SystemSpec::readsGeometry`**, a trailing, defaulted `false`, like
`FieldSpec::sourceReadsTimeDerivatives`. It is the case's statement that its
flux, sources and aux rows read `State::geom`. The solver uses it three ways:

* **No model, `readsGeometry = false`:** the path as it is today. Pinned
  byte-identical, by the hand check plus a unit test that a geometry-blind case
  never has geometry rows allocated.
* **No model, `readsGeometry = true`:** the case supplies its own fixed geometry
  (what `ThetaPinchDiffusion`'s `VacuumGeometry = true` does now), and the
  solver still evaluates nothing. A case that reads geometry, has no fixed
  geometry of its own and has no model attached is refused by the case, as
  `ThetaPinchDiffusion` already refuses it.
* **A model attached to a case with `readsGeometry = false`:** refused, by name,
  in `setFieldModel`. The field could not reach the solution, so the run would
  pay for it and change nothing. A field computed only for output is in "Not in
  this work".

Every pass-through is *structural*: a branch on a flag set once, never a hook
called to report "no contribution". Nothing per residual, per Jacobian build or
per solve may be spent finding out that there is no coupling.

### Tier 0: evaluate G -- the equilibrium as a black box

```cpp
class EquilibriumModel
{
public:
    /// The equilibrium for the input at the physics nodes, and the geometry
    /// slots there. `warm` is the model's own last answer for this run.
    virtual void Solve(Equilibrium &out, GlobalState const &states,
                       std::vector<Position> const &points, Vector const &weights,
                       Time t, Equilibrium const *warm) = 0;
    virtual void Geometry(MatrixRef g, Equilibrium const &e,
                          std::vector<Position> const &points) = 0;
    virtual void resetForRun() {}
};
```

`Equilibrium` is opaque to the solver, holding whatever the code caches (a
spline, Fourier coefficients, a file handle). The solver's side:

* **The field is a cache, not an unknown.** It is refreshed at every
  `physicsNodesAt`, so the residual, every Jacobian build and both
  initial-condition sweeps see the geometry of the current input. Nothing about
  it enters the DOF layout, the error test or IDA's Newton norms. That last part
  is not cosmetic: the experimental build kept `psi` in the state vector with
  rows `psi - psi*`, and zeroing its error weights changed `IDACalcIC`'s line
  search enough to fail at T = 0.9 where the weighted version completed.
* **Memoised on the input.** The residual is evaluated repeatedly at one state
  (KINSOL's merit function, a Jacobian build at the last residual's state), so
  the cache is keyed on the input bytes and a repeat costs nothing.
* **Warm-started, and reset per run.** The warm start is the cache from the
  last distinct input, cleared in `initialize()`'s unconditional part. A solver
  reused for a second run starts from the cold equilibrium like a fresh one:
  that is the `RF_cellwise` trap, and
  `a_coupled_solver_reused_matches_a_fresh_one_bit_for_bit` gets a tier-0
  sibling.
* **Converged well below IDA's tolerance.** An equilibrium that stops at its
  own looser tolerance puts that error into every residual, and IDA reads it as
  local error -- the same mechanism as the sweep stall the coupled MMS study
  found. The model is handed the run's `rtol` and `atol` and asked to resolve
  `g` to `1e-3` of the error weights; a model that cannot is warned about once,
  by name, at `initialize()`.
* **The Jacobian carries no coupling.** Transport derivatives are taken at
  `G(p_Jac)` with the geometry held fixed; there is no `FieldCoupling` call, no
  A1 and no A2.
* **Output and restart.** The geometry slots at the nodes go into the netCDF
  field group per output slice. A restart recomputes the cache from the
  restored state, cold; the restart file carries nothing for the field.
* **No adjoint through the field.** Refused by name with a field attached,
  rather than run as a frozen-field gradient, which is wrong by the field's
  share -- a third of the closed-form gradient on the theta pinch
  (`FieldAdjointTests.cpp`'s frozen-psi check).

### Tier 1: apply G' -- the coupled Jacobian without psi

Add the directional derivative

```cpp
/// dg = G'(p) dp at the cached equilibrium. Default: a forward difference of
/// Solve, warm-started from the cache, which is one equilibrium solve.
virtual void applyResponse(MatrixRef dg, Equilibrium const &e, GlobalState const &dp, ...);
```

and, on the transport side, the existing `dSigmaFn_dGeometry`,
`dSources_dGeometry` and `dAuxG_dGeometry`. The coupled Jacobian is then the
reduced one, with the transport unknowns alone:

```
J_red dx = r,    J_red = A + A_g G' P
```

where `A` is today's transport operator, `P` takes the input (`p` at the nodes)
out of `dx`, and `A_g` is the case's `dflux/dgeometry` projected into the cell
rows -- `dPhysics_dField_Mat` with `dg/dpsi` replaced by the identity. Two
solves, both matrix-free in G':

* **`iterative`:** today's block Gauss-Seidel with `psi` eliminated. A sweep is
  one transport solve, one `applyResponse`, one `A_g` product, and the
  stopping test is the one `solveCoupledJacIterative` has now, in the run's
  error weights.
* **`krylov`:** GMRES on `J_red`, right-preconditioned by `A^-1`, at one
  transport solve and one `applyResponse` per iteration. For an affine map this
  is what Irons-Tuck approximates with a rank-one secant, and it is the
  successor `TODO` already names for couplings with several eigenvalues outside
  the unit circle.

An *exact* tier-1 solve would assemble G' column by column, one equilibrium
solve per input node, which is free on the metric but `N(k+1)` solves per
Jacobian build. It is offered only as a verification tool, as `exact` is now.

**What tier 1 still lacks is the adjoint**, which needs G'^T. A forward
difference cannot supply a transpose action, but the assembled G' can, at
`N(k+1)` equilibrium solves per adjoint: free on the metric, so the adjoint is
available at tier 1 with an assembled response. That is a later step, below.

### Tier 2: the residual-form `FieldModel`, as now

`psi` in the DAE, `R`, `dR/dpsi`, `dR/dp`, `dg/dpsi` and the transposes.
Required for:

* **a field with dynamics** -- a differential `psi`, resistive evolution,
  anything that is not an instantaneous function of the transport state, which
  G cannot express;
* **the adjoint with no assembled response**, through A1^T, A2^T and B^T as
  now.

**Every tier-2 model is a tier-1 and a tier-0 model, by an adapter.** Its Newton
on the field rows alone is tier 0's `Solve` (the experimental `consistentField`,
moved), and `-B^-1 A2` applied to `dp` then mapped through `dg/dpsi` is tier 1's
`applyResponse`. So the theta pinch runs in every mode, which keeps every
comparison between tiers on one problem.

## Choosing the mode

From what is present, the way `PyTransportSystem` classifies a case as
pointwise or vectorised:

| field | case supplies `dflux/dgeometry` | mode |
|---|---|---|
| none, or `readsGeometry = false` | -- | tier -1 |
| tier 0 | no | `consistent` |
| tier 0 | yes | coupled, G' by differencing `Solve` |
| tier 1 | yes | coupled, the model's own G' |
| tier 2 | yes | coupled DAE, adjoint, field dynamics |

"Supplies" needs a declaration, since the hooks default to no-ops that mean
"no coupling": `SystemSpec::geometryDerivatives`, `None` / `Analytic` /
`Differenced`, where `Differenced` has the solver difference the flux in each
geometry slot (`n_geom` value calls per point per Jacobian build, a new
evaluation-plan site). The default is `None` when `readsGeometry` is set, so a
case is never silently given a coupled Jacobian built on zero blocks.

`FieldCoupling = "auto" | "consistent" | "coupled"` forces a lower mode than
`auto` would pick, and is refused if it asks for more than the model and case
supply. `auto` takes the coupled mode for time marching and for `Newton`, and
`consistent` for `PseudoTransient` when the case's geometry derivatives are
`Differenced` -- what the measurements above support. `FieldSolve` keeps
choosing among the coupled solves (`iterative`, `exact`, `krylov`).

## Testing

| test | catches |
|---|---|
| Tier -1: a geometry-blind case allocates no geometry rows and makes no field call; the regression configs byte-identical by hand, as now | a pass-through that is not one |
| `setFieldModel` refuses a model on a `readsGeometry = false` case | a field that pays and changes nothing |
| Tier 0 on the theta pinch (via the adapter) reaches the coupled solution's answers to the run tolerance, transient and steady | a cache read at the wrong input or the wrong time |
| Memoisation: a repeated residual at one state makes no `Solve`; a changed state always does | a stale cache, which is a wrong residual |
| A reused tier-0 solver matches a fresh one bit for bit | the warm start surviving `resetForRun` |
| Tier 1 `J_red` against a finite difference of the tier-0 residual (the `FieldJacobianTests` method), for a model's own G' and for the differenced default | a wrong response or a wrong `A_g` |
| `iterative` and `krylov` agree with tier 2's `exact` on the theta pinch, in the run's error weights | the reduced coupled solve |
| Physics-call counts per mode on the theta pinch pinned exactly, as `a_steady_solve_spends_the_physics_sweeps_it_has_to_and_no_others` pins the steady budget | a mode that quietly spends transport calls |
| The coupled MMS study (`ThetaPinchMMSTests.cpp`) run at every tier | a tier that converges to something else |

## Order of work

1. `readsGeometry` and the tier -1 pass-throughs, with the refusal. Small, and
   it fixes the contract the rest builds on.
2. `EquilibriumModel`, the tier-2 adapter, and the tier-0 mode with the field
   as a cache. Replaces the experimental `FieldSolve = "consistent"`, which
   keeps `psi` in the state vector.
3. `geometryDerivatives`, mode selection and `FieldCoupling`.
4. Tier 1: `applyResponse` with the differenced default, the reduced
   `iterative` solve, then `krylov`.
5. The tier-1 adjoint through an assembled G'.
6. Python: tier 0 is the natural first Python field model -- a function from
   the pressure to the geometry -- and the Python phase of the theta-pinch work
   should start there rather than at the tier-2 interface.

## Not in this work

* A field computed for output only, beside a geometry-blind case.
* A tier-0 or tier-1 field with time dependence of its own. That is tier 2.
* Noise in a black-box equilibrium. A code converged no tighter than its own
  tolerance is the noisy-flux problem moved to the field side, and belongs with
  `feat/noisy-transport-system`.
* Latching between modes during a run, e.g. falling back from coupled to
  `consistent` when the sweeps stop paying.
