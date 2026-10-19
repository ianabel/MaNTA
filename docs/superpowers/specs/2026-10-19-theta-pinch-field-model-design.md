# The first real field model: a diamagnetic θ-pinch in a flux-conserving wall

Date: 2026-10-19
Status: draft, for review

## Why

`FieldModel` has been on `main` since #24, and nothing registered uses it. The
two models in the tree are unregistered fixtures under `Tests/UnitTests`, so the
shipped binary has nothing to name in `FieldModel`, there is no coupled `.conf`,
no coupled regression case, and several pieces of the machinery exist only in
fixtures:

* `FieldSolve = iterative` has **never won**. The break-even is
  `#sweeps < nField + 1`, and the fixtures have `nField` of 1 and 5.
* No model overrides `updateFieldJacobian` / `applyB` / `solveB` and their
  transposes, so the seam a structured field solver is meant to plug into has
  never had anything plugged into it.
* No physics case in `PhysicsCases/` reads geometry. The autodiff layer's
  `d*_dGeometry` hooks are covered by one test fixture
  (`GeometryDerivativeTests.cpp`).
* A coupled run has never written output, restarted from a file, or been driven
  from a config, except in unit tests.

This spec builds one model with real physics, small enough to verify against
closed forms, and uses it to run every one of those paths. The model is not the
end goal: if the infrastructure turns out wrong, this is how we find out.

## The physics

A straight cylinder of radius `a`, axially uniform, with a perfectly
conducting wall that conserves the axial flux `Φ_a`. The plasma carries no axial
current (`B_θ = 0`, a θ-pinch). Radial force balance is

```
d/dr ( p + B_z² / 2μ₀ ) = 0     ⇒     p + B_z² / 2μ₀ = C(t)
```

on every surface. The constant `C` is not given. It is fixed by flux
conservation: the wall does not move and the flux inside it cannot change, so

```
∫₀ᵃ 2π r B_z dr = Φ_a .
```

Raising the pressure therefore digs a diamagnetic well: `B_z` falls on axis,
rises at the wall, and the flux surfaces move outward in `r`.

### The coordinate is the flux label, and that is what makes it geometry

Transport runs on the normalised axial flux `x = Φ / Φ_a ∈ [0, 1]`, not on `r`.
This is JETTO's resolution (see the #24 design's reading of `refs/JETTO.pdf`).
Labels are fixed and surfaces move, so the transport mesh never regrids and the
response of the equilibrium appears purely as *metric coefficients* on a fixed
grid. With `A(x)` the area enclosed by surface `x`, and since `dΦ = B_z dA`:

```
V'(x)        = dA/dx         = Φ_a / B_z(x)
A(x)         = ∫₀ˣ V' dx'
⟨|∇x|²⟩      = (dx/dr)²      = 4π A / V'²
```

All three are per unit length. A transport problem on `r` would see no geometry
change at all, only a field strength in its coefficients. The flux label is
what makes this a geometry response, and producing that response is the purpose
of the work.

### The field unknown is `V'` itself

`V'` is a better unknown than `B_z`, for two reasons:

* **Flux conservation becomes linear**, `∫₀¹ V' dx = π a²`: the wall's area.
* **`A(x)` is an exact integral of the representation**, so the geometry costs
  no quadrature and `dA/dψ` is exact.

`B_z = Φ_a / V'` is then derived.

`V'` is a cubic B-spline on the model's own uniform grid of `M` intervals
(`M + 3` coefficients `v_n`), plus the unknown `C`. So `nField = M + 4`, all
algebraic. The rows are Galerkin against the same B-splines `φ_m`:

```
R_m  = ∫₀¹ φ_m ( Φ_a² / V_h'² + 2μ₀ p_h − C ) dx  = 0       m = 0 … M+2
R_C  = ∫₀¹ V_h' dx − π a²                          = 0
```

**Why a C² spline and not a P1 field.** The transport flux needs `∂_x V'` (next
section). A field that is only C⁰ hands the transport discontinuous coefficients
*inside* its cells wherever a field knot lands there, and the transport scheme
loses its order to that. A cubic spline is C², so the geometry is smooth
everywhere and its own error is `O(h_f⁴)`.

**Why the model's own grid and not the transport nodes.** It keeps `nField`
independent of `k` and of the transport mesh. A restart onto a different degree
can then carry `ψ` across verbatim (see Restarts below). And the field
resolution becomes a dial of its own, which is exactly what is needed to put
`nField ≫` the transport's own count and test whether `iterative` wins.

### How the rows see the pressure: an auxiliary variable

The field model reads **a pressure in physical units** and nothing else about
the transport. The transport case exports it as an auxiliary variable,
`AuxG = φ − T N / V'`, and the model's config names which aux index holds it.
This keeps the model physics-agnostic: it does not know what `N` is, whether
there is a temperature, or how many variables the case has. It also exercises
aux and field coupling together, which nothing in the tree does today. `A₂`
already carries aux columns (`applyFieldCoupling` writes
`row.Aux(a).getCoeff(i)`).

`FieldResidual` gets `φ` at the `k+1` basis nodes of every transport cell
(even under `Superconvergent`), with no geometry rows. The model rebuilds the
per-cell Lagrange interpolant from those nodal values. That is exact, since a
nodal DG field *is* its interpolant. It then integrates `φ_m p_h` over each
(field interval ∩ transport cell) with a Gauss rule of order `3 + k`:

```
∫ φ_m p_h dx = Σ_j P_mj p_j,     P_mj = ∫ φ_m ℓ_j dx
```

`P` depends only on the node positions, so it is computed on the first call and
recomputed only if `points` changes. `dR_m / dp_j = 2μ₀ P_mj` is constant.

**This departs on purpose from "use the weights you are handed".** That rule
exists because the weights integrate a polynomial on each transport cell
exactly, and a rule of the case's own choosing integrated something else
(`ScalarTestLD3`, 8% out). Here the integrand is not polynomial on a transport
cell: `φ_m` has knots inside cells. So the weights would be the inexact rule,
and the exact one is the interpolant integrated piecewise. The term in `V'`
alone, `Φ_a² / V_h'²`, is the model's own function on its own grid and is
integrated with its own rule per field interval.

### Units

The model works in normalised units: lengths in `a`, field in the vacuum field
`B₀`, pressure in `B₀² / 2μ₀`. So `Φ_a = π`, `2μ₀ = 1`, and `p` *is* the local
β. The rows become `R_m = ∫ φ_m (π² / V'² + p − C)` and `R_C = ∫ V' − π`. The
vacuum solution is `V' = π`, `C = 1`, which is also `InitialFieldValue`.

### Geometry slots

| slot | value | used for |
|---|---|---|
| `Vprime` | `V_h'(x)` | `N = V' n`, the aux row, the source |
| `dVprime_dx` | `V_h''(x)` | `∂_x n` from `q = ∂_x N` |
| `gradx2` | `4π A_h / V_h'²` | the flux |
| `B` | `π / V_h'` | the transport coefficient |

`dGeometry_dpsi` is exact. `Vprime` and `dVprime_dx` see only the four splines
that are nonzero at `x`. `gradx2` sees every coefficient whose support starts
left of `x`, through `A`, so it is dense lower-triangular in `ψ`, the nonlocal
coupling the #24 design anticipated. No slot depends on `C`.

**Geometry is cached on `ψ`.** `Geometry` is called once per point with the same
`ψ`. The cumulative `A` at the knots is recomputed only when `ψ` differs from the
cached copy, compared bitwise, the way the end-of-run adjoint compares states.

### The model's own block, and the custom solve

```
B = [ K    −c ]      K_mn = −2π² ∫ φ_m φ_n / V_h'³ dx    (banded, 3 either side, symmetric)
    [ dᵀ    0 ]      c_m  = ∫ φ_m dx,   d_n = ∫ φ_n dx   (so d = c)
```

This is a saddle-point system with a one-row, one-column border. `K` is
symmetric and negative definite, and the border is symmetric up to its sign.
`updateFieldJacobian` extracts `K` into a `util/BandedMatrix.hpp` band and
factorises it. `solveB` / `solveBTranspose` eliminate the border with two banded
solves (`K⁻¹ r` and `K⁻¹ c`) and a scalar Schur complement `dᵀ K⁻¹ c`. `applyB`
and `applyBTranspose` are banded matvecs. Total cost is `O(M)` per solve, against
the base class's dense `O(M³)` LU.

**Two interface costs this does not remove**, both recorded for later rather
than fixed here:

* `FieldResidualPrime` hands the model a dense `nField × nField` `dRdpsi`, and
  the base constructor allocates a dense `B`. At `M` in the hundreds that is
  hundreds of kilobytes per Jacobian build: tolerable, and the model fills only
  the band.
* `A₁` is stored dense per transport cell, `(3nVars + nAux)(k+1) × nField`, and
  `a2` is `nField` full-length vectors. That is `O(nField · N_transport)`.

Both are where scaling stops. This model will measure how far it gets before
they bite, which is better than guessing.

### Failure

`V_h' ≤ 0` anywhere on the model's quadrature points makes `B_z` meaningless.
`FieldResidual` throws, `static_residual` returns 1, and IDA cuts the step. This
is the "failing gracefully" path `docs/field_coupling.rst` describes and nothing
has exercised. A test drives a state into it deliberately.

## The transport case: isothermal density in flux coordinates

`ThetaPinchDiffusion`, on `AutodiffTransportSystem`, so the three
`d*_dGeometry` hooks come for free.

**The evolved variable is `N = V' n`**, particles per unit flux per unit length.
In flux coordinates that move with the field, continuity is

```
∂_t (V' n) + ∂_x ( V' ⟨Γ·∇x⟩ ) = V' S
```

and it is exactly conservative **with `V'` time-dependent**, because `V' n` is
what sits under `∂_t`. Evolving `n` would need `n ∂_t V'`, a geometry time
derivative, which is the unbuilt phase 2 of
`2026-09-21-time-derivative-sources-design.md`. Choosing `N` avoids it entirely.
`aFn` is 1.

**The flux.** With `Γ·∇x = −D ⟨|∇x|²⟩ ∂_x n` and MaNTA's sign convention
(`σ̂` enters as `−∂_x σ̂`):

```
σ̂ = V' D ⟨|∇x|²⟩ ∂_x n,     n = N / V',     ∂_x n = (q − n V'_x) / V'
D = D₀ (B_ref / B)^γ        γ = 1 Bohm-like, γ = 2 classical-like
```

`V' ⟨|∇x|²⟩ = 4π A / V'` vanishes on axis, so the axis flux is zero by
construction and a Neumann condition there is consistent.

**The aux row** is `φ − T N / V' = 0`, with `T` a constant normalised temperature.

**Source**: `V' S(x)`, a fuelling profile from config.

**Boundaries**: zero flux at `x = 0`. At `x = 1`, Dirichlet on `N` with a
constant datum. **The datum is `N`, not `n`**, because `LowerBoundary` /
`UpperBoundary` cannot read geometry. So a changing `V'(1)` moves the edge
density slightly. That is stated in the case's documentation rather than hidden.
A mixed outflow condition `σ = −γ_s N`, which is geometry-free, is the
alternative if that matters.

**It is isothermal by assumption.** An energy equation needs the compression work
`p ∂_t V'`, i.e. the same geometry time derivative. That is the natural next
step, and it is blocked on that phase rather than on this one.

## Bugs this walks into, fixed as part of it

1. **A projected coupled restart starts from `ψ = 0`.** The projection path
   (`SystemSolver.cpp` ~683-720) transfers `u`, `q`, aux, the scalars and
   `λ`, never `ψ`, and `InitialFieldValue` is skipped on every restart. For this
   model that is `V' = 0`, i.e. an immediate throw. Fix: on the projection path,
   copy `ψ` verbatim when the file's `nField` equals the model's (which is the
   only case that reaches there, given the existing refusal). Pin it with a
   degree-changing coupled restart.
2. **`AutodiffTransportSystem::writeDiagnostics` evaluates `AuxG` on a state with
   no geometry rows** (`State s = y.eval(x)`, ~591). This case's aux row reads
   `Vprime`, so output would read out of bounds, silently outside `DEBUG`. Fix:
   build the state with geometry, through `evaluateGeometry`'s path, or refuse.
   A test with this case and `WriteDatFile` / diagnostics on catches it.
3. **`EvaluationPlan.hpp:45`** says `InitialCondition` runs on the basis nodes.
   It runs on the physics nodes. The comment gets corrected.

Two known gaps stay open and are named, not fixed: an objective reading geometry
loses `dG/dψ` (and `GFn` sees no geometry rows at all), and `∂R/∂p` has no hook.
The adjoint test below uses an objective and a parameter that touch neither.

## Testing

The #24 design's three failure classes still apply, and each needs its own
test.

| test | catches |
|---|---|
| Spline basics: partition of unity, exact integral, `V''` against differences | the representation |
| `dGeometry_dpsi` against finite differences, every slot, including `gradx2`'s lower-triangular block | a wrong `A₁` second factor |
| `FieldResidualPrime` against finite differences: `dRdpsi` and `dR` through the aux column | a wrong `B` or `A₂` |
| `solveB`/`solveBTranspose`/`applyB`/`applyBTranspose` against a dense LU of the same `B` | a wrong custom solve, e.g. the border sign |
| Standalone equilibrium: prescribed `p(x)`, Newton on the field rows alone, against the closed form `V' = π/√(C − p)` with `C` from a scalar root-find; local orders in `M` approaching 4 | a wrong residual |
| Coupled `J dy = g` with `FieldSolve = exact` (the `FieldJacobianTests` method) | a wrong coupled Jacobian |
| MMS on the coupled system: manufactured `n(x,t)`, with `p = T n` independent of `V'`, so `V'_exact` is closed form; local orders in `h` at fixed fine `M`, and in `M` at fixed fine `h` | a converged, plausible, wrong coupled residual |
| Adjoint gradient against finite differences, coupling armed, parameter `D₀` | a wrong `A₁ᵀ` / `A₂ᵀ` / `Bᵀ` |
| `exact` and `iterative` agree; reuse is bit for bit (`a_coupled_solver_reused_matches_a_fresh_one`) | the iterative path and `resetForRun` |
| `V' ≤ 0` throws and IDA recovers | the recoverable-failure path |
| Low-β limit against `B ≈ 1 − p/2` | a sanity check on units |

**Registered, with a `.conf`, a regression reference and a restart round trip**
in `Tests/RegressionTests`. These are the first coupled ones, and they retire the
"no field model is registered" notes in `CLAUDE.md`, `FEATURES.md`, `TODO` and
`docs/field_coupling.rst`.

**The measurement this exists to make.** Wall time and Jacobian-solve counts for
`exact` against `iterative` over `M ∈ {8, 32, 128, 512}` at fixed transport
resolution, with sweep counts and fallbacks. If `iterative` wins anywhere, the
"no fixture is on the winning side" statement (five places, per `TODO`) changes
with it. If it never wins, that is the finding, and it goes in the same five
places.

## Python (second phase)

After the C++ case is in:

* Allow `FieldModel` (and the four `FieldSolve*` keys) in a `Runner.configure`
  dict, naming a C++-registered model. Today it is a `ProblemSelection` key and
  an error there. The model's own table arrives as a nested dict, which means
  `DictConfigSource` handing the model a `toml::value`, a conversion that does
  not exist yet.
* Lift `PyRunner`'s blanket refusal of coupled restart files.
* A Python version of the transport case reading `s.geom`, checked against the
  C++ case on the same config. `StateView.geom` is indexable by position only.
  Indexing it by slot name, as `u`/`q`/`phi` already are, is part of this.

A Python-*implemented* field model stays out of scope, as in #24.

## Deliberately not in this work

* `B_θ` from a prescribed or evolved current (a screw pinch). The balance then
  becomes a radial ODE and `K` a genuinely elliptic band. It is the obvious
  follow-on, and the spline/banded machinery here is what it would reuse.
* An energy equation. It needs `∂_t V'` (time-derivative sources, phase 2).
* Adaptation with a field model attached. It stays refused. The model's own grid
  makes it *possible* later, since `nField` would not change.
* `dRdot` / `states_dot` in `FieldResidual` (`TODO`). This model's rows have no
  time derivative.
* Sparse/structured `dRdpsi` and `A₁` storage in the interface.
