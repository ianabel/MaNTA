# Self-consistent magnetic fields, as a coupled algebraic block

Date: 2026-08-15
Status: approved, ready for an implementation plan

## Why

`FEATURES.md`'s third roadmap entry asks for a magnetic field that responds to
the plasma rather than being read from a file. Its own summary of the
algorithmic situation is right: a self-consistent field is a large set of
algebraic constraints, which IDA already handles, and the question is what
happens to the Jacobian solve when the coupled system has the block form

```
( HDG Jacobian | A1              )
( A2           | B^{GS} Jacobian )
```

with `N_magnetics >> N_HDG`, because MaNTA's static condensation only solves the
top left.

Today MaNTA has no route to this at all. `PhysicsCases/MagneticFields.hpp` is a
*static* geometry interface — `Psi_V`, `B`, `VPrime`, `MirrorRatio`, `L_V`, a
flux-surface average — read from analytic formulae or from a netCDF file, and it
has no live C++ consumer: `AdjointPlasma` includes it with its field member
commented out, and `MagneticFieldTest.cpp` is the whole of its coverage. The
physics that actually runs carries its geometry in Python
(`python-physics/mirror-plasma`, `python-physics/stellarator`), and the
stellarator stack reaches DESC only from an *outer* optimisation loop, never
from inside a solve.

## What the literature settles, and what it does not

Three references were read in full before designing (`refs/Refs.md` indexes
them; the PDFs are gitignored).

**`refs/TORAX.pdf`** — Citrin et al., a differentiable tokamak transport
simulator solving heat, particle transport *and* current diffusion as one
system. Its "self-consistent field" is the poloidal flux `psi` alone on a
**frozen** geometry: every metric coefficient (`V'`, `g2`, `g3`, `F`, `Phi_b`)
is an input, read once from a CHEASE file, and §IV A puts geometry
categorically in the known-input vector `u`, never in the state. There is no
Grad–Shafranov solve in the loop, not once per step and not once per run. Even
their roadmap item for time-dependent geometry names *interpolation between
precomputed equilibrium files* as the mechanism. So the most advanced code in
this space treats the geometry response as out of scope, and what it
demonstrates is narrower than it looks: a differentiable *residual* driving a
Newton solve, with end-to-end sensitivity unimplemented, forward-mode, and
deferred.

**`refs/JETTO.pdf`** — Cenacchi & Taroni, the prior art. First-order Lie
splitting: transport on the old geometry for a macro-step, then one equilibrium
solve, then the next transport step. The cadence `Delta t_eq` is a literal list
of times in the input deck, not adaptive. Two findings transfer and one warning
does.

* **The real trick is that it never regrids.** `rho_max` is held constant, so
  flux surfaces move in `(R,Z)` while their *labels* do not. There is no
  mesh-to-mesh projection anywhere. The remap is in the profile *values*, via
  the invariants `n V' = n* V'*` and `p (V')^{5/3} = p* (V'*)^{5/3}` at fixed
  `rho`.
* **The interface is a small vector of profile fields**: `V'(rho)`,
  `<1/R^2>`, `<(grad V/R)^2>`, `<|grad rho|^2>`, `f(rho)`, and `R_max`/`R_min`
  for the inverse aspect ratio. Per-`rho` data, no `(R,Z)` leakage into the
  physics.
* **The warning**: JETTO *asserts* that the splitting avoids accumulating
  numerical error and defers the justification to a conference paper. There is
  no stability analysis, no criterion for `Delta t_eq`, no under-relaxation, and
  no diagnostic that would reveal a bad choice. That is precisely the failure
  mode this repository keeps flagging — converged, plausible, wrong, with
  nothing reporting it.

**`refs/NewtonGSMFEM.pdf`** — Serino et al., the paper closest to what the
roadmap entry assumes exists. Two findings materially change the design.

* **The differentiability the roadmap assumes is not there.** They have
  `dB/d(alpha)` for a single scalar profile amplitude; there is no derivative
  with respect to a profile as a function, no forward sensitivity, no reduced
  gradient, and no derived-geometry output at all. Adding `dB/d(theta)` is easy
  — a pure volume integral needing none of the shape calculus — but the deeper
  obstacle is that **their residual is only piecewise smooth in `psi`**: the
  x-point is located by a discrete search over mesh vertices and the plasma
  domain by a discrete tree search, so the sensitivity is exact within a piece
  and jumps when the saddle hops a vertex or an element enters `Omega_p`. Their
  own non-monotone residual history, and the case where a *more accurate*
  W-cycle preconditioner raised the Newton count from 5 to 15–20, both read as
  symptoms of that.
* **Cost, and its one redeeming number.** They report no wall times and no DOF
  counts, so this is reconstructed: `N_dof ~ 10^3`–`10^4` at demonstrated scale,
  `10^5`–`10^6` in production. But the DOF ratio understates the asymmetry,
  because the two solvers are in different complexity classes — one equilibrium
  is 15–176 FGMRES iterations each costing two AMG applications, so `10^2`–`10^4`
  AMG cycles, against a single direct condensation pass per MaNTA Newton step.
  The redeeming number: a **warm-started** equilibrium costs ~2.0 Newton
  iterations against 4–8 cold, uniformly across every preconditioner and every
  case in their Table 2.

That cost finding looks fatal to per-residual coupling and is not, and the
distinction is the one the design rests on. **What is expensive is calling an
equilibrium *solver*; what this design evaluates per residual is the equilibrium
*residual*** — a sparse matvec plus a nonlinear source assembly. In a coupled
formulation there is never a standalone equilibrium solve: its Newton is
subsumed into IDA's, and the equilibrium's Jacobian is applied once per Newton
iteration, which is the count it would have had anyway. Structurally, every
coupled step is a warm start.

## Decisions taken

1. **A configuration-agnostic interface, with one concrete first client.** Not a
   mirror-specific or tokamak-specific coupling.
2. **The field is coupled inside every residual evaluation.** The constraint is
   part of the DAE and IDA's error test sees it. This is `FEATURES.md`'s own
   reading and it is affordable for the reason above.
3. **The provider declares its own label** and supplies the metric on it. MaNTA
   stays agnostic about whether `x` means flux-surface volume, normalised
   toroidal flux, or anything else. This is cheap because `MagneticField` has no
   live C++ consumer to retrofit.
4. **The first client is a manufactured equilibrium in C++**, verifiable against
   closed forms, with no physics value of its own.
5. **The adjoint coupling is in v1.** This is the one place a missing block is
   silently wrong rather than merely slow.

## Architecture

### Geometry is derived, not an unknown

The field model's DOFs `psi` join the IDA vector. The metric fields are a
*function* of `(psi, x)`, evaluated at the physics nodes and cached per
residual — the same standing as `sigmaHat` or the existing source cache. So the
state grows by `nFieldDOF` only, and the DOF layout gains one block at the end:

```
[ sigma | q | u | aux ] per cell,  then lambda,  then mu,  then psi
```

Additive to `DGSoln::getDoF` and `DGSoln::Map`; nothing existing shifts.

### Geometry reaches a case through `State`, not a changed signature

`State` gains a `geom(g)` accessor beside `u`/`q`/`sigma`/`sigmaHat`/`phi`/
`scalars`, and `GlobalState` a `"Geometry"` key. `SigmaFn(i, State, x, t)` and
every other hook signature is unchanged, so the trampoline, the JAX adapters,
the generated stubs and every existing physics case are untouched.

### The new hooks are the derivatives

`dSigma_dGeometry`, `dSources_dGeometry`, `dAux_dGeometry`. They follow the
established convention — derivative out-parameters arrive zeroed, so a case that
does not read geometry contributes an identically zero block and pays nothing.
`AutodiffTransportSystem` gets them by widening its `RealVector`.

**Why not auxiliary variables.** Routing the metric fields through `nAux`, with
`AuxG_g = phi_g - V'(x; psi)`, would have reused the entire aux machinery and
needed no new hooks at all. It was rejected: aux variables are strongly coupled
with an exact Jacobian inside the cell-local `MX` block, whereas geometry's
dependence on `psi` is nonlocal and wants a different solver structure. Forcing
it through `MX` would either be wrong or make that block dense for every case.
Keeping the coupling outside `MX` is the point — `MX` stays cell-local and
exact, aux included.

### The field model declares itself as data

A `FieldSpec`, built once, validated once, handed to the `FieldModel`
constructor, immutable after — the way `SystemSpec` already works, so a
part-built model cannot exist. It carries:

* `nFieldDOF`, and per-DOF differential/algebraic flags;
* the label it declares;
* the geometry slots it fills, with names, descriptions and units;
* which quantities it reads back from the transport state.

The physics case never names the field model. It reads geometry slots declared
in the spec. A mirror model and a GS model differ in what they fill, not in the
interface a case is written against.

### Hooks on `FieldModel`

```
FieldResidual(psi, dpsi_dt, state, t)          -> residual rows
Geometry(psi, x)                               -> the metric at a point
dFieldResidual_dpsi                            -> B, as an operator
dFieldResidual_dState                          -> A2, as an operator
dGeometry_dpsi                                 -> the second factor of A1, as an operator
```

Every one of the three operators must offer `apply` **and** `applyTranspose`,
and `B` must additionally offer `solve` and `solveTranspose`. The transposes are
not optional extras for a later adjoint: the adjoint sweep runs the block
elimination in reverse and needs all three, and the asymmetry below is why a
model that supplies only the forward direction cannot be silently accommodated.

## The Jacobian solve

With `A` the existing transport operator — HDG condensation plus the scalar
bordering, i.e. today's whole `solveJacEq`:

```
[ A    A1 ] [ dx   ]   [ r1 ]
[ A2   B  ] [ dpsi ] = [ r2 ]
```

where `A1 = (dPhysics/dGeometry) . (dGeometry/dpsi)`, the first factor per-node
from the new hooks and the second the field model's and generally dense.

**Both solves are built, and the choice is declared rather than inferred.**

*Exact Schur onto `psi`*: `(B - A2 A^-1 A1) dpsi = r2 - A2 A^-1 r1`, costing
`nFieldDOF` applications of `A^-1`. Affordable only for a small field block. It
earns its place for a reason beyond being an oracle: `SolveJacTests.cpp`
finite-differences the residual and demands `J dy = g`, and that test style only
extends to the coupled system if an exact solve exists.

*Block Gauss–Seidel*, the production path: sweep `A dx = r1 - A1 dpsi^k`, then
`B dpsi^{k+1} = r2 - A2 dx`. One transport solve and one field solve per sweep.
This is `FEATURES.md`'s "assume that the coupling is weak, and iterate", and it
is safe for the reason `CLAUDE.md` already states — the Jacobian is never
assembled, IDA tolerates an inexact linear solve, and an error there costs
Newton speed rather than correctness. Serino's block-triangular preconditioners
drop the Schur complement outright and still converge.

Selection is a schema key, not a hidden threshold on `nFieldDOF`.

**`FieldSolve = exact` warns on every run that selects it.** The message is
emitted once, at configure time rather than per solve, at `LOG_LEVEL::WARNING`,
and names the concrete cost: `nFieldDOF` full transport solves per Jacobian
solve, so the linear algebra is `nFieldDOF` times the iterative path's. It says
in terms that the exact solve is a verification tool and is not intended for
production. Tests that select it deliberately will emit it; `CapturedOutput`
already exists in the suite, so that is not noise.

## The adjoint

`initializeMatricesForAdjointSolve` assembles the same local blocks as
`updateMatricesForJacSolve` and stores `M.transpose()`, so the two must be kept
in step block for block. The coupled adjoint therefore needs `A1^T`, `A2^T` and
`B^T`, the field model must supply a transpose apply *and* solve rather than
only a forward one, and the block sweep reverses order.

**The "inexact is fine" licence does not carry over, and this is the design's
sharpest asymmetry.** An under-converged forward Jacobian costs Newton
iterations; an under-converged adjoint returns a wrong gradient with a perfectly
good `G`. So the adjoint path either uses the exact Schur or iterates to a
tolerance with a convergence check that **fails loudly**. It may not silently
return its last iterate. This is the same class of defect as the missing
`dSigma/dPhi` block, which cost nothing visible until `test_adjoint_aux.py` was
written.

## Two existing traps this walks into

**The differential-without-a-time-derivative misdeclaration.** The field block's
flags go into the `isDifferential` vector at `Solver.cpp:186` alongside
`isScalarDifferential`. A field DOF declared differential whose residual carries
no `d/dt` is a row every unknown of which `IDA_YA_YDP_INIT` holds fixed, so no
Newton direction touches it and the backtracking loop runs to exhaustion —
`IDA_LINESEARCH_FAIL (-13)`, a message about the linesearch for a defect in the
declaration. That is what kept `python-physics/mirror-plasma`'s voltage
controller from ever starting. `FieldSpec::validate` refuses it at construction.

**Per-run state in `initialiseMatrices`.** `initialize` skips
`initialiseMatrices` when `initialised` is already set, so anything the field
model computes once must be genuinely run-independent or refreshed per run. This
is the `RF_cellwise` trap, which made a second integration on one solver solve
its initial `dydt` out of the previous run's final-time boundary values. Covered
by extending `a_second_integration_on_one_solver_matches_a_fresh_one`, which is
at exactly zero tolerance and must stay there.

## The manufactured client

Two field models, both in `Tests/UnitTests`, neither registered for production.

**`ManufacturedField`, `nFieldDOF = 1`, algebraic.** Constraint
`R = psi - Int u dx`, one geometry slot `g(x; psi) = 1 + psi c(x)`, and a flux
`sigma_hat = g kappa q` — the metric scaling the diffusivity, which is the
physically right shape and gives a nonzero `dSigma_dGeometry`. With the
harness's shared `u = sin(pi x)(1+t)`, `psi_exact(t) = (2/pi)(1+t)` on `[0,1]`
and the source follows in closed form. The coupling is genuinely two-way: `A2`
is dense across the `u` DOFs and `A1` is nonzero.

**`ManufacturedFieldVector`, `nFieldDOF = n`.** `L psi = f(state)` with `L` a
fixed SPD tridiagonal — a stand-in for a 1-D elliptic operator, with its own
Thomas solve, so `B` is not a scalar. `dGeometry/dpsi` is dense, because
geometry at `x` interpolates all of `psi`. This is what exercises the block
solve rather than a degenerate case of it.

**One trap to design around**, recorded from the aux order studies: a
compensating term written against the *discrete* state can be an exact row
operation. `residual` evaluates the hooks on the same states at the same
abscissae and pushes them through the same `projectOntoTestSpace`, so a
compensation of that shape cancels identically and the study silently measures
the uncoupled problem. The field constraint is therefore compensated against
`u_exact(x,t)`, never against the discrete state.

## Testing

**Three failure classes, three tests, and no test covers another's class.**

| Wrong where | Symptom | Only caught by |
|---|---|---|
| Residual (`A1`/`A2`/`B` sign in the equations) | Converges at the right rate to the wrong function | MMS against the closed form |
| Jacobian block | Slow Newton, correct answer | `SolveJacTests` — FD the coupled residual, require `J dy = g` |
| Adjoint transpose | Wrong gradient, perfect `G` | Gradient vs finite differences, coupling armed |

Also:

* Exact and iterative field solves agree on both manufactured clients.
* `FieldSpec::validate` refuses a differential DOF whose residual has no `d/dt`.
* Bit-for-bit solver reuse with a field model attached, at zero tolerance.
* `Superconvergent = true` with coupling armed. Geometry is a function of
  `(psi, x)` and star nodes are just more points, so it should work through
  `ComputePhysics`'s `states.size()` loop — but it gets an order study, and
  throws rather than guessing if the rate does not hold.
* **The zero-coupling invariant**, which is the strongest guard available. With
  no field model configured, output must be identical to today *bit for bit* —
  verified the way `zeroFlux` was, by building the previous binary and diffing
  the netCDF byte for byte, not by the regression suite's 1e-2 comparison. Every
  config in the tree exercises that path.

**Method note** from the existing order studies: read **local** orders, not the
least-squares slope. A fit averages a changing rate away, which is how the
nonlinear-flux superconvergence breakdown stayed invisible to `n <= 32`.

## Configuration, failure and output

**Configuration** is declared in `ConfigSchema.cpp`, the single point, so both
surfaces get it at once and `--list-options` prints it: `FieldModel` (name,
`Category::ProblemSelection`), `FieldSolve` (`exact | iterative`, default
`iterative`), and the iteration's tolerance and sweep cap. Absent `FieldModel`
means no coupling. Registration mirrors `PhysicsCases::map` — static-init side
effects, duplicate name throws, unknown name throws with the list of what *is*
registered — and inherits the same trap, that a missing entry is a link-line
problem with no compile error.

**Failure must be recoverable, not exceptional.** A real provider will fail to
evaluate for some states — no x-point, a boundary that has left the domain.
`residual` already returns IDA's convention (0 success, positive recoverable,
negative fatal), so the field model gets a recoverable channel and IDA cuts `h`
and retries. Throwing would abort a run that a shorter step would have survived.
When a coupled run does stall, `SUNLOGGER_INFO_FILENAME` gives IDA's per-attempt
record for no code change; write it to a file rather than `stderr` if a test is
running, because `CapturedOutput` redirects the standard descriptors.

**Output and restart.** `psi` and the geometry fields get their own netCDF
group, and the restart format grows by the field block. Restarting is already
recorded as fragile at tight tolerances with `nAux > 0`; `psi` adds to that, so
the round-trip regression case runs at the tightest tolerance that completes and
records which that is.

**Python.** `State` gaining `geom` means `PyState` exposes it and
`GlobalState`'s dict gains a `"Geometry"` key — and that caster **transposes in
both directions**, so a round-trip test cannot detect a missing transpose; check
the orientation from inside a batched call. Python cases can read geometry and
supply the derivative hooks, and an absent hook is a zero block, which is the
correct meaning. A Python-*implemented* field model is out of scope.

## Deliberately not in v1

Each is a separate spec, and none is blocked by this one:

* **2-D Grad–Shafranov** as a field model. This design is what it would plug
  into; building it is a much larger piece of work, and the piecewise-smooth
  residual finding above says a coupled Newton against it needs its own
  treatment.
* **DESC as a provider.** Needs a Python-implemented field model and packages CI
  cannot install.
* **A current-diffusion physics case** — `psi` as an ordinary transport
  variable, TORAX-style, on frozen geometry. This needs *none* of this
  machinery and is worth doing on its own merits; it is the cheapest useful
  self-consistent field in the tree and should not wait behind this.
* **Remapping between equilibria.** Not needed while the provider declares a
  fixed label, which is JETTO's own resolution of the question.
* **Free-boundary shape calculus.**
* **Woodbury-exact solves for large field blocks.**
