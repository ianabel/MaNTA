# Self-consistent magnetic fields — design notes

**Branch:** `features/self-consistent-b-fields` (44 commits, `4d0afa3..35f5ffd`)
**Written:** 2026-08-16, after implementation and review, against the tree as built.

## What this document is

Four other documents describe this feature, and none of them covers what is
here:

| Document | What it is |
|---|---|
| `docs/superpowers/specs/2026-08-15-self-consistent-b-fields-design.md` | the design **as agreed, before implementation**. Parts of it are now wrong; see *Departures* below. |
| `docs/superpowers/plans/2026-08-15-self-consistent-b-fields.md` | the task-by-task construction order. Of historical interest only once the branch lands. |
| `docs/field_coupling.rst` | the **interface reference**: what a field-model author writes, what a physics-case author reads, the maths, the config keys. The document to send someone who wants to *use* this. |
| `CLAUDE.md`, "Self-consistent magnetic fields" | the **traps**: what breaks silently if you edit the coupling, and which test catches it. |

These notes are the fifth thing: *why the design has the shape it does*, where
the built system departed from the spec and why, what was measured, and what is
still open. They also carry the deferred-findings triage out of
`.superpowers/sdd/`, which is git-ignored scratch and will not survive a
`git clean -fdx`.

---

## The shape, in one page

A `FieldModel` contributes:

* `nFieldDOF` unknowns `psi`, appended to the solution vector **after** the
  global scalars, with one residual row each;
* `nGeometry` derived **geometry slots** `g_s(psi, x, t)`, evaluated at the
  physics nodes and cached per residual, in the same standing as `sigmaHat`;
* its own Jacobian block `B`, behind four entry points — `applyB`, `solveB` and
  their transposes.

The coupling is two-way and each direction goes through exactly one channel:

```
field rows  ---- read the transport solution ---->  via GlobalState + weights
transport   <--- read geometry, never psi -------   via State::geom(s)
```

Nothing else crosses. A physics case never sees `psi`; a field model never sees
a basis coefficient. That is what makes the feature addable without touching a
single existing physics case.

Three structural choices carry most of the weight:

**Geometry is derived, not an unknown.** It could have been `nGeometry` extra
DOFs constrained to equal the map's output. Making it a cached function of
`(psi, x)` instead keeps the DOF count down, keeps the geometry out of IDA's
error test, and — the reason that matters — means `Superconvergent = true`
needs no special case: star nodes are just more `x`.

**`psi` goes last.** Everything before it keeps its index, which is what makes
the zero-coupling invariant a byte comparison rather than an argument.

**The model owns `B`.** The default is a dense `PartialPivLU`, right for a small
block. A real Grad–Shafranov solver overrides all four entry points and plugs in
its own sparse machinery without the solver knowing. `applyBTranspose` and
`solveBTranspose` are declared beside the forward pair deliberately: a model
that supplied only one direction would otherwise be silently accommodated
forwards and produce a wrong gradient backwards.

---

## Departures from the spec

Five, and the first is the substantial one.

### 1. Plain block Gauss–Seidel does not converge

The spec's production path was `FEATURES.md`'s "assume the coupling is weak, and
iterate": sweep `A dx = r1 − A1 dpsi^k`, then `B dpsi^{k+1} = r2 − A2 dx`. The
fixed point is `dpsi^{k+1} = c + M dpsi^k` with `M = B⁻¹A₂A⁻¹A₁`, and on the
manufactured client

```
rho(M) = 1.611  at cj = 0
rho(M) = 1.571  at cj = 1e8
```

`rho` is a property of the **coupling**, not of the time step — which is why it
barely moves between those two, and why "take smaller steps" is not a fix.
`RichGeometricDiffusion` could not be integrated on the iterative path at *any*
step size.

SOR does not rescue it. The damped iteration matrix is `(1−ω)I + ωM`, with
eigenvalues `1 − ω + ωλ`; no `ω > 0` brings a real `λ > 1` inside the unit
circle. Damping a divergent real eigenvalue is not something a relaxation
parameter can do.

**Irons–Tuck (vector Aitken Δ²) instead:**

```
mu = Δ_k · (Δ_k − Δ_{k−1}) / ‖Δ_k − Δ_{k−1}‖²
p* = G(p_k) − mu Δ_k
```

Exact in one step for a scalar affine map at *any* spectrum, divergent ones
included; a rank-one secant for vectors. After it,
`RichGeometricDiffusion` integrates on `FieldSolve = iterative` with **0
fallbacks over 1965 Jacobian solves**, gradient to 1.09e-10.

The rank-one limit is real and recorded: a coupling with two comparable
eigenvalues outside the unit circle still falls back. The depth-`m`
generalisation (Anderson / GMRES on the Schur complement) is the right answer
for the large `nField` this path exists to serve, and is separate work.

### 2. The adjoint escalates instead of failing loudly

The spec was explicit that the adjoint "either uses the exact Schur or iterates
to a tolerance with a convergence check that **fails loudly**. It may not
silently return its last iterate." As built, exhausting
`FieldSolveMaxAdjointSweeps` falls back to `solveCoupledAdjointExact` and warns
once.

This is strictly stronger, not weaker. The spec's requirement was *never
silently wrong*; escalation meets it and additionally never fails. The caller
gets a correct gradient where before it got an exception. The sweep's last
iterate is discarded rather than blended in — `solveCoupledAdjointExact`
overwrites `adjoint_squ`, `adjoint_lambdas` *and* `adjoint_field`.

`FieldSolveMaxAdjointSweeps` defaults to 100 against the forward cap's 20,
because the adjoint always runs at `cj = 0`, where the coupling is stiffest.

### 3. Escalation changes what the tests can mean

Once the iterative path escalates, it **cannot be wrong, only slow**. That is
what makes it defensible as a default. It also means "iterative agrees with
exact" is a test that passes with the accelerator deleted — confirmed: deleting
Irons–Tuck failed six tests, and that comparison was not among them. See
*Vacuity* below; this is the second of the three shapes.

### 4. The cost ranking is the reverse of what the spec assumed

The spec called the exact Schur solve "a verification tool rather than a
production path". Measured, at the sizes in this tree, it is the **cheaper** of
the two. One sweep costs one transport solve; the exact solve costs `nField + 1`.
So

```
iterative wins  iff  #sweeps < nField + 1
```

and no fixture here is on the winning side: ~1.5× worse at `nField = 1`,
2.2–6.3× at `nField = 5`, and a run that falls back pays `FieldSolveMaxSweeps`
*and* the exact solve on top.

The default was left as `iterative` anyway, because it can only cost time and
the design target is `N_magnetics ≫ N_HDG`, where `nField + 1` transport solves
plus an `O(nField³)` dense Schur solve is hopeless. But the honest description
is *a bet on a large field block*, not a free improvement, and both the
`INFO` line the solver logs and `docs/field_coupling.rst` now say so in those
terms. The decision about whether to change the default is parked; see below.

### 5. No field model ships registered

Spec decision 4 was "the first client is a manufactured equilibrium in C++".
What exists is two unregistered fixtures under `Tests/UnitTests`. Consequences,
all recorded in `Tests/README.md` and `CLAUDE.md`'s Known limitations: nothing
exercises the coupled path through a `.conf` file, there is no coupled
regression case, and the config plumbing and netCDF group have unit-test cover
only.

---

## What was measured

| Quantity | Value | What it pins |
|---|---|---|
| Zero-coupling invariant | 14/14 `.nc` **byte identical** to a `main` build; 14/14 `.restart.nc` differ by exactly `int nField = 0` | that an uncoupled run is unchanged. The regression suite's 5e-3 is far too loose to see this. |
| Coupled order, `u_h` | k=1: 1.90/1.96/1.98 · k=2: 2.95/2.98/2.99 · k=3: 3.96/3.99/4.00 · 5-DOF k=2: 2.93/2.98/2.99 | the coupled **residual**. A 5% error in the field row collapses k=2 to 0.264/0.004/−0.000. |
| Superconvergent, `u*` | k=1: off 2.206/2.091/2.048, on 3.117/3.048/3.014 · k=3: off 4.987/4.893/4.706, on 5.265/4.984/4.942 | that the flag still works under coupling. k=1 is the only configuration where flag-on is not also satisfied flag-off. |
| Two-mode agreement | orders agree to 1.8e-9 / 3.2e-8 / 1.3e-7; **fallbacks zero at every refinement** | that the study measured the *iterative* mode. Without the fallback assertion it would silently become a study of the exact mode. |
| Sweeps per solve, real integration | 2.5–3.6, cap never reached | that Newton's right-hand sides are far more benign than random vectors… |
| Sweeps per solve, random RHS at `nField = 5` | 13–38 over six seeded directions; three of six exhaust the default cap of 20 | …and that neither number says anything about the cap on its own. |
| `rho(M)` | 1.611 (cj = 0), 1.571 (cj = 1e8) | that the divergence is the coupling's, not the step's. |
| Cost ratio, iterative ÷ exact | ~1.5× at `nField = 1`, 2.2–6.3× at `nField = 5` | the crossover above. |
| Irons–Tuck invariant probes | forward 3.30e-16 vs 2.25e-11; backward 3.86e-16 vs 7.32e-9 | that the accelerator is actually running — see below. |
| Coupled adjoint gradient | 1.09e-10 against finite differences | the transposes. |

---

## Which test catches which failure class

The three-way split is the point, and it is not redundancy — each class is
invisible to the other two tests.

| Failure | Only test that sees it | What it looks like otherwise |
|---|---|---|
| Wrong `A1` or `A2` (coupling Jacobian) | `FieldJacobianTests.cpp` — finite-difference the residual, require `J dy = g`, `FieldSolve = exact` | extra Newton iterations and **nothing else**. The coupled Jacobian is never assembled. |
| Wrong coupled *residual* — a sign, a factor | `MMSFieldTests.cpp` — closed-form comparison | converges at the right rate to the wrong function. No Jacobian check sees it. |
| Wrong *transpose* of either | `FieldAdjointTests.cpp` | a silently wrong gradient beside a perfectly good `G`. This is the `dSigma/dPhi` asymmetry again. |
| The sweep silently not running | `checkNoFallbacks` in `MMSFieldTests.cpp` | every order **bit-for-bit unchanged** and the two-mode gap *improves* to 0.00e+00. Demonstrated by capping sweeps at 1: 382 fallbacks in 382 solves, and only `checkNoFallbacks` failed. |
| `psi` not copied on restart | `psi_round_trips_through_a_restart`, oracle read from the raw netCDF array | with `getSolution()` as the oracle, both sides go to zero and agree. |
| A model caching across runs | `a_coupled_solver_reused_matches_a_fresh_one_bit_for_bit` | a second run that completes, looks plausible, and is wrong by 3.2e-4. |

`A1^T` and `A2^T` are **materialised** in
`initializeMatricesForAdjointSolve` rather than transposed at each use,
precisely so a test can zero one and require the gradient to go wrong.

---

## Vacuity: three ways a test here goes quiet

Worth stating as a group, because all three bit during this branch and all three
generalise past it. Each was demonstrated by mutation, not argued.

**1. The Jacobian is never assembled.** So a Jacobian error costs Newton speed
and produces the right answer. This is a pre-existing property of MaNTA; the
coupling inherits it, which is why `FieldJacobianTests.cpp` exists at all.

**2. A fallback path indistinguishable from success.** Once
`solveCoupledJacIterative` escalates, "iterative agrees with exact" agrees
*perfectly* — the two are the same code. A test written against the observable
answer stops testing the thing it names. The fix is to assert on the
*mechanism* (`fallbacks == 0`), not on the answer.

**3. An oracle sharing a code path with the code under test.** The restart
test's `psiBefore` came from `getSolution()`, filled by `setJacEvalY` via
`DGSoln::copy` — the same function under test. Deleting `psi_ = other.psi_` left
the test **passing**, both sides zero. The oracle is now `Y[nDOF_file − 1]` read
straight off disk, and the same deletion fails three ways.

A fourth, from Task 13's review, is worth keeping beside them: **the mutations
you design to prove a test discriminates can themselves be no-ops.** Two of the
three planned probes caught nothing, because extrapolation at the stopping point
is second order in an increment already at tolerance — and for `nField == 1` a
literal no-op. Replaced with probes on row two of the transposed system, which
do discriminate (the 3.30e-16 vs 2.25e-11 in the table above).

---

## Open questions and parked decisions

Recorded in `TODO`; nothing here is implemented.

**The `FieldSolve` default.** The crossover in *Departure 4*. Two candidates:
default `FieldSolveMaxSweeps` to `nField + 1` (stateless, bounds the worst case
at 2× exact), or a running-mean latch that switches a run to exact once the
observed sweep count exceeds the break-even (better asymptotically, but adds
per-run state). **Awaiting a decision — do not implement either unprompted.**

**Anderson / GMRES on the Schur complement.** The depth-`m` generalisation of
Irons–Tuck, for the large-`nField` regime this path is a bet on.

**`FieldResidual` receives no `states_dot`.** So a field row cannot depend on the
transport time derivatives, and `FieldResidualPrime`'s `dRdot` cannot be filled
— leaving it zero is *correct* today. The slot exists because the coupling
assembly already weights it by `alpha`, so the day the value hook gains `ydot`
the derivative is right rather than one term short. The hazard is a model author
finding it unfillable and putting `d(row)/d(psi')` there instead: that belongs
in `dRddpsidt`, and written into `dRdot` it lands in the `A2` row at the wrong
DOFs with nothing to say so. `FieldModel.hpp` warns about exactly this.

**Two structurally missing adjoint terms**, both identically zero today. No
`dg/dgeometry` hook exists, so the field row of the adjoint right-hand side
`G_field` is zero; and `FieldModel` has no notion of an adjoint parameter, so
there is no `d(field residual)/dp`. Adding either means adding it to both the
model interface and `computeAdjointGradients`.

**`nScalars > 0` with a field model is refused**, at `setFieldModel`. The
non-superconvergent `dSources_dScalars` branch builds its `State` from
`DGSoln::evalOnNode`, which has no geometry rows — so a case reading geometry
there would work with `Superconvergent = true` and read out of bounds with it
off. Lifting this means giving `evalOnNode` geometry.

**Out of scope, live on `main`:** two `delta_t` misreadings in `Solver.cpp`.
`IDACalcIC` is passed the output *cadence* where SUNDIALS wants the first output
*time*, and `IDASetInitStep(IDA_mem, dt)` on a restart makes the cadence the
first step. They pull against each other, so no cadence avoids both, and
restarting is a documented feature. Confirmed byte-identical to `main`; deserves
its own branch. Also `manta.pc` propagates no include directories for Eigen,
SUNDIALS or toml11, which affects the pre-existing physics-plugin route
identically.

---

## Deferred minor findings

Carried out of the git-ignored SDD workspace. None blocks merge; the full text
of each is in the review record, which does not survive.

**Hygiene** — unused `<memory>` in `FieldModel.hpp` and `<algorithm>` in
`FieldModelSpec.hpp`; `FieldModel.hpp` relies on `FieldModelSpec.hpp` to pull in
`<vector>`; `evaluateGeometry`'s parameter is `t` in the header and `t_eval` in
the definition; `_test_dSigmaFn_dGeometry` is the first `_test_`-prefixed
binding in `Python.cpp` and sets a convention with no precedent; code comments
round the cost ratio to "2–6×" where the measurement is 2.2–6.3×.

**Coverage gaps** — every field-Jacobian fixture is `nVars = 1`, so the `nVars`
factors in the coupling blocks are untested; no coupled fixture has
`nScalars > 0`, so the sweep never exercises `solveTransportJac`'s Woodbury
branch; `CoupledResidualTests.cpp`'s `ManufacturedCoupledDiffusion` twin has no
`dSigmaFn_dGeometry`, so its `A1` is zero — fine for the exact-path run it
drives, silently vacuous if anyone adds an `A1` check against it;
`relativeDifference` normalises by the whole-vector norm, diluting a
field-block-only error ~7.5×; `InitialFieldValue` is never called by name in any
test; neither of the two new refusals has a test; `manufacturedU` duplicates
`MMSHarness.hpp`'s `exactSolution`.

**Resource handling on throw paths — one ruling, not three.**
`solveCoupledJacExact` leaks `nField + 2` `N_Vector`s on a throw;
`solveCoupledJacIterative`'s `work` leaks if `solveTransportJac` throws;
`CoupledFixture`'s defaulted move constructor is a latent double free relying on
NRVO. Either the codebase accepts leaks on a throwing path that ends the run
anyway, or it does not — these three should be decided together and follow the
house answer.

**Behaviour, noted rather than changed** — `allocateJacobianStorage` now
value-initialises (real, benign); the `dG/dt` refusal guards the caller rather
than `assembleDenseJacobian` itself; the `nScalars` refusal fires only after the
whole integration is paid for; `WriteRestartFile` takes its length from member
`y` while receiving `Y` as a parameter; a throw from `finaliseDiagnostics`
destroys a held `adjointFailure`, so the caller sees the output error rather
than the adjoint refusal (the ERROR log line survives); with a refused adjoint
`setJacEvalY` now runs where before it did not, so `yJac` holds the final
solution rather than the last Jacobian-eval state — an improvement, and it
matches the success path, but it was unremarked at the time.
