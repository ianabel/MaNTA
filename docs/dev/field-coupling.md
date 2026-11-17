# Self-consistent magnetic fields

Field models, equilibrium models, the coupling tiers and the coupled solves. `docs/field_coupling.rst` is the user-facing interface document.

A `FieldModel` (`FieldModel.hpp`) contributes `nFieldDOF` unknowns `psi`, one
residual row each, and `nGeometry` derived *geometry slots* `g_s(psi, x, t)`.
It follows the physics-case pattern throughout — a validated spec, a
process-global registry with the same two throws, its own toml table, selected
by name from the config. `docs/field_coupling.rst` is the interface document and
`--list-options` prints the five keys; what follows is what neither says.

**The geometry slots are the only channel into the transport physics**, and they
are not unknowns: they are a function of `(psi, x)` evaluated at the physics
nodes and cached per residual, in the same standing as `sigmaHat`. A case reads
them as `State::geom(s)`. Nothing else crosses in either direction.

**`FieldModelSpec` is not `FieldSpec`.** `SystemSpec.hpp` already defines a
global `struct FieldSpec`, the per-transport-variable descriptor bound to Python
as `manta.Field`. Reusing the name compiles every translation unit cleanly and
fails only at link time, as an ODR violation naming neither type. Do not shorten
it back.

**The whole thing is inert when unused, and that is checked by hand rather than
asserted.** `psi` goes *last* in the layout so nothing before it moves, and
every existing config has no field model. The check is to build `main`, run both
binaries over every `Tests/RegressionTests/*.conf` from the same directory and
`cmp` — netCDF files carry no timestamp of their own, so a byte comparison is
legitimate, and the regression suite's 5e-3 is far too loose to see a change of
this kind. Last measured: all 14 `.nc` byte identical; all 14 `.restart.nc`
identical apart from the deliberately added `int nField = 0`. Re-run it after
anything that touches the DOF layout, the residual or the Jacobian solve.

**Which test catches which failure class — the three-way split is the point.**

* A wrong `A1` or `A2` (the coupling Jacobian blocks) costs Newton iterations
  and *nothing else*, because the coupled Jacobian is never assembled. Only
  `FieldJacobianTests.cpp` sees it, by finite-differencing the residual and
  requiring `J dy = g` with `FieldSolve = exact`.
* A wrong coupled *residual* — a sign, a factor — converges at the right rate to
  the wrong function. No Jacobian check sees it; only the closed-form comparison
  in `MMSFieldTests.cpp` does. A 5% error in the field row, not even a sign
  error, drops the `k = 2` orders to 0.26, 0.00, -0.00.
* A wrong *transpose* of either is a silently wrong gradient beside a perfectly
  good `G`, and only `FieldAdjointTests.cpp` sees that. **This is the adjoint
  asymmetry, and it is the same one `dSigma/dPhi` demonstrated**:
  `assembleAdjointOperator` stores `A1^T` and `A2^T` beside `M^T`, so
  a coupling added to `updateMatricesForJacSolve` and not to it degrades a
  forward run's convergence and corrupts a gradient. They are *materialised*
  rather than transposed at each use precisely so a test can zero one and
  require the gradient to go wrong. **The field blocks are the one thing the
  adjoint and forward operators share**: the adjoint writes its `alpha = 0`
  blocks over the forward ones and sets `fieldBlocksHoldAdjoint`, and
  `solveJacEq` puts the forward ones back from `jacobianDerivatives` before its
  next use. Drop that restore and an adjoint solve between two slices of a
  steady solve changes the forward solve;
  `an_adjoint_solve_leaves_the_forward_field_blocks_alone` is what notices.

Two more properties nothing else pins. `resetForRun()` is called from the
**unconditional** part of `initialize()` — not from `initialiseMatrices`, which
`initialize` skips when already initialised: that is the `RF_cellwise` trap, and
a model caching an equilibrium across runs is exactly the shape that falls into
it. `a_coupled_solver_reused_matches_a_fresh_one_bit_for_bit` is the only thing
standing between that and a second run that completes, looks plausible and is
wrong; breaking either the fixture's `resetForRun` or `initialize`'s call to it
moves the answer by 3.2e-4. And `psi_round_trips_through_a_restart` uses a
**differential** field DOF deliberately: `IDA_YA_YDP_INIT` solves for algebraic
*values*, so an algebraic `psi` would be recomputed from the restored transport
state and the case would pass without ever reading `psi` off disk.

**A field can also be a black box: `EquilibriumModel`, `FieldCoupling =
"consistent"`** (tier 0). It adds no unknowns: the
solver holds the last `Equilibrium` as a cache keyed byte for byte on the
solution vector's cell prefix and the time (`equilibriumAt`), solves a new one
only when that changes, warm-starts it from the last, and fills `State::geom`
from it in `evaluateGeometry`. `FieldModelEquilibrium` runs any `FieldModel`
that way -- Newton on its rows alone -- and is what `geometrySourceFromConfig`
attaches for `"consistent"`. Four things are load-bearing. **The cache is dropped in
`initialize()`'s unconditional part**, the `RF_cellwise` trap again: deleting
that moves a reused theta-pinch run by 1.6e-8, and
`a_reused_tier_zero_theta_pinch_solver_matches_a_fresh_one_bit_for_bit` is the
guard (the manufactured field's linear row lands on the same bits from any warm
start, so `equilibrium_tests` alone cannot see it). **`IDACalcIC` runs with its
line search off** at tier 0: the Jacobian carries no coupling, so its Newton
direction need not descend `||F||`, and on the theta pinch at T = 0.9 the line
search fails where full steps converge. **No adjoint** -- refused in
`initialize()` and, for the on-demand paths, in `assembleAdjointOperator`. And
**one source of geometry**: `setFieldModel` and `setEquilibriumModel` share
`checkGeometrySource`, which refuses the second.

**Which of the two runs is `FieldCoupling = "auto"`'s call (`chooseFieldCoupling`,
`SolverConfig.cpp`), from `SystemSpec::geometryDerivatives`**: `None` (the
default) cannot be coupled -- `setFieldModel` refuses it, since the hooks'
no-op defaults would build A1 out of zeros -- and runs as a black box;
`Analytic` couples; `Differenced` couples transients and runs steady solves as
a black box, because the solver then differences `ComputePhysics` once per
geometry slot plus once at the state per Jacobian build, and a steady solve
builds a Jacobian nearly every iteration, so coupling a differenced steady
solve costs more than the black box saves; `docs/field_coupling.rst` has the
measured table. A field with dynamics or an
adjoint forces the coupled mode, and is refused for a `None` case. Two things
are load-bearing. **The `Analytic` path is byte identical to the code before
`geometryResponse`**, which only hoisted `dGeometry/dpsi` out of the row loop:
re-run the theta-pinch `cmp` (the regression config, a `Superconvergent` and a
`Newton` variant) after touching the A1 assembly. **The unperturbed
`ComputePhysics` goes last** in `differencedGeometryResponse`, because
`ComputePhysics` also writes the case's source cache, which the output reads;
`a_build_spends_what_its_geometry_derivatives_declare` pins that and the call
counts.

**Tier 1 is `FieldCoupling = "reduced"`: the equilibrium coupled through its
response** (`EquilibriumModel::applyResponse`, `solveReducedJac`). The
Jacobian is `A + A_g G' P` with no unknowns added; `A_g` is A1 without the
chain through dGeometry/dpsi, built per Jacobian build into `reducedBlocks`
beside the equilibrium at `yJac`, and the response defaults to a forward
difference of `Solve` (`FieldModelEquilibrium` supplies the exact one, tier 2
with psi eliminated). The coupled solves are written once for both tiers over
a *coupling variable* (`CouplingOperator`: dpsi at tier 2, the geometry
increment at the physics nodes at tier 1): `couplingSweep`, `couplingKrylov`
(SUNDIALS SPGMR on c itself, `(I + L A^-1 C) c = c0 + L A^-1 r`, scaled by c's
error weights, so its residual is the change one more sweep would make to c)
and `couplingExact`. At tier 1 the sweep re-solves dx against the accepted c
before returning -- the transport increment is the whole Newton update there,
and handing back the one solved against the previous c cost IDA 199 steps
against the exact solve's 163 -- and GMRES weights c by the AlgebraicMax rule
applied to the geometry (`ReducedBlocks::geometryWeights`), since IDA has no
weights for it.
Three things are load-bearing. **The tier-2 sweep is byte identical across
that split**: `fieldCouplingOperator`'s `affine` is the old arithmetic
operation for operation; re-run the theta-pinch `cmp` after touching it.
**`ReducedBlocks::states` is a `shared_ptr<GlobalState const>`**, never assigned,
because `GlobalState`'s copy *assignment* copies `Aux` only into an object
already sized for it -- a default-constructed one dropped the pressure, and the
model then read a state with no aux rows -- and the blocks are copied into every
`PhysicsDerivatives` record and every adjoint. And **`reducedBlocks` and
`adjointReducedBlocks` are cleared in `initialize()`'s unconditional part**, the
`RF_cellwise` trap again: they hold an equilibrium of the last run's.

**The tier-1 adjoint** solves `(A^T + P^T G'^T A_g^T) z = G_y + P^T G'^T G_g`
through the transposed coupling code (`AdjointCouplingOperator`:
`adjointCouplingSweep`, `adjointCouplingKrylov`, `adjointCouplingExact`), in
the coupling variable's adjoint `zeta = G_g - A_g^T z`; tier 2's `krylov`
adjoint is the same GMRES with `zeta = B^T z_psi`. It needs
`EquilibriumModel::applyResponseTranspose`, whose default assembles G' column
by column through `applyResponse` -- (3 nVars + nAux) N(k+1) columns, each an
equilibrium solve when differenced -- and caches it on the `Equilibrium`
itself, so it cannot outlive the equilibrium it differentiates.
`FieldModelEquilibrium` transposes exactly. The same discipline as A1^T and
A2^T: **`Ag_transpose_cellwise` is materialised** in `assembleAdjointOperator`
so a test can zero it, and **the record carries the blocks**
(`PhysicsDerivatives::reduced`, from `buildReducedBlocks`), so an adjoint on
demand makes no physics call. `auto` still couples a field model for an
adjoint: the tier-2 adjoint needs no transposed response.

**A native `EquilibriumModel` has a registry of its own** (`EquilibriumModels`,
`EquilibriumModel = "<name>"`), the field models' exactly, and the two keys are
refused together. It runs at `consistent` or `reduced` only;
`chooseEquilibriumCoupling` is its own function beside `chooseFieldCoupling`,
and takes the model's response as an input, since a differenced one multiplies
equilibrium solves on a steady solve. `ThetaPinchPressureBalance` is the
registered one: the theta pinch's balance in closed form on the transport's own
`p_h`. **Its `dV'/dx` reads `p_x` from a fit to cell means, and that is
load-bearing**: read off `p_h` the manufactured study goes at k, and fitted to
`p_h`'s nodal values it decays to order 1 at k = 1, because the fit
differentiates the shape the interpolation error repeats in every cell.
`the_native_balance_converges_slot_by_slot` is the guard.

**The theta pinch is the registered model** (`PhysicsCases/ThetaPinchEquilibrium`,
read by `PhysicsCases/ThetaPinchDiffusion`; `docs/field_coupling.rst` has the
physics). Three things about it are load-bearing. Its pressure crosses as the
transport case's aux variable, so its `A2` lives *only* in aux columns and its
`A1` runs through four geometry slots — the combination the manufactured
fixtures never reach. Its spline's knots are the transport mesh's faces
(`IntervalsPerCell`, default 1), so `nField` is independent of `k` -- which is
what lets a degree-changing restart carry `psi` verbatim
(`setInitialConditions`' projection path) -- and not of the mesh, so a restart
onto another mesh is refused by its `nField`. The knots follow the mesh because
`p_h` jumps at every face and a spline finer than the mesh resolves the jumps
into spikes in `dV'/dx`, which the flux reads: four intervals per cell costs an
order at `k = 1`, and more stalls IDA under either `FieldSolve`. The model's
header has the measurements, and refuses an unknown key in its table rather
than running on a default. And
`ThetaPinchDiffusion` refuses to run without a field model unless
`VacuumGeometry = true`, so a frozen-geometry run is a choice rather than a
forgotten key.

**Irons-Tuck accelerates only along aligned increments**, and that guard is
the difference between converging and diverging on the theta pinch's adjoint.
The step extrapolates along the last increment as though the sweep had settled
into its dominant eigendirection. A non-normal iteration matrix amplifies and
turns an increment before that — tenfold, at `cos = -0.27`, on the theta pinch
at `cj = 0`, with a spectral radius of 0.14 — and extrapolating from that pair
took the sweep to a backward error of 1e68, every gradient silently rescued by
the exact fallback. `ironsTuck` now requires `|cos| >= 0.9`; the divergent
fixtures it exists for give `|cos| > 0.99`. A fallback that always rescues is
also a fallback that hides: `getFieldSweepStats().adjointFellBack` is what
`the_objective_estimate_carries_the_field_share` asserts against.

**`FieldSolve = iterative` is a cost choice, never an accuracy one.** The block
Gauss-Seidel sweep escalates to the exact Schur solve when it exhausts its cap,
in *both* the forward and the adjoint directions, so it can be slower than
`exact` and can never be less accurate. The break-even is
`#sweeps < nField + 1` — one transport solve per sweep against `nField + 1` for
the exact solve, plus its dense `O(nField^3)` Schur factorisation — and which
side a problem is on is a property of the coupling. The manufactured fixtures
lose (2.0-2.8 sweeps per solve against an `nField + 1` of 2 to 6); the theta
pinch wins at every resolution measured, from 1.5x at `nField = 12` to 12x at
`nField = 132`, because its sweep contracts at 0.10-0.19 at every `cj` and
takes ~3.7 sweeps per solve whatever `nField` is. `docs/field_coupling.rst` has
the table. Note that isolated Jacobian solves with *random* right-hand sides
need 10-23 sweeps at `nField = 5`, while a real integration averages 2.0-2.8
and has never hit the cap; Newton's right-hand sides are far more benign than
random vectors, and neither number says anything about the cap on its own.

**The sweep has to stop in IDA's own terms, or "a cost choice" becomes a 100x
cost, and it took three tests to learn which terms.** The forward sweep stops once one sweep
changes the *whole* increment, transport and field, by less than
`FieldSolveTolerance` (1e-4) in the run's error-weighted RMS norm -- absolute,
in the norm IDA's Newton test is in -- or by less than a millionth of the
increment, whichever is larger. Each alternative was measured failing, and
the failure is never a wrong answer, only IDA stalling at order one with a tiny
step, which looks like a hard problem rather than a solver defect:

* a relative 2-norm test on `psi` alone (the original) leaves an error in
  `sigma` and `q` near the axis, where values sit near zero and the weights are
  ~1/atol, that is O(1) in IDA's norm: 20000+ steps against the exact solve's
  213 on the manufactured theta pinch, uncured by tightening it to 1e-11;
* an absolute bound of 1e-2 is too loose for IDA's Newton rate estimate,
  `||del_m|| / ||del_0||`: 658 steps against 166;
* a bound *relative to the increment* at 1e-3 lets a first Newton update of
  size ~2 carry 2e-3: 29000 steps against 214 at `k = 4`;
* and the absolute bound alone cannot be met on `IDACalcIC`'s first step,
  whose increment at `cj ~ 1e9` is ~4e8 units: every run escalated to the exact
  solve exactly once, which is what the millionth is for -- it governs only
  above 100 units, where a 1e-6 relative error moves Newton's rate by nothing.

`the_iterative_solve_does_not_stall_ida` (`ThetaPinchMMSTests.cpp`) and
`exact_and_iterative_field_solves_give_the_same_run` (`ThetaPinchRunTests.cpp`)
are the guards, and they have to be *runs*: a unit test with random right-hand
sides at a fixture state cannot see any of the three, because the residue only
concentrates where it hurts on Newton's own corrections. The adjoint sweep has
no error test to measure against and keeps a relative backward error, under its
own key, `FieldSolveAdjointTolerance`.
`FieldSolveMaxAdjointSweeps` defaults to 100 against `FieldSolveMaxSweeps`'s 20
because the adjoint always runs at `cj = 0`, where the coupling is stiffest.
`TODO` records two candidate latches for a run that falls back on every solve;
neither is implemented.

The refusals, each at the earliest point the combination is known:
a field model on a case whose `SystemSpec::readsGeometry` is false
(`setFieldModel` — the geometry slots are the field's only channel, so it
could not reach the solution; `PlainDiffusion` in `CoupledResidualTests.cpp`
declares it while reading nothing, on purpose, to hold the transport rows blind
for the layout tests, and `PlainDiffusion(false)` is what the refusal is tested
against); `nScalars > 0` with a field model (`setFieldModel` — the non-superconvergent
`dSources_dScalars` branch builds its `State` from `DGSoln::evalOnNode`, which
has no geometry rows, so a case reading geometry there would work with
`Superconvergent = true` and read out of bounds with it off); a field DOF
declared differential whose row carries no `d/dt` (`initialize`, naming the DOF,
because left to IDA it is `IDA_LINESEARCH_FAIL`); a restart whose file's
`nField` disagrees with the configured model's; and **four naming refusals that
exist because the spec's names are now netCDF names** — a name netCDF would
reject, and a DOF sharing a name with a geometry slot, both in
`FieldModelSpec::validate()`; a group name colliding with a transport variable,
an aux variable or one of `Grid`/`RestartData`/`x`/`t`/`nVariables`, in
`setFieldModel`, which is the earliest point that knows both. Left to netCDF
these are an `NcBadName` or `NcNameInUse` out of `ncGroup.cpp` at the *first
write*, naming netCDF's source and a line number and neither MaNTA nor the
spec. The DOF-versus-slot one is new with this layout: the two lists used to be
written nowhere near each other and now share one group, and `checkNames`
compares each list only with itself.

**The restart test's oracle is the raw netCDF array, not `getSolution()`.**
`yJac` is filled through `DGSoln::copy`, which is the function that carries the
field block — so a comparison rooted in `getSolution()` on both sides agrees at
zero when `psi_ = other.psi_` is deleted, and the case passes while `psi` is not
being copied at all. Measured: green under that deletion with the old oracle,
three failures with the new one. The same trap applies to any future check of a
quantity `copy` moves.
