# Sources that carry time derivatives

Date: 2026-09-21
Status: phase 1 implemented; phases 2 and 3 not started.

Where the implementation departed from this document, the text below has been
left as written and the departure recorded at the point it happens, rather than
edited away. Two of them: the superconvergent open question was **answered**
(interpolating `udot` onto the star nodes does not cap the `k+2` rate -- measured
3.09, 4.09, 5.01 at k = 1, 2, 3), and the call to force `IDACalcIC` on turned out
to be **unnecessary** and was not made.

## Why

The transport equations MaNTA exists to integrate, in the form Abel *et al.*
derive them (Rep. Prog. Phys. **76** (2013) 116201, `~/Papers/RoPP-final.pdf`),
do not have the shape MaNTA assumes. The heat transport equation is their (194),

```
(3/2)(1/V') d/dt|_psi [ V' <n_s>_psi T_s ] + (1/V') d/dpsi [ V' <q_s>_psi ]
    = P^visc + P^Ohm + P^comp + P^pot + P^turb + <C^(E)> + <S^(E)>
```

and the potential-exchange heating on its right-hand side is their (201),

```
P^pot_s = - < Z_s e phi_0 ( dn_s/dt + V_psi . grad n_s ) >_psi
          - < Z_s e n_s phi_0 div V_psi >_psi
          + ( omega^2(psi) / 2 V' ) d/dt|_psi [ V' m_s < R^2 n_s >_psi ]
          - (1/2) m_s omega^2(psi) < n_s V_psi . grad R^2 >_psi
          + < ( Z_s e phi_0 - (1/2) m_s omega^2(psi) R^2 ) S^(n)_s >_psi .
```

Three of those terms carry a time derivative:

1. `dn_s/dt` outright, in the first term. `n_s` is an **evolved profile** — in
   MaNTA's vocabulary a `u`. It appears in the source of a *different*
   variable's equation (the temperature's), multiplied by `Z_s e phi_0`.
2. `d/dt|_psi [ V' m_s <R^2 n_s>_psi ]` in the third, which expands to a `dn/dt`
   term and a `dV'/dt` term. `V'` is a **metric coefficient** — a geometry slot.
3. `V_psi`, the flux-surface velocity, is `d psi/dt` by their (44), and it
   appears here and again in the compressional heating (200). That is the
   **field model's `psi`, differentiated in time**.

MaNTA today solves

```
a_i d_t u_i - d_x[ sigma_hat_i(x, t, u, q, phi, g) ] = S_i(x, t, u, q, sigma, phi, g)
```

and there is no route by which `S_i` can see any time derivative at all. So
these equations cannot be posed, let alone integrated. That is the gap.

**This is not a niche request.** The same shape turns up wherever a conserved
density is evolved but a *different* variable is the unknown — any formulation
that evolves temperature rather than energy density, or that carries a
potential-energy exchange, produces a `d(other variable)/dt` on the right-hand
side. It is the generic consequence of choosing state variables that are not
the conserved quantities.

**RoPP eq. (201) is the motivation, and it is the only one.** ASTRA's
stiffness stabiliser looks at first like a second, much better-worn case for the
same feature: `refs/ASTRA-8.pdf` Sec 5.4 calls it *"an additional artificial
diffusion, proportional to the time derivative of the relevant kinetic
profile"*, and says it *"has also been crucial for almost every predictive
transport modelling study which was carried out with the ASTRA code in the last
two decades"*. The primary source, `refs/PereverzevCorrigan.pdf` (Comput. Phys.
Commun. **179** (2008) 579), does not support that reading. What it adds is a
large implicit diffusive flux `Dbar u_x` against an explicitly evaluated
compensating convection `Vbar u`, `Vbar = Dbar u_x/u`; the two cancel
identically in the differential equation, and the residue the difference scheme
leaves is their eq. (16), `q~ = -tau Dbar u_xt`. That differentiates the
*gradient*, so in MaNTA's variables it wants `dq/dt` in the **flux**, not
`du/dt` in a source — and it carries the time step in front of it, so it is not
a physics term at all.

**Which makes it a description of pseudo-transient continuation, not of this
feature.** A term proportional to the step, vanishing at the fixed point,
present only to make the Newton step survivable, is what `PseudoTransient`
already does on the solver side of the interface. So ASTRA corroborates the
solver MaNTA has rather than the hook this spec adds, and a case that
reimplemented the stabiliser as a physics term on top of a `PseudoTransient` run
would be damping twice. A physics term of that shape would also modify the
effective mass matrix `X - dS/d(udot)` that the singularity check below guards:
there, that is not a hazard to be worked around but the whole mechanism by which
such a term does its job.

## What the solver already has

Three pieces of the machinery are already in place, which is what makes this a
modest change rather than a structural one.

**`residual()` is handed `dYdt` and already uses it.** The `u` row ends with
`+ XMats[i] * dYdt_h.u(var).getCoeff(i).second`, and `XMats[i]` is the
`aFn`-weighted mass matrix. The value is there; it simply never reaches a
physics hook.

**The scalar hooks already take both.** `ScalarG(s, y, ydot, ...)` and
`ScalarGPrime(dG, dGdot, y, ydot, ...)` are exactly this feature, for the
scalar rows, built and working. `assembleScalarCoupling` already forms
`s.u(v) + alphaValue * s_dt.u(v)`, which is the precise arithmetic the transport
rows will need. There is no new idea here, only a second application of one the
tree already contains.

**The absence is already a known defect.** `TODO` records that
`FieldModel::FieldResidual` receives no `states_dot`, that
`FieldResidualPrime`'s `dRdot` out-parameter is consequently unfillable by any
model that can exist, and that the fix is to pass the time derivatives through.
`FieldModel.hpp` says at the declaration that the slot "is here because the
coupling assembly already weights it by alpha, so the day the value hook gains
ydot the derivative is right rather than silently one term short." This design
is that day, for the transport rows; the field rows follow in phase 3 below.

## Decisions taken

**The time derivative reaches a case through `State`, not through a changed
hook signature.** This is the decision the self-consistent-fields spec took for
geometry (`2026-08-15-self-consistent-b-fields-design.md`, "Geometry reaches a
case through `State`, not a changed signature") and the reasons carry over
unchanged: every one of the forty-odd cases in `PhysicsCases/` keeps compiling,
every Python case keeps working, the batched path inherits it for free through
`GlobalState::operator[]` and `setWithState`, and the autodiff layer inherits it
by building one more `RealVector` from one more `State` field. A case that does
not read `s.udot()` is unaffected, bit for bit.

**Only `u` gets a time derivative. Not `q`, not `sigma`, not `phi`.** Those
rows are algebraic: the residual carries no `d/dt` for them, `id` marks them
zero for `IDASetId`, and `SuppressAlgebraicError` may take them out of the local
error test entirely. Their `Ydot` entries exist in the vector but are not
quantities with a meaning a physics case should read — IDA computes whatever
makes the algebraic rows hold, and under `SuppressAlgebraicError` does not even
error-control it. A source that read `sigma_dot` would make an algebraic row
differential without saying so, which changes the index of the system behind the
solver's back. RoPP (201) needs none of them.

**The derivative is reported through a second matrix, not a second slice.** The
value goes in `State`; the derivative `dS/d(udot)` goes in its own
`GlobalStateMatrix`, alongside `dSource_vals` rather than inside it. This is
asymmetric and deliberate: it is what `ScalarGPrime(dG, dGdot, ...)` and
`FieldResidualPrime(dR, dRdot, ...)` already do, and it keeps the multiplication
by `alpha` in exactly one place per assembly rather than scattered through the
slices.

**Opt in, per variable, in the spec.** `FieldSpec` gains
`bool sourceReadsTimeDerivatives = false`. It is what decides whether the new
derivative hook is called at all, and — more importantly — what arms the
singularity check below. `ScalarSpec::differential` is the precedent: a
declaration, as data, made before any hook can be called.

## Architecture

### The value

`State` gains one vector and its accessors, in the shape `geom` already has:

```cpp
/// d(variable)/dt at this point. Filled by the solver from the dYdt vector
/// before a physics hook is called; a case only reads it. Meaningful only for
/// the evolved variables -- there is deliberately no qdot, sigmadot or phidot,
/// because those rows are algebraic (see the design note).
double &udot(Index i);   double udot(Index i) const;
Vector &udot();          Vector const &udot() const;
```

`GlobalState` gains the matching `m_VariableDot` rows, so that
`states.setWithState` and `states[j]` carry it, and so that `DGSoln::evalOnNodes`
has somewhere to put it.

`residual()` already builds `dYdt_h`. It gains one call before
`ComputePhysics`, filling the new rows of the `states` it is about to hand over:

```cpp
states.setVariableDot(superconvergent
                          ? postprocessor->interpolateVariableOnStarNodes(dYdt_h)
                          : dYdt_h.evalOnNodes().Variable());
```

Nothing else in the value path changes: `Sources` keeps its signature and reads
`s.udot(j)` if it wants to.

`interpolateVariableOnStarNodes` is new but trivial, and the emphasis is on
*interpolate*. `Postprocessor::evalOnStarNodes` cannot be reused as it stands:
it substitutes `u*` for `u`, and the `u*` it has is the one `computeUStar(Y_h)`
built — reconstructed from `Y`'s `(u, q)`, not from `dYdt`'s. Reconstructing a
`u*` for `dYdt` instead is not the answer either, because `d(u*)/dt` involves
`q_dot`, which by the decision above has no meaning here. What is wanted is
`du_h/dt` interpolated from its own `P_k` coefficients onto the star points —
which is precisely what `evalOnStarNodes` already does to `q`, `sigma` and the
auxiliary variables, through `V(cell)`. The new method is that loop with the
`u*` branch removed, and `V(cell)` is the whole of its content.

**Whether interpolating rather than reconstructing caps the `k+2` rate is the
one genuinely open question in this design**; see Testing.

### The Jacobian

The `u` row of the residual becomes

```
res.u = B sigma + D u + E lambda - RF - S(u, q, sigma, phi, g, udot) + X udot
```

so the Jacobian gains exactly one term. `alpha` is IDA's `cj` for the forward
solve, `1/dt` for pseudo-transient continuation and `0` for a pure Newton solve
— all three already plumbed through `setAlpha`. In `assembleCellMatrix`, beside
the existing

```cpp
MX.block(uRows, uCols) += alphaValue * XMats[i];   // the mass term
MX.block(uRows, uCols) -= Su;                      // dS/du
```

goes

```cpp
MX.block(uRows, uCols) -= alphaValue * Sudot;      // dS/d(udot)
```

with `Sudot` built by `DerivativeSubMatrix` from the new
`dSourceDot_vals.Variable(i)` in the plain branch, and by
`accumulateStarBlocks(..., pp.V(i), ...)` in the superconvergent one — the
`sigma` chain, not the `u` chain, for the reason given above.

That is the whole of it. The term is a *coupling within the `u` block*, which
static condensation already handles: it changes the numbers in `MX`, not its
sparsity pattern, not the bandwidth of the condensed trace matrix, and not the
`[sigma|q|u|aux]` DOF layout.

### The effective mass matrix, and the one real hazard

Writing `M` for the mass matrix of the cell and `A = diag(a_i) M`, the `u` block
of `dF/d(udot)` is no longer `A` but

```
A_eff = A - (dS/d(udot)) M
```

which is no longer diagonal in the variable index and **is not guaranteed
non-singular**. If a case writes `S_i` containing `a_i du_i/dt`, the time
derivative cancels, the row becomes algebraic, and the index of the system rises
— which IDA will not diagnose. What it will do is fail in
`IDACalcIC` or stall the Newton iteration, three hundred lines from the cause.

For RoPP (201) the structure is benign: the `dn/dt` terms appear in the
*temperature* row, so `A_eff` is triangular with `a_n` and `a_T` on the diagonal.
But that is a property of those equations, not of the interface, so:

**At `initialize()`, for every cell, assemble `A_eff` at the initial state and
refuse a run whose `A_eff` is singular, naming the variables involved.** The
model for this is already in `Solver.cpp`, where a field DOF declared
differential whose residual row carries no `d/dt` is rejected before
`IDACalcIC` with a message that names the DOF — written precisely because the
alternative was `IDA_LINESEARCH_FAIL (-13)`, "a message about the linesearch for
a defect in the declaration". Same disease, same cure. A reciprocal condition
number below a threshold should warn rather than throw, since a badly
conditioned mass matrix is a legitimate if unpleasant thing to integrate.

The converse check is cheap and worth having: a variable declaring
`sourceReadsTimeDerivatives` whose `dSources_dudot` comes back identically zero
at the initial state should warn. The declaration costs a hook call per node per
Jacobian build, and a silent no-op means the author believes they wrote
something they did not.

### Pseudo-transient continuation and steady states

**No change, and the reason is worth recording because it is the design's best
property.** `steadyResidual` sets `ptcDYdt = id * (u - u_prev)/dt` and passes it
as `dYdt`; at the fixed point `u == u_prev`, so `udot` is exactly zero, and the
equations solved are the steady ones with `d/dt = 0` everywhere *including
inside the sources*. That is the physically correct reading of (194) with (201).
`SteadyMode::Newton` sets `ptcDYdt` to zero from the outset and gets the same
thing immediately. Neither needs special-casing.

One caveat belongs in `docs/running.rst`. The continuation's damping operator is
`A_eff/dt`, not `A/dt`, so the Kelley–Keyes argument that `1/dt` damps requires
`A_eff` to be positive definite, not merely non-singular. A case with a large
`dS/d(udot)` may find the SER schedule behaving unlike it does elsewhere. The
singularity check above does not catch this; it is a documented caution, not a
guard.

### Consistent initialisation

`setInitialConditions` solves `X udot = -B sigma - D u - E lambda + RF + S`
per variable per cell, by inverting the diagonal block. With a `udot`-dependent
source that system is wrong twice over: it is coupled across variables, and `S`
may be nonlinear in `udot`.

**Leave it alone.** It is a starting guess, and `IDACalcIC` with
`IDA_YA_YDP_INIT` is exactly the solver for this problem — it holds the
differential *values* fixed and solves for the algebraic values and the
differential derivatives, by damped Newton on the full residual, `dS/d(udot)`
included. The guess being one term short costs iterations, not correctness.

What must change is the *skip*. `Solver.cpp` skips `IDACalcIC` when the initial
weighted residual is small enough, and a case with `sourceReadsTimeDerivatives`
must not take that path: its initial `udot` is wrong by construction, and the
residual test cannot see it, because the residual is evaluated with the same
wrong `udot` that produced it. Force the call on, and say so in the log line
that already reports the decision.

*(Wrong, and not implemented. There is no such path to force. The only two that
skip are a steady solve -- where `udot` is the continuation's damping term
`(u - u_prev)/dt` rather than the guess `setInitialConditions` built, so the
guess is never used -- and a copy-path restart, whose stored `dYdt` was
consistent for these same equations. A cold time-marching run, which is the case
that would have needed forcing, already calls `IDACalcIC` unconditionally. No
code was added.)*

### Geometry time derivatives (phase 2)

Terms 2 and 3 of (201) need `dV'/dt` and `d psi/dt`, which are geometry, not
transport, quantities. They do not need a new field-model hook:

```
g_dot = (dGeometry_dpsi) . psi_dot   +   dGeometry_dt
```

and `dGeometry_dpsi` already exists, with `psi_dot` available as
`dYdt_h.getField()`. So `State` gains `gdot()` filled by an extended
`evaluateGeometry`, and `FieldModel` gains one new hook, `dGeometry_dt`,
defaulting to zero — needed only by a model whose metric has an explicit time
dependence at fixed `psi`, which is the `t` argument `Geometry` already takes and
which would otherwise be silently dropped.

The Jacobian side is heavier than phase 1: `dS/d(gdot)` contributes to the `A1`
coupling block against `psi_dot`, so it is weighted by `alpha` there, in
`assembleFieldCoupling`, rather than in `assembleCellMatrix`. That is the same
block the existing `dSources_dGeometry` feeds, one `alpha` heavier.

**Phase 1 alone does not close RoPP (201).** It closes the first term and the
`dn/dt` half of the third, which is the part that has nothing to do with the
field model and can be built, tested and used against a fixed geometry. Phase 2
is only reachable on a coupled run, and `docs/field_coupling.rst`'s own
statement stands — there is no Grad–Shafranov model in the tree yet.

### Field rows (phase 3)

`FieldResidual` gains `states_dot`, `dRdot` becomes fillable, and `TODO`'s entry
closes. Nothing in this design blocks it; it is listed so the sequence is on the
record. Its own prerequisite, per that entry, is "a field model whose residual
genuinely carries `d/dt` of a transport quantity, checked against a closed form
first."

### The autodiff layer

Mechanical. `AutodiffTransportSystem::Source` gains a `RealVector udot`
overload defaulting to forwarding to the existing one — the pattern the
geometry-aware `Flux` overload already uses, for the same reason — and
`dSources_dudot` is `dSources_du` with `wrt(udot)` in place of `wrt(u)`. Four
lines plus a declaration. A case that never mentions `udot` gets an identically
zero gradient from the same mechanism that gives a real one to a case that does.

### The Python layer

One line in `PyState.hpp`'s `StateView` and one in its binding, beside the
existing `u`, `q`, `sigma`, `sigmaHat`, `phi`, `geom`, `scalars`. A Python case
then writes `s.udot[0]` and needs no signature change, because the value travels
in `State`. `manta.jax`'s batched interface needs `udot` added to the arrays it
builds, alongside `u` and `q`.

## The adjoint

**Unchanged, and provably so.** `initializeMatricesForAdjointSolve` builds the
transpose of the *steady* Jacobian, at `alpha = 0`. The new term is
`alpha * Sudot`, so it is identically absent from the adjoint system. A case
with `udot`-dependent sources gets exactly the gradient it got before, and the
discrete adjoint stays the transpose of the forward Jacobian it is supposed to
be the transpose of.

The corollary is worth writing down where someone will find it: a *transient*
adjoint, which MaNTA does not have, would need this term, and would need it with
the sign and the `alpha` that the forward solve used at each step.

## Testing

The tree's rule is "write the case that would catch it being wrong before
changing it", and it is recorded twice in `TODO` for exactly this kind of
interface change. Four tests, in the order they should be written.

1. **A manufactured solution whose source carries `du/dt`.** Two variables, the
   second's source containing `c * du_1/dt` with `c` a constant of the case, and
   an exact solution with genuine time dependence — the existing MMS machinery
   prescribes the solution and derives the source, so this is a case file rather
   than new infrastructure. The order study must show `k+1` in `L^2`. Running it
   at `c = 0` and `c != 0` with the same exact solution is what separates "the
   term is implemented" from "the term is ignored and the source absorbed it".

2. **The assembled Jacobian against a finite difference of the residual, at
   `alpha != 0`.** `FiniteDifferenceJacobian.hpp` and `SolveJacTests.cpp`
   already do this for the mass term. The new term is invisible at `alpha = 0`,
   so a test that only checks the steady Jacobian passes with it missing — which
   is the failure this must be built to catch. `ScalarDerivativeCheck.hpp` is
   the pattern for differencing with respect to `Ydot` specifically.

3. **The superconvergent rate.** Test 1 with `Superconvergent = true`, asking
   whether `u*` still reaches `k+2`. If interpolating `udot` onto the star
   points rather than reconstructing it caps the rate, this is where it shows,
   and the answer changes the design: the fallback is to refuse the combination
   at configure time with a named reason, which is honest and cheap, rather than
   to ship a scheme that quietly converges one order low.

4. **The singularity check fires.** A case declaring `S_1 = a_1 du_1/dt` must be
   refused at `initialize()` with a message naming variable 1, not fail later
   inside `IDACalcIC`.

A regression case of the RoPP shape — density and temperature, the temperature's
source carrying `dn/dt` — belongs in `Tests/RegressionTests/` once 1–4 pass, and
in `python-examples/` if it is small enough to read in one sitting.

## Configuration, failure and output

No new configuration key. The feature is declared per variable in the spec,
which is where a property of the *equations* belongs; a run cannot turn it on or
off, because doing so would change which equations are being solved.

Three failure modes, all at `initialize()`, all naming the variable:

* `A_eff` singular — throw.
* `A_eff` ill-conditioned — warn, with the reciprocal condition number.
* declared but `dSources_dudot` identically zero — warn.

No output change. `udot` is `dYdt`, which the restart file already carries.

## Deliberately not in v1

* `qdot`, `sigmadot`, `phidot` in sources. Decided against above, on index
  grounds, not on effort.
* Scalars' `dS/d(nu_dot)` reaching the transport rows. The scalar hooks already
  take `ydot`; the transport source reading a scalar's time derivative is a
  separate coupling and no equation in (201) asks for it.
* Second time derivatives. Nothing in the formulation produces one.
* Making the hand-rolled initial `udot` solve exact. `IDACalcIC` is the right
  tool and is already called.
