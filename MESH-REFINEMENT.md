# Mesh refinement — working notes

Notes for `features/mesh-refinement`, written at a deliberate pause. Nothing in
this file is implemented. `FEATURES.md:4` is the roadmap entry; `refs/Refs.md`
("Mesh adaptivity") indexes the four papers and records what each supplies.
This file records what was **measured** here, which is a different thing, and
what the measurements changed about the plan.

Everything below came from throwaway Python spikes driving `manta.Runner`
through `Grid_points`, with no change to the core. Those scripts are gone; the
numbers and the methods are here because they are what is worth keeping.
Anything reconstructed from this file should be re-measured before being
trusted — several of the numbers below replaced earlier ones that were wrong,
and the retractions are recorded deliberately.

## Status

| | |
| --- | --- |
| Phase 0 (spike) | done, on three benchmarks |
| Phase 1 (into the solver) | **not started, and the plan as written should not be built** — see "What the measurements changed" |
| Landed on `main` | nothing from this work |
| Open PRs found in passing | **#13** restart use-after-free, **#14** `getDerivative`; both green, both unmerged |

## 1. The loop runs with zero core changes

Confirmed end to end on `park-convergence`, `shestakov-nonlinear` and
`AdjointPoster` (`Tests/RegressionTests/nonlin_ss.conf`, ported to Python
because no binding instantiates a registered C++ case). `Grid_points` accepts
an arbitrary mesh, `run_ss()` is the solve, `getSolution` /
`getPostprocessedSolution` give `u_h` and `u*` anywhere, and a warm start rides
on `Runner.configure()` rebuilding the `SystemSolver` while keeping the same
`TransportSystem` instance.

That validates the plan's Route A — a fresh solver per level, transfer through
the initial-condition hooks, never an in-place remesh. It is the one part of
the design that survived contact unchanged.

One wart: `Grid_size` is `requiredToml = requiredDict = true`
(`ConfigSchema.cpp:26`) even when `Grid_points` supersedes it entirely
(`makeGrid` ignores it outright, `SolverConfig.cpp:322`). The spike passed a
dummy. A conditional rule beside the one at `SolverConfig.cpp:272-281` is the
tidy fix.

## 2. `Superconvergent = true` is a prerequisite for the indicator, not a refinement

The most actionable result here, and true whether or not AMR is ever built.
Park, Spearman rank correlation between the accuracy indicator's per-cell
ranking and the **true** per-cell error's:

| level | cells | flag off | flag on |
| --- | --- | --- | --- |
| 0 | 4 | +0.400 | **+1.000** |
| 1 | 6 | +0.200 | **+1.000** |
| 2 | 9 | +0.383 | **+0.983** |
| 3 | 14 | +0.248 | **+0.991** |
| 4 | 21 | +0.114 | **+0.999** |

With the flag off the indicator barely beats chance. `refs/Refs.md` already
said "anything *calibrated* rather than used as a relative ranking should be
built on a superconvergent run"; this shows it bites the **ranking** too. An
adaptive driver must require the flag, or default to it loudly.

Related, and why the plan stopped using a Richardson target size:
`Tests/README.md:300-330` measures `u*` **not** superconverging at `k = 1` with
the flag off (1.91, 2.15, 2.26 against the assumed `k+2`), and for a nonlinear
flux the rate is *transient* — 6.9, 11.7, 9.1, then 2.3. A loop that calibrates
`h_target = h·(ε_target/ε_K)^{1/(k+1)}` on the coarse-grid ratio over-predicts
the gain and then spends its whole cell budget missing the target. Giorgiani's
degree rule (`Δk_i = ceil(log_b(E_i/ε_i))`, `10 ≤ b ≤ 100`) is the property to
steal instead: it assumes no convergence order at all.

## 3. h-adaptivity does not pay on any benchmark in this tree

Flag on, matched DOF, cost in physics visits (the unit `PERFORMANCE.md` uses).

**Park** — smooth, `u = exp(1-x²)-1`, entire:

| DOF | uniform | adaptive | gain | visits (adaptive cum. / uniform) |
| --- | --- | --- | --- | --- |
| 20 | 3.146e-06 | 3.146e-06 | 1.00× | 256 / 256 |
| 45 | 5.545e-08 | 4.450e-08 | 1.25× | 2242 / 576 |
| 105 | 8.045e-10 | 5.962e-10 | 1.35× | 4314 / 1344 |

At order `k+1 = 5` a 1.35× error gain is a **6% DOF saving**, bought with 3.2×
the visits. Predicted before the spike: the optimal equidistributed mesh varies
cell width by only 1.6–1.9× across the domain. There is nothing to adapt to.

**AdjointPoster** — a Gaussian source of width 0.141 at `x = 0.3`, i.e. a
genuinely localised feature, `k = 3` from 3 cells:

| DOF | uniform | adaptive | gain |
| --- | --- | --- | --- |
| 16 | 1.253e-02 | 2.434e-03 | 5.15× |
| 36 | 7.496e-04 | 3.523e-04 | 2.13× |
| 128 | 3.647e-06 | 2.038e-06 | 1.79× |

Sustained 1.8–2.6×, mesh concentrating at `x ≈ 0.2–0.4`. **Cost 17800 against
uniform-32's 6272 — 2.8× more.** Still a net loss.

**Shestakov** — adaptive at 66 DOF beat uniform at 90 (a 27% DOF saving) for
44% more visits; against the *kink-aligned* uniform trend the real gain is
≈1.5×.

### Two measurement traps, both hit

* **Kink alignment dominates everything at these sizes.** Shestakov's source kink is at `x = 0.1`, and `Grid_size = 10` puts it on a cell boundary: 10 → 6.75e-4, 15 → 2.41e-3, 20 → 3.33e-4, 25 → 1.41e-3, 30 → 2.21e-4. A 4–7× swing with nothing to do with adaptivity. An intermediate "5.02× gain" here was exactly this artefact (adaptive-15 against uniform-15) and is **retracted**. `AdjointPoster` has the same problem — its source centre `x = 0.3` lands on a boundary only when `0.3n` is integral, and 6 cells is worse than 5. Any AMR baseline on either case must use aligned meshes only.
* **Shestakov's error is not local, so this indicator structurally cannot see it.** `python-examples/shestakov-nonlinear/ANALYSIS.md` §5 established the error is one global mode living in `sigma`, "the shares constant to 0.1% over an 8× refinement". Rank correlation there is +0.41, +0.23 even with the flag on. Both indicators find where the error is *made*, not where it *lands*.

An unresolved oddity: on `AdjointPoster` the rank correlation is poor
(+0.005…+0.25) *with the flag on*, yet the mesh it produces is good. The likely
explanation is that it gets the few large-error cells right and the many
near-equal small ones contribute rank noise — a hypothesis, not a measurement.

## 4. The transfer must be the element polynomials, and `q` is half the value

This section replaced an earlier one that concluded the opposite; the earlier
conclusion came from a bad transfer.

MaNTA uses nodal elements, so the transfer is: evaluate the coarse cell
polynomials at the new nodes. `getSolution` already *is* that evaluation, so
the `CubicSpline` the first spike fitted was a pointless intermediate for `u` —
and for `q` it was the wrong function outright, off by **0.245 pointwise on a
field ranging to −1.0** (25%, 1.4% in L2). `q` could not be read from Python at
all, which is what PR #14 fixes.

On 32 cells at `k = 3`, cost in visits:

| initial `u` | initial `q` | PTC | Newton |
| --- | --- | --- | --- |
| `u = 0.3` flat | 0, and `q = d_x u` **exactly** | 6272 | 3328 |
| 12-cell solution | 0, inconsistent | 11392 | 10496 |
| 12-cell solution | `d_x` of that same `u` | **8064** | **7168** |
| exact solution | 0, inconsistent | 2688 | 1920 |
| exact solution | `d_x` of that same `u` | **1792** | **1024** |

1. **Making `q` consistent with `u` is worth ~30%, every time.**
2. **But accuracy in `u` dominates**: exact+consistent 1792 against 12-cell+consistent 8064, a 4.5× gap from an L2 difference of 2.3e-4.

A smooth sweep of the initial state from exact (`s=0`) to flat (`s=1`) rises
monotonically — 2688, 3840, 4864, 6272, 8320, 10624, 11392, 12672, 12928 — and
then **collapses to 6272 at `s = 1` exactly**. `u = 0.3` flat is the one state
where `q = 0` is correct, so it is exactly consistent *and* makes `a q / u^1.5`
vanish identically. Any partial approximation to the answer is worse than
either end. Worth knowing before anyone treats "closer initial guess" as
monotone.

With `u` and `q` both from the coarse element polynomials, the warm loop wins
and its margin grows with refinement: cumulative 16672 against 17800 cold, 16%
at L5 and 22% at L6. Accuracy is bit-identical at every level; only cost
differs. But a **single** 12→32 cell jump still loses to a cold start (7680
against 6272) — mesh sequencing pays because each level starts from one only
slightly coarser, not because transferred states are good.

## 5. The binding constraint is robustness, not accuracy

PTC fails on Shestakov past ~15 cells, and it is neither the adapted mesh nor
the warm start:

| configuration | result |
| --- | --- |
| 22-cell graded mesh, cold | FAIL `KINSol -7` |
| 22-cell graded mesh, warm | FAIL `KINSol -7` |
| **uniform 22 cells, cold** | **FAIL `KINSol -7`** |
| uniform 22 cells, warm | OK, 2.780e-04 |

Uniform *and* cold fails too, so this is PTC's own limitation on a degenerate
`D0 q³/u²` flux — consistent with `run.conf` pinning `TimeMarch` and with
`docs/running.rst:189-192` recording 705 visits against 283. Note the warm
start *helps* here: uniform-22 converges warm and fails cold.

**Newton beats PTC throughout on `AdjointPoster`** — 3328 against 6272 cold,
1024 against 1792 from the best start. `FEATURES.md`'s "or possibly the direct
newton solve as well" looks like the better first target, and it comes free
either way: `SteadyMode::Newton` is `solveSteadyState` with
`ptcStep = infinity` and no separate path (`SteadyState.cpp:17-19`).

## 6. p beats h by seven orders, and hp needs a smoothness sensor

On `AdjointPoster` at a matched ~130 DOF and equal cost: `k = 3` on 32 cells
gives 3.6e-6; `k = 10` on 12 cells gives **4.7e-13**. Adaptive *h* at the same
budget reached 2.0e-6. One degree bump beats the whole h-adaptive machinery.

Most of that needs no per-cell degree: choosing the **global** `k` by
Giorgiani's rule reached 2.8e-9 at 90 DOF in two iterations and 3060 visits,
against 2.0e-6 at 128 DOF for 16672 with adaptive h.

The counter-case is here too. Holding 10 cells and raising `k` from 2 to 12,
`AdjointPoster` falls eleven orders while Shestakov falls **19× and stops** —
`x^{4/3}` caps the regularity, and 19 is not a coincidence: `ANALYSIS.md` fits
that error to `1.8 h0/(k+1)²`, and `(13/3)² = 18.8` against 19.1 measured. So
this tree has one benchmark on each side of the classical hp criterion.

**Per-cell degrees are a much larger change than they look.** `DGSolnImpl`
holds one `const Index k` and one `const BasisType Basis` by value, and there
are ~320 `(k+1)` sites in the core — 200 in `SystemSolver.cpp`, 74 in
`Matrices.cpp`. That is the same single-`k` assumption `TODO` records as the
blocker for paper II's HDG+ family. A global-`k` loop is cheap; per-cell is a
project.

## 7. The modal-decay sensor works, and drives it from the decay *rate*

Woopen §4.3 (after Persson & Peraire), algebraically identical to Capasso
eq. (13):

```
S_K = (w − w_H, w − w_H)_K / (w, w)_K ,   w_H = L2 projection onto P_{k−1}
```

In a Legendre basis the projection just drops the top coefficient, so with
`u|_K = Σ_j û_j P_j` and `‖P_j‖² = 2/(2j+1)`:

```
S_K = [û_k²/(2k+1)] / Σ_j [û_j²/(2j+1)]
```

scale-free, no `|K|` factor — the `h/2` from the reference map cancels.

**It localises the singularity exactly.** `k = 4`, 10 uniform cells:

| | worst cell | value | rest of domain |
| --- | --- | --- | --- |
| `AdjointPoster` | cell 2, `x≈0.25` | 3.66e-10 | 8e-14 … 4e-10 |
| Shestakov | **cell 0, `x≈0.05`** | **1.42e-07** | ~1e-31 |

Shestakov's ratio across the domain is **2.1e24** — one cell carries the whole
signal, and it is the cell `ANALYSIS.md` §7 identifies. Separation between the
two benchmarks' worst cells: 389×. The accuracy indicator agrees (6.4e-06 in
that cell against ~2e-17 in the other nine).

**It is predictive** — roughest cell, `k` from 2 to 10:

| k | Poster S_K | Poster L2 | Shestakov S_K | Shestakov L2 |
| --- | --- | --- | --- | --- |
| 2 | 8.69e-06 | 4.98e-03 | 1.94e-05 | 6.75e-04 |
| 4 | 3.66e-10 | 8.28e-06 | 1.42e-07 | 2.40e-04 |
| 10 | 1.54e-23 | 1.46e-12 | 2.74e-10 | 4.93e-05 |

Poster's `S_K` collapses 17 orders and its error 9; Shestakov's falls 5 and its
error 1.1. The sensor's *behaviour under p* is what says whether p is worth
spending.

**But a fixed threshold `S*` is unsafe across degrees**, which Woopen does not
address. The separation is 389× at `k = 4` and only **2.2× at `k = 2`** — at low
`k` neither solution is resolved, so both look rough. Persson & Peraire's
`S* ~ 1/k⁴` is calibrated for shock capture and sits orders above anything here.

**So use the per-cell decay rate instead**: fit `log|û_j| = c − s·log j` over
`j = 1..k` within each cell (Mavriplis). One solve, no cross-degree
calibration. At `k = 6`, 10 cells:

| cell | x | Poster `s` | Shestakov `s` |
| --- | --- | --- | --- |
| 0 | 0.05 | 4.78 | **2.93** |
| 1 | 0.15 | 5.68 | 8.47 |
| 9 | 0.95 | 10.38 | 7.69 |

Shestakov's cell 0 is unambiguous — 2.93 against 8.47 in its neighbour, a 2.9×
jump between adjacent cells — and nothing in Poster falls below 4.78. `s < 4`
separates them.

**Margin, stated honestly:** that is a 1.6× gap between Poster's minimum and
Shestakov's singular cell, not a large one, and on milder singularities the
populations could overlap. Also Poster's top modes at `k = 6` are near
round-off, so its fitted `s` is a lower bound on smoothness rather than a
measurement — harmless for the decision, but not a regularity estimate.

**In the solver it is cheaper than the spike made it look.** The spike used
Gauss quadrature only to avoid touching the core. `NodalBasis` is built *from*
`LegendreBasis` and already stores `Vandermonde(i,j) = P_j(x_i)` as a member
(`Basis.hpp:363`, filled at `:391`), so nodal→modal is one reference-cell `V⁻¹`
shared by every cell, with no quadrature at all.

For reference, the spike's decomposition — a `k+3`-point Gauss rule is exact
here because `u_h` restricted to a cell *is* degree `k`:

```python
xg, wg = np.polynomial.legendre.leggauss(k + 3)
P = np.stack([np.polynomial.legendre.Legendre.basis(j)(xg) for j in range(k + 1)])
# The parentheses are load-bearing: * and @ share precedence and associate
# left to right, so without them the scaling multiplies u, not the integral.
uhat = ((2 * np.arange(k + 1) + 1) / 2.0)[None, :] * ((u * wg[None, :]) @ P.T)
```

## What the measurements changed about the plan

The approved plan is `~/.claude/plans/add-mnt-c-users-ian-downloads-crsc-tr02-humming-lake.md`.
Against it:

* **Phase 1 as scoped — accuracy indicator, Richardson target, h-equidistribution — should not be built.** §3: it does not pay on any benchmark here, and §6 says the degree is the stronger lever by seven orders.
* **The transfer design was right** (§4), and is the one piece to keep verbatim.
* **The Richardson exponent is out**, replaced by budgeted equidistribution or Giorgiani's order-free degree rule (§2).
* **`Superconvergent = true` becomes a prerequisite, not an option** (§2).
* **The `steadyNorm` work is unchanged and still blocked on the same thing**: `SteadyState.cpp:188-193` is a flat unweighted `sqrt(N_VDotProd(res,res))` read by the convergence test (`:198`, `:234`) *and* the SER ratio (`:256`), so `steady_state_tol` means something different on every mesh and `dt` cannot cross a remesh. `KINSetFuncNormTol` (`:154`) is handed the same flat norm, so normalising one alone decouples the inner and outer tests by `sqrt(N)`; and the loosened early-return at `:198` is hit *first on every level* right after a transfer. **`solveSteadyState` has no test at all** — `grep -rl` over `Tests/` finds only config parsing (`ConfigSourceTests.cpp`) and the tolerance setter (`SolverPlumbingTests.cpp:52-54`); nothing calls the function. A PTC unit test lands before that norm is touched, and `:154`/`:198`/`:234` move together or not at all.
* **Never two solvers alive at once.** `Integrator`'s cache (`PyIntegrator.hpp:16-43`) is process-global and keyed on one `(order, grid)` pair, and `residual()` calls `invalidateIfStale` on every evaluation (`SystemSolver.cpp:1255`). Two live solvers with different grids thrash it, and `getIntegrationWeights` returns a reference *into* the map `clear()` destroys. Extract grid + a `std::vector<double>` of `yJac`, destroy, then build — `PyRunner::configure`'s own discipline.
* **Adaptivity with `spatialParameters = true` must throw**, for the identical reason `Superconvergent = true` already does at `SystemSolver.cpp:1650`: a remesh redefines how many parameters there are.

## If picking this up again — suggested order

1. **`Grid(std::vector<Position>)` validation** — monotone, finite, positive width. It currently checks only `size() >= 2` (`gridStructures.hpp:151-167`), and `Interval(a,b)` silently swaps when `a > b` (`:28-32`); a degenerate cell gives `MassMatrix = 0` and a singular per-cell `FullPivLU`. Independent of everything else, one test beside `grid_from_points_rejects_too_few_points` (`GridTests.cpp:252`).
2. **A PTC unit test.** Nothing calls `solveSteadyState`, and everything else here touches it.
3. **The modal sensor** (§7) — small, self-contained, testable against these two benchmarks *without* a solve, and it is the piece all four papers' strategies need and only one of them supplies.
4. **Global-`k` selection by Giorgiani's rule** — 2.8e-9 at 90 DOF against 2.0e-6 at 128, no per-cell machinery, and it is the measured win.
5. Only then per-cell `p` (the ~320-site blocker) or `h`.

Two side quests this turned up, both worth doing regardless of AMR:

* PR **#13** — restart use-after-free, live today with no adaptivity involved.
* PR **#14** — `getDerivative`. `PyRunner` still binds nothing for `sigma`, `lambda` or `aux`, which will matter for `nAux > 0` and for the scalars.
