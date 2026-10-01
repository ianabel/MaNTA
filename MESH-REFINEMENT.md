# Mesh refinement — working notes

Notes for `features/mesh-refinement`, written at a deliberate pause and kept up
as pieces of it land. `FEATURES.md:4` is the roadmap entry; `refs/Refs.md`
("Mesh adaptivity") indexes the four papers and records what each supplies.
This file records what was **measured** here, which is a different thing, and
what the measurements changed about the plan. The status table says which parts
have since been built and where they live.

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
| Phase 1 (into the solver) | **the plan as written should not be built** — see "What the measurements changed". Working through the revised order at the end of this file instead. |
| Landed on `main` | the two side quests, **#13** restart use-after-free and **#14** `getDerivative`; then **#15** steady-state output, PTC diagnostics and the SER config keys — most of step 2 below; then **#16**, a restart that can resume at a different polynomial degree, which is step 4's state transfer |
| On this branch | steps 1–3 below: `Grid` validation, the merit function named and measured, and the modal sensor built — plus the merit function now **weighted** so the tolerance is mesh-independent near a solution, which is what step 2 was measuring towards. Merged up to `main` at `c585326`; no PR yet. |
| In review | **#17**, step 4 — global-`k` by Giorgiani's rule, plus `SteadyStateSolve`. Built on `feature-degree-adaptivity`, branched from `main` rather than from here, so none of it is below. |
| Next | **Built.** `MeshAdaptation` runs the p → h → p sequence (`MeshAdaptation.{hpp,cpp}`, `docs/adaptivity.rst`), and **§11** removed the last blocker: `KINSol -7` treated as fatal was the whole of §5's "PTC cannot do Shestakov", so the driver now runs that problem at 262×. What is open: the two cell counts §11 leaves unexplained, carrying `dt` across a remesh, and TimeMarch. Everything below §8 settles the scheme; it is now description rather than plan. Formerly:** Solve uniform at `k >= 4`; decide from the per-cell decay rate whether to grade and at which end (§10); if so rebuild at the *same cell count*, graded as hard as the solver tolerates (§9, worth 14900× at fixed DOF); then run global-`k` to tolerance (#17). **The order is forced, not chosen** — at `k = 2` the grading decision is not merely unreliable but *reversed*. The mesh half is now configurable (`GradedGridBoundary`), so what is left to build is the decision and the loop. Per-cell `p` is gated *no*. Also open: carrying `dt` across a remesh, now that the norm allows it. |

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

~~One wart: `Grid_size` is required even when `Grid_points` supersedes it
entirely, so the spike passed a dummy.~~ **Fixed.** `GridSize`, `LowerBoundary`
and `UpperBoundary` are now required only when `GridPoints` is absent and the run
is not a restart, and the rule lives inside `checkRequired`'s aggregation so a
config missing several of them hears about all of them at once. The keys are
`UpperCamelCase` now, with the old spellings kept as deprecated aliases.

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

## 3. *Indicator-driven* h-adaptivity does not pay on any benchmark in this tree

**Read §8 before acting on this section.** Everything measured here stands, but
its heading originally read "h-adaptivity does not pay" without the qualifier, and
that generalised too far: §8 measures a *hand-graded* mesh on Shestakov at 14900×
the accuracy of a uniform one at matched DOF. What does not pay is the
equidistribution loop below, driven by the accuracy indicator — and §3's own
second measurement trap says why, which is the part that turned out to matter.

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

## 5. ~~The binding constraint is robustness, not accuracy~~ — **retracted; it was one return code**

**Read §11 first.** The conclusion below — that PTC has an intrinsic limitation on
Shestakov's degenerate flux — is wrong, and every `KINSol -7` in this section is the
same defect: the continuation loop treated that return code as fatal when it means
"this dt was too ambitious", which is exactly what the loop already handled for
`KIN_MAXITER_REACHED`. One line, and PTC now converges on Shestakov at 5, 8, 10, 12,
20, 25, 30, 40 and 50 cells, at every degree and with `Superconvergent` either way.

The measurements are kept because they are what a reader of this section will find
and because the *warm-start* observation in them stands. But do not conclude
anything about PTC's limits from them.

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

## 6. p beats h by seven orders *on a smooth problem*, and hp needs a smoothness sensor

The qualifier is §8's, added after it measured `h` beating `p` by 14900× on
Shestakov. Both results stand: the lever depends on the regularity, which is the
whole reason a smoothness sensor is what decides between them.

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

## 8. Grading towards Shestakov's singularity is worth 14900×, and it is `h` not `p`

This is the gate measurement "Per-cell degree" asks for, run on the one benchmark here that
global `k` cannot serve. It came back decisively — and pointing at a different
lever from the one the gate was written to test. Throwaway Python driving
`manta.Runner` through `Grid_points`, `TimeMarch` as `run.conf` pins, the
benchmark's own relative-L1 metric against `ExactSolution` on 201 points, plus a
relative L2 on 2400 points including a geometric fan into the axis (a uniform
sample puts only three points inside the innermost cell of a graded mesh, so a
metric built on it could miss the error there entirely; the two agree throughout).

**Why this problem is the clean test.** Its exact steady state is
`[w0 + (1/3)(S0 d/D0)^{1/3}(Lx - x)]^3` for `x >= d = 0.1` — a **cubic in x**, so
any cell lying entirely outside the source is represented exactly at `k >= 3`. The
inner branch carries `0.75(d^{4/3} - x^{4/3})` cubed, i.e. powers `x^{4/3}`,
`x^{8/3}`, `x^4`, and the first has an unbounded second derivative at the axis.
One singular point, at a known location, with the rest of the domain a polynomial.

### One law covers both mesh families

Meshes graded geometrically towards `x = 0` — `n_in` cells at `d*sigma^j`, then
`n_out` uniform on `[d, Lx]`. At `k = 5`, with `h0` the innermost cell's width:

| mesh | `h0` | rel L1 | `error / h0` |
| --- | --- | --- | --- |
| uniform 10 | 1.000e-01 | 4.908e-03 | 0.04908 |
| uniform 20 | 5.000e-02 | 2.446e-03 | 0.04892 |
| uniform 30 | 3.333e-02 | 1.629e-03 | 0.04887 |
| uniform 40 | 2.500e-02 | 1.221e-03 | 0.04884 |
| graded 4+3 `s=0.3` | 2.700e-03 | 1.316e-04 | 0.04874 |
| graded 5+3 | 8.100e-04 | 3.946e-05 | 0.04872 |
| graded 6+3 | 2.430e-04 | 1.183e-05 | 0.04868 |
| graded 7+3 | 7.290e-05 | 3.551e-06 | 0.04871 |
| graded 8+3 | 2.187e-05 | 1.066e-06 | 0.04874 |

**`error = 0.0487 h0`, to 0.5% across a 4600× range of `h0` and across both
families.** So the mesh does not matter and the *number of cells* does not matter:
only the width of the cell touching the singularity does. The law predicted the
best run before it was made — `graded 9+1` has `h0 = 6.56e-6`, predicting
3.195e-07 against 3.292e-07 measured, 3% out.

Which turns the whole question into "how cheaply can `h0` be made small":

* uniform gives `h0 = (k+1)/DOF` — **algebraic**;
* graded at `sigma = 0.3` gives `h0 = d sigma^{n_in - 1}` — **exponential in DOF**.

At 60 DOF and `k = 5`, that is `h0 = 0.1` against `6.56e-6`, a factor 15200 —
against a measured error ratio of **14900×** (4.908e-03 uniform, 3.292e-07
graded). Extrapolating the law, matching the graded run's error on a uniform mesh
needs ~148000 cells, **889000 DOF against 60**. Against the best uniform run
actually measured — 40 cells at `k = 5`, 240 DOF, 91680 visits, 1.221e-03 — the
graded one is 3700× more accurate for 4× fewer DOF and 4.9× less cost.

### It is `h`, not per-cell `p`, and that is measured rather than argued

Three results, and together they answer that gate in the negative:

* **The outer region contributes nothing, so there is nothing for per-cell `p` to
  win there.** `graded 9+1`, `9+3` and `9+5` at `k = 5` give **identical** errors —
  3.292e-07 and 4.832e-07 on both metrics, to every digit — so 24 of the 84 DOF in
  the widest one are pure waste. That is a 29% DOF saving, and it is available by
  *choosing fewer outer cells*, with no per-cell machinery at all.
* **The outer region contributes nothing even at `k = 2`**, which was the
  hypothesis this was meant to test and it failed. Holding `n_in = 5` and taking
  `n_out` from 5 to 40 moves the error by 0.3% and then not at all: 1.573e-04,
  1.578e-04, 1.578e-04, 1.578e-04. At `k = 3`, 8.875e-05 four times over,
  unchanged to every digit. So the degrees a per-cell rule would shave off the
  smooth region were never buying anything to begin with.
* **`p` is not what is capped — it is what makes the `h0` law hold.** At `k = 5`
  the law is exact over five gradings. At `k = 2` it breaks once `h0` is small:
  7+5 gives 2.666e-05 and 9+5 only 2.247e-05, a 1.19× gain for a 11× smaller
  `h0`. Not the outer region (above), so it is the *other* inner cells — at `k = 2`
  they resolve `x^{4/3}` badly enough to floor the total. So a global `k` of about
  4 is a prerequisite for grading to pay, and beyond that more degree buys
  nothing: the singular cell's `O(h0)` is insensitive to it.

So per-cell degrees buy a few percent of DOF on this problem, against a gate of
3×. **Per-cell degree as scoped should not be built.** What should is the thing §3's loop
failed at for a reason §3 itself identified: putting the small cells in the right
place.

### And §7's sensor is exactly what picks the place

§3's equidistribution loop got 1.5× here because it was driven by the accuracy
indicator, which §3 measured as blind on this problem — rank correlation +0.41,
+0.23 even with `Superconvergent` on, because the error is one global mode living
in `sigma` and the indicator finds where error is *made*, not where it lands.
§7's modal sensor is not blind: it puts the singular cell at `x ≈ 0.05` with a
**2.1e24** ratio across the domain, and the decay rate falls monotonically towards
it (3.93, 6.73, 7.83, … 10.65). One cell, unambiguously, and it is the right one.

The rule that follows needs no indicator calibration and no target-size formula:
**find the cell the sensor says is singular, split it geometrically towards the
singular end, repeat.** `error = 0.0487 h0` says each split at `sigma` multiplies
the error by `sigma` for one extra cell, which is exponential convergence for a
linear DOF cost, and the constant is measurable on the fly.

### The binding constraint is robustness, and it is now the whole ceiling

§5 recorded this for PTC. It holds for `TimeMarch` too, on hand-graded meshes, and
it is what stops this being an open-ended win. Of 52 runs in the first sweep, **15
failed** — `IDASolve` corrector failures at `h = 1e-7`, and two `IDACalcIC`
failures. One of the fifteen is a *uniform* mesh, 40 cells at `k = 3`, so this is
not a grading artefact. The pattern is not monotone in anything: `9+1` and `9+3`
at `k = 5` both succeed with identical answers while `9+2` at `k = 5` fails, every
`sigma = 0.15` mesh fails at every `k` tried, and `9+2` succeeds at `k = 3` and
`k = 4` but not at 2, 5, 6, 7 or 8.

So the reachable floor is `h0 ~ 7e-6`, i.e. an error of ~3e-7 — which is where the
best run above sits. **Any h-adaptive loop here will hit this before it runs out of
accuracy to gain**, and will need to treat a failed level as a rejected step rather
than as a fatal error. Worth knowing before the loop is written, not after.

**§9 refines this in two ways and one of them corrects it.** The `h = 1e-7` in
those corrector failures is the *default* `MinStepSize`, so the solver is hitting a
configured floor rather than diverging; setting it to `1e-12` buys exactly one more
rung, to 1.385e-07. And this paragraph originally called it "a solver limit rather
than a method one", which overstates: at that rung `err/h0` has risen from 0.0487
to 0.0908, so the `h0` law is breaking at about the same place the solver gives up.
Both walls, near 1e-7, together.

## 9. It should move cells, not add them: `r`-adaptivity at fixed DOF

§8 established `error = 0.0487 h0` with no dependence on the cell count. Taken
seriously that says the useful move is *redistribution* at a fixed budget rather
than refinement — and §8's own headline run was already exactly that without
saying so. `graded 9+1` is **10 cells and 60 DOF, the same as `uniform 10`**. The
14900× was bought by moving boundaries, not by adding any.

Measured deliberately, uniform against the best redistribution at an identical cell
count and identical DOF, `k = 5`:

| N | DOF | uniform | best graded, same N | gain | winning mesh |
| --- | --- | --- | --- | --- | --- |
| 10 | 60 | 4.908e-03 | **3.292e-07** | **14900×** | 9+1 `s=0.3` |
| 10 | 40 (`k=3`) | 1.112e-02 | 2.268e-06 | 4900× | 9+1 `s=0.25` |

Only `N = 10` and 20 are quoted because only they are kink-aligned, and §3's first
trap is that misalignment swings the uniform baseline 4–7×. The misaligned budgets
give larger numbers — 245× at `N = 5`, 1190× at `N = 8` with `k = 3`, 50169× at
`N = 8` with `k = 5`, 13714× at `N = 15` with `k = 3` — and they are *not* quoted as
the result, because a chunk of each is the baseline being handicapped. At `N = 20`
every grading tried failed to solve, so there is no number.

### Against which, refinement at a fixed distribution is nearly worthless

The comparison that makes the case. Uniform, `k = 5`, quadrupling the budget:
10 → 40 cells is 60 → 240 DOF and 4.908e-03 → 1.221e-03, a gain of **4.0×** — which
is just `error = 0.0487 h0` with `h0 = 1/N`, i.e. the DOF and the accuracy trade
one for one.

So **4× the DOF buys 4×, and 0× extra DOF buys 14900×.** That is the whole finding,
and it is why the loop wants to move boundaries rather than split cells.

### Which is a much smaller change than h-refinement

Worth stating plainly, because it is the practical payoff of the above. A fixed-DOF
remesh does not change the DOF count, so none of what made Phase 1 "Route A"
awkward applies: no reallocation of `Y` and its clones, no change to the layout
formula duplicated across five files, no restart-format change, no new
`SystemSolver`. `Grid_points` already accepts an arbitrary mesh, `DGSoln::copy`
still refuses a different grid but the projection transfer §4 settled is what a
remesh wants anyway, and `Integrator`'s cache invalidates on the grid as it should.

And the target is forgiving. Down the whole working range the error is *monotone*
in `sigma` with no optimum to find — 1.902e-05, 8.187e-06, 3.190e-06, 1.094e-06,
3.292e-07 at `sigma` = 0.5, 0.45, 0.4, 0.35, 0.3 — so the rule is "grade as hard as
the solver tolerates", not "hit the right grading". A crude rule loses very little.

Redistribution is not free in *cost*, only in DOF: the winning mesh takes 18660
physics visits against uniform-10's 12240, 1.52× for 14900×.

### Warm starting buys cost, not reach — and the wall is not a bad guess

The obvious hypothesis for the §8 ceiling was that it is a globalisation problem:
each mesh is solved cold, an adaptive loop grades gradually, so sequencing might
walk through it. §4 supports it — "uniform-22 converges warm and fails cold".

**Tested and refuted.** A ladder at fixed `9+1`, 60 DOF, `sigma` from 0.5 down,
each rung started from the previous rung's `u` and `q` (each from its own element
polynomials, per §4), **fails at `sigma = 0.3`, which the cold start solves.** So
the wall is not distance from the answer, and mesh sequencing does not lift it. The
likely reason warm was *worse* there is §4's other warning: the transfer is a cubic
spline through the singular region, and an interpolant that overshoots `x^{4/3}`
gives a state whose inconsistency is not the kind Newton walks off.

What the ladder does buy is cost, and a lot of it — per rung, warm against cold:

| `sigma` | error (both) | cold cost | warm cost |
| --- | --- | --- | --- |
| 0.45 | 8.187e-06 | 16080 | **2880** |
| 0.40 | 3.190e-06 | 25380 | **3060** |
| 0.35 | 1.094e-06 | 18780 | **4020** |

**5–6× cheaper per level, for an answer identical to four significant figures** —
which is §4's finding again ("accuracy is bit-identical at every level; only cost
differs"). But the *cumulative* ladder is 26280 to reach 1.094e-06 where one cold
solve on that mesh costs 18780, so sequencing loses unless the intermediate levels
were wanted anyway. Also §4's finding, and it means an adaptive loop should expect
to pay about what a direct solve would.

### The ceiling is two limits arriving together at ~1e-7

Diagnosed one config key at a time on the meshes that failed, at `9+1`, `k = 5`:

* **`MinStepSize` is a real part of it, and buys exactly one more rung.** The
  failure is `IDASolve` reporting corrector failure at `h = 1e-07`, which is the
  *default* `MinStepSize` — so it is hitting the floor rather than diverging. Set
  it to `1e-12` and `sigma = 0.25` solves, reaching **1.385e-07** at 60 DOF. `1e-16`
  gives the identical answer, so 1e-12 is not the binding constraint.
* **Then it is hard.** At `sigma = 0.2` every variant tried fails: `MinStepSize`
  1e-12 and 1e-16, `SuppressAlgebraicError = false`, `atol = 1e-6`,
  `rtol = 1e-10`, `Newton`, `PseudoTransient`. At `sigma <= 0.15` the failure moves
  from `IDASolve` to **`IDACalcIC`** — the initial condition cannot be made
  consistent at all, which is a different and worse problem.
* **The law is breaking at the same place**, so this is not purely a solver
  ceiling. `err/h0` runs 0.04870, 0.04869, 0.04868, 0.04857 for `sigma` 0.5 to
  0.35, then 0.05018 at 0.3 and **0.09075** at 0.25 — nearly double. Part of that
  is the metric: the relative-L1 sample is 201 uniform points whose first interior
  one is at `x = 0.005`, so past `sigma ≈ 0.35` it never looks inside the innermost
  cells and is measuring the error those cells *propagate* rather than the error
  they contain. Either way there is a floor near 1e-7 in the quantity anyone would
  quote, and it arrives just as the solver does.

So the reachable gain from redistribution on this problem is a little over four
orders of magnitude at fixed DOF, and both walls sit at about the same place. **A
loop should therefore treat a failed grading as a rejected step and back off**,
which is also what §8 concluded, and it should not expect to be limited by its
rule — it will be limited by the solver first.

## 10. The decision is decidable, and it forces p before h

§9 says grading is the win and §7 says the sensor localises the singularity. Neither
asked the question a driver actually has to answer: **from a uniform solve, should
this mesh be graded at all, and at which end?** That is not the localisation
question. §7 compared two benchmarks' worst cells against *each other*; a rule needs
to compare one problem's end cells against *its own interior*, and to stay quiet on a
problem that wants nothing.

Rule under test, from a uniform 10-cell solve with `Superconvergent = true`: fit the
per-cell decay rate `s` (§7), then **grade whichever end's cell is rougher than the
interior median by 2x or more; otherwise leave the mesh alone.** Compared against the
interior rather than a fixed threshold, because §7 measured a fixed threshold as
unsafe across degrees.

| | k = 2 | k = 3 | k = 4 | k = 5 |
| --- | --- | --- | --- | --- |
| **Shestakov** — wants Lower | grade Lower, 2.69x | grade Lower, 3.09x | grade Lower, **6.80x** | grade Lower, 6.44x |
| **Park** — wants nothing | **grade Lower, 3.63x** | uniform, 0.99 | uniform, 1.19 | uniform, 1.01 |
| **Jardin** — wants nothing | uniform, 0.97 | uniform, 1.00 | uniform, 0.97 | uniform, 1.15 |

**At `k >= 3` it is clean, and the margin grows.** Shestakov 3.09-6.80 against Park
and Jardin 0.97-1.19, so any threshold in (1.2, 3.0) separates them, with 5.7x of
headroom at `k >= 4`. "Which end" is unambiguous where it fires: Shestakov's ends read
2.85 and 19.2 at `k = 4`, and Park's read 6.66 and 7.98 with a median of 7.95.

**At `k = 2` it is not merely blind, it is inverted.** Park's ratio (3.63) *exceeds*
Shestakov's (2.69), so no threshold works at all: the rule would grade the entire
function harder than the singular one. §7 recorded that two modes cannot see a
singularity; this is worse than that, and the mechanism is specific.

### Why `k = 2` inverts, and why it would keep inverting

The raw spectra of the first cell, normalised to the mean:

| | `û_1` | `û_2` | `û_3` | `û_4` |
| --- | --- | --- | --- | --- |
| Shestakov, `k = 4` | 4.76e-02 | 6.81e-03 | 1.90e-03 | 9.72e-04 |
| Park, `k = 4` | 7.92e-03 | 2.63e-03 | **7.91e-06** | 1.14e-06 |

Shestakov decays by 7.0x, 3.6x, 2.0x -- slow and algebraic, all the way out. Park
decays by **3.0x and then 332x**: its spectrum has a *knee*, and the honest decay only
starts at `j = 3`. A two-point fit over `j = 1, 2` sees the knee and reports
`log(2.63/7.92)/log 2 = 1.575`, which is the 1.576 measured. Nothing about roughness.

**The knee is not an accident of this problem, it is what a Neumann axis does.** Park
is even about `x = 0` and the axis is zero-flux, so the solution is locally flat there
and its *linear* content on the first cell is suppressed relative to its quadratic
content. Every MaNTA problem with a zero-flux axis has that, on the very cell the
question is being asked about. So a two-point fit does not fail at random on a smooth
problem -- it fails **systematically, in the direction of a false positive, at exactly
the boundary a driver is interrogating.** `k >= 3` is the fix, and `k >= 4` puts two
genuinely decaying modes into the fit rather than one.

### Drive it from the decay rate, not from the accuracy indicator

The indicator `e_K = ||u* - u_h||/sqrt|K|` has far more *contrast* --
`e(cell 0) / e(worst interior)` is 5.6e11, 1.5e11, 9.8e10 for Shestakov at
`k = 3, 4, 5` against 1.16, 0.21, 1.22 for Park -- eleven orders against about one.
Tempting, and the wrong choice, for three measured reasons:

* **Its denominator vanishes.** Shestakov's exact solution is a cubic outside the
  source (§8), so at `k >= 3` every interior cell is *exact* and `e(interior)` is
  round-off, 2.4e-17. The eleven orders are partly that. Jardin is worse: its steady
  state is degree 1, so `e` is ~5e-16 **everywhere** and the ratio 1.57-2.06 is
  round-off over round-off -- the identical trap that made degree adaptation climb to
  its ceiling on `LinearDiffusion` (see step 4 above), arriving from a different
  direction.
* **`s` has a fixed scale and `e` does not.** `s` is an exponent, so "rougher by 2x"
  means the same thing at every degree and on every problem; `e` carries the
  solution's units and its own convergence order.
* **`s = infinity` is a *meaningful* verdict.** Jardin's first cell has nothing above
  the round-off floor at `k = 2, 3, 4`, and the fit correctly returns infinity: as
  smooth as representable. The rule reads that as "do not grade". The indicator has no
  such answer -- it returns noise.

### Which forces the order of an hp scheme, and the answer is p first

Two independent measurements, from opposite directions, say the same thing:

* **§9:** `k >= 4` is a prerequisite for grading to *pay*. At `k = 2` the
  `error = 0.0487 h0` law breaks once `h0` is small -- an 11x reduction bought 1.19x --
  because the non-innermost cells stop resolving the singularity.
* **§10, above:** `k >= 3` is a prerequisite for the grading *decision to be correct*,
  and at `k = 2` it is not merely uncertain but reversed.

So the scheme is:

1. **Solve uniform at `k >= 4`.** Cheap, and both of the above need it.
2. **Decide from the per-cell decay rate**: grade an end, or leave the mesh alone.
3. **If grading, rebuild at the same cell count** and grade as hard as the solver
   tolerates -- §9 measured the error monotone in the ratio with no optimum to find,
   so there is nothing to calibrate.
4. **Then global-`k` adaptation to tolerance** (step 4 above, #17) on that mesh.

Doing it the other way round -- grade first from a cheap low-order solve -- grades the
wrong problem at the wrong end. That is the whole content of the `k = 2` row.

**Scope, stated plainly.** Three problems, one of which wants grading, and one
threshold picked from the gap between 1.19 and 3.09. That is enough to establish the
mechanism and the ordering; it is not a calibration. A milder singularity than
`x^{4/3}` would sit closer to the smooth population, and the honest response is that
the threshold should be recorded as a config key with this measurement beside it
rather than buried.

## 11. Shestakov was never PTC's limit — it was one return code treated as fatal

§5 concluded that pseudo-transient continuation had an intrinsic limitation on
Shestakov's degenerate `D0 q^3/u^2` flux. That is retracted. Re-measured after #15,
#16, #17 and this branch's weighted merit function had all landed, the whole thing
was `KINSol` returning `KIN_MXNEWT_5X_EXCEEDED` (-7) and `solveSteadyState`
treating it as a hard error.

### The diagnosis: every schedule lever was inert

The tell was not the failure but its *invariance*. At 20 cells and `k = 2`, all of
these failed at the **same continuation step with the same residual**:

* `PseudoTransientMaxStep` at infinity, 1e4, 1e3, 1e2, 10, 1;
* `PseudoTransientSERRate`/`Floor` at (1, 2), (0.5, 1.2), (0.25, 1.1), (0, 1.1);
* `PseudoTransientInitialStep` at 1e3, 1, 1e-2, **1e-4**;
* `SteadyStateTolerance` at 1e-11, 1e-8, 1e-5, 1e-3.

An initial `dt` of 1e-4 makes the damping term `id*(u - uPrev)/dt` enormous, so the
Newton step should be tiny; it failed identically anyway. **A failure that does not
move when the schedule moves was never a schedule failure**, and that is what
redirected attention from the continuation to the error handling.

### The fix

`-7` is KINSOL's report that five consecutive Newton steps hit the maximum length.
`KINSetMaxNewtonStep` is 1e10 here, so the direction really is that long — but that
is a statement about the *iteration*, not about the problem, and the loop already
had the right response for `KIN_MAXITER_REACHED`: restore the state, damp, retry.
The comment beside that check even argued the case in general terms — "the ordinary
way an attempt at too large a dt ends" — and then omitted this code.

`SteadyState.cpp` now tolerates `-7` alongside `-6`. What it was worth on
Shestakov at `k = 2`:

| cells | `-7` fatal | `-7` a rejected step |
| --- | --- | --- |
| 5, 8, 12, 20, 25, 30, 40, 50 | **fail** | converge |
| 10 | converge | converge |
| 6, 15 | fail | fail, differently — see below |

And with `Superconvergent` on, which `MeshAdaptation` requires, **every**
combination of `k` = 2…5 at 10 and 20 cells went from failing to converging. That
flag had been the second blocker: at `k = 2` on 10 cells the run converged with it
off and threw `-7` with it on, which is why the driver could not touch this problem.

Also dropped in passing: the same condition excluded `KIN_STEP_LT_STPTOL`, which is
`+2` — a warning-level return that cannot reach a `retval < 0` branch. It never did
anything and looked load-bearing.

### The driver now runs it, at 262×

Which was the point. `MeshAdaptation` on Shestakov, 10 cells, `k = 3`:

| | relative L1 |
| --- | --- |
| uniform | 1.989e-02 |
| **p → h → p** | **7.603e-05** |

Same cell count, same DOF. This is the problem every measurement in §8–§10 came
from and that the driver was documented as unable to run.

### What still fails, and what it is not

6 and 15 cells still fail, and with a *different* symptom: the loop exhausts its 200
continuation steps rather than hitting a hard `KINSol` error. Both are rescued by
putting a cell boundary on the source kink at `x = 0.1` through `GridPoints`, which
drops each to the 10-cell error exactly.

**Cell-boundary alignment is not the rule, though.** It was the obvious hypothesis
and it is wrong: 5, 8, 12 and 25 cells are all misaligned and all converge. So the
kink is implicated in those two cases without explaining them, and the remaining
failure is unexplained rather than understood. Recorded as such.

### The general lesson

The 15 failures out of 52 in §8, and the ceiling §9 attributed to the time
integrator, are now suspect in the same way. §9's `MinStepSize` finding stands on its
own evidence — the failure there was `IDASolve` at `|h| = MinStepSize` and lowering
the key bought a level — but "the solver cannot do this" deserves the same test that
broke this one open: **change the schedule and see whether the failure moves.** If it
does not, the schedule is not what is failing.

## What the measurements changed about the plan

The approved plan is `~/.claude/plans/add-mnt-c-users-ian-downloads-crsc-tr02-humming-lake.md`.
Against it:

* **Phase 1 as scoped — accuracy indicator, Richardson target, h-equidistribution — should not be built.** §3: it does not pay on any benchmark here, and §6 says the degree is the stronger lever by seven orders.
* **The transfer design was right** (§4), and is the one piece to keep verbatim.
* **The Richardson exponent is out**, replaced by budgeted equidistribution or Giorgiani's order-free degree rule (§2).
* **`Superconvergent = true` becomes a prerequisite, not an option** (§2).
* **The `steadyNorm` work is measured, and now done.** It was a lambda inside `solveSteadyState` that nothing could reach; it is `SystemSolver::steadyResidualNorm`, weighted by `residualWeights()`, with four cases in `SolverLifecycleTests.cpp`. What they establish:
  * **Flat, it went like `sqrt(h)`.** 4 / 8 / 16 cells on the same problem and the same initial function gave `‖F‖` = 0.5557 / 0.3935 / 0.2784 — a ratio of 0.70806 then 0.70742 against `1/sqrt(2) = 0.70711` — while the state itself agreed to 1.0e-3. The cell rows are pairings against the basis, so each carries a mass factor going like `h`, and the row count goes like `1/h`. **Weighted by `1/sqrt(h_K)` per cell row** the same five meshes give 1.11145 / 1.11294 / 1.11344 / 1.11358 / 1.11361, departures from the limit falling by 3.2, 3.8, 4.8 — so what is left is second-order discretisation error in the state, not a mesh factor. The trace rows keep weight 1: a `lambda` row is a flux condition at one face and has no `h` in it.
  * **KINSOL is measuring the identical quantity**, and now by construction rather than by coincidence. `KINSetFuncNormTol` gets the same `steady_state_tol` and KINSOL's test is `N_VWL2Norm(fval, fscale)`, so it is handed *the same weights* as `f_scale` — `solveSteadyState` passes `resScale` where it used to pass `kinScale` twice. In `Newton` mode (damping identically zero) the two agree bit for bit, 1.302e-15. `u_scale` stays unit; that one drives the step-length test and the Newton step clamp.
  * **The `sqrt(h)` finding was only half the story, and the other half is a caution.** That scaling holds *near a solution*, where the algebraic rows are satisfied and the `u` row carries the residual. Far from one the `q` and `lambda` rows hold the `O(1)` trace and derivative terms of the weak form instead — the `1/h` from `phi'` cancels the `h` from the measure — and the flat norm **grows** like `1/sqrt(h)`: 4.275, 6.166, 8.834, 12.58, 17.86 on 4 to 64 cells, against `sqrt(h)` *down* for the consistent state. Weighting makes that regime worse, not better (measured `1/h`). Two mechanisms, opposite signs, and which dominates is a property of the state, so **no fixed row weighting is mesh-independent everywhere**. What is fixed is the regime the convergence test fires in. A reported starting `‖F‖` is still not comparable across meshes, which matters for the next bullet.
  * The early-return an adaptive driver hits first on every level, right after a transfer, is still a risk — but a *smaller* one now, and in the safe direction. A transferred state is far from a solution, so it sits in the regime where the weighted norm is larger than the flat one was; the early return therefore fires less often than before rather than more. It is not eliminated, and nothing yet tests it.
  * Verified on the two configs in the tree that arm a continuation solve, both at `1.0e-11`: the initial `‖F‖` rises by exactly `1/sqrt(h)` — park-convergence 0.553794 → 1.10759 on 4 cells, jardin-critical-gradient 1.23603 → 3.90868 on 10 — and **neither run's step count changes**, 3 and 5. `shestakov-nonlinear` pins `TimeMarch`, whose test is on `dY/dt` and never reaches this function.
* **Never two solvers alive at once.** `Integrator`'s cache (`PyIntegrator.hpp:16-43`) is process-global and keyed on one `(order, grid)` pair, and `residual()` calls `invalidateIfStale` on every evaluation (`SystemSolver.cpp:1255`). Two live solvers with different grids thrash it, and `getIntegrationWeights` returns a reference *into* the map `clear()` destroys. Extract grid + a `std::vector<double>` of `yJac`, destroy, then build — `PyRunner::configure`'s own discipline.
* **Adaptivity with `spatialParameters = true` must throw**, for the identical reason `Superconvergent = true` already does at `SystemSolver.cpp:1690`: a remesh redefines how many parameters there are.

## If picking this up again — suggested order

1. ~~**`Grid(std::vector<Position>)` validation**~~ — **done on this branch.** It checked only `size() >= 2`, and `Interval(a,b)` silently swaps when `a > b` (`gridStructures.hpp:28-32`), so an out-of-order list built overlapping cells and a repeated point built a zero-width one — whose `MassMatrix` is `(h/2)·RefMass`, identically zero (`Basis.hpp:548-551`), and whose `toRef` divides by `h`. Now: finite, then strictly increasing, then the same `1e-14` total-span rule the `(lBound, uBound, nCells)` constructor already applies, so `Grid_points` cannot build what `Grid_size` would reject. Three cases in `GridTests.cpp`.
2. ~~**A PTC unit test**~~, ~~**then the mesh-independent norm**~~ — **both done.** #15 added five cases driving `solveSteadyState` plus work counters (`SteadyStats`) that make the cost of a change *measurable*; this branch adds four more covering the merit function itself, which is the named `SystemSolver::steadyResidualNorm` rather than an unreachable lambda. The `sqrt(h)` scaling those measured is now divided out — see the bullet above for what that does and does not buy. Step rejection, the `KINSetMaxNewtonStep` clamp and the hard-`KINSol`-failure path are still uncovered, and so is the early return after a transfer.
3. ~~**The modal sensor** (§7)~~ — **done on this branch**, as `SmoothnessSensor.{hpp,cpp}` over a new `NodalBasis::ToModal`. It went in as designed and §7's prediction about the implementation held: the nodal→modal map is the stored `Vandermonde` inverted once per order, no quadrature. Three things came out of building it that the spike had not shown, all in `SmoothnessSensorTests.cpp`:
   * **A floored coefficient must not be *fitted*, only skipped.** A function even about a cell centre has every odd Legendre coefficient identically zero. Pinning those at the round-off floor and running the least-squares line through the resulting alternating sequence gives `s = -8.3` at `k = 6` for `|x|^{4/3}` — a *negative* rate, i.e. the opposite of the truth, on the sharpest feature in the tree. Skipping them gives **2.4047**, against the `j^{-(a+1)} = j^{-7/3} = 2.333` that theory predicts for `|x|^{4/3}`. That 3% agreement is the strongest evidence here that the fit measures decay rather than merely avoiding the defect.
   * **The floor is `(k+1)·eps`, not `eps`,** and it has to be: each `û_j` is a length-`(k+1)` dot product, so its round-off is bounded by `(k+1)·eps·scale`. At one epsilon exactly one structural zero survived the filter (`û_1/scale = 2.56e-16`) and that one mode caused the whole defect above. The two populations are eleven orders apart, so the exact position in the gap does not matter — but being on the correct side of it does.
   * **§7's caveat about low `k` applies to the decay rate too, not only to `S_K`.** At `k = 2` neither indicator separates `exp(x)` from a singular function, and both get the *sign* wrong — the singular one reads as smoother on both measures. Two modes cannot see a singularity whichever quantity is formed from them. **§10 found the specific mechanism, and it is worse than blindness:** a two-point fit cannot tell a spectral *knee* from slow decay, and a zero-flux axis puts a knee on the first cell of essentially every problem here — so the failure is a systematic false positive at exactly the boundary a driver would be interrogating.

   What does hold, and is the real argument for the rate: on one unchanged function `S_K` runs 3.3e-3 / 8.0e-7 / 3.8e-11 at `k` = 2 / 4 / 6 — nearly eight orders — while `s` runs 2.04 / 3.63 / 5.39. `S_K` is an energy share with no fixed scale, so any threshold on it is a per-degree quantity; `s` is an exponent, so a rule like "`s < 4` is rough" means the same thing at every degree. Persson & Peraire's `S* ~ 1/k^4` is exactly such a per-degree calibration, and for shock capture rather than for this.

   The localisation claim reproduces end to end: `x^{4/3}` on 10 cells at `k = 4` gives decay rates 3.93, 6.73, 7.83, … 10.65 — monotone away from the singular cell, which is cell 0, the one `ANALYSIS.md` §7 identifies.
4. ~~**Global-`k` selection by Giorgiani's rule**~~ — **built, as #17, on a
   branch off `main` rather than off here**, so none of it is on this branch.
   The spike's case for it held: 2.8e-9 at 90 DOF against 2.0e-6 at 128, and no
   per-cell machinery. §2's conclusion was acted on rather than merely recorded —
   `Superconvergent` is required and defaulted on, and asking for it off is a
   configuration error. The state transfer between levels is #16, and needed no
   adaptivity-specific code. Three things came out of building it that the spike
   had not shown:
   * **The error scale cannot be purely relative.** `LinearDiffusion`'s exact steady state is `u = 0`, so the ratio was round-off over round-off — `1.6e-16 / 2.6e-15`, a meaningless 6.2e-2 — and the loop climbed to the degree ceiling on a problem it had solved exactly at `k = 1`. A `> 0` guard does not catch it. `Absolute_tolerance` is the right floor and costs no new key, and on a problem with a real solution it changes nothing (NonlinDiffTest 1.510e-2 → 1.508e-2).
   * **Order the non-finite check before the stopping test.** A NaN compares false against everything, so `!(E > eps)` is *true* for one, and the rule reported a solve that produced garbage as converged.
   * **"Steady solves only" has to be checked against `solvesForSteadyState()`, not against the `SteadyStateSolver` key.** That key defaults to `PseudoTransient` and is only consulted once termination is *armed*, by the presence of `SteadyStateTolerance` — so a config that simply never set it passed validation and time-marched every level, each one restarting from the last one's final state and integrating the interval again. Measured 7.5% off a fixed-degree run on NonlinDiffTest at `k = 4`. Anything that adapts between solves needs the same guard, this one included.
5. **A sensor-driven moving mesh at fixed DOF** — §8, §9 and §10. The decision is
   measured in §10 and the mesh is now a config option, so this is a driver, not new
   numerics. `error = 0.0487 h0` on
   Shestakov, set by the width of the one cell touching the singularity and by
   nothing else — not by the cell count, which is what makes this a redistribution
   rather than a refinement. **14900× at an identical 10 cells and 60 DOF**, against
   4.0× for spending 4× the DOF on uniform refinement. Needs §7's sensor to pick the
   cell (§3's indicator is blind here), a global `k` of 4 or more, and a loop that
   treats a failed grading as a rejected step, because §9's ceiling is the solver
   before it is the rule. The target is forgiving: error is monotone in the grading
   with no optimum to hit, so "grade as hard as the solver tolerates" is the rule.
   Do **not** reach for h-refinement to get it — §9 records why the fixed-DOF version
   is a far smaller change, and warm starting is worth 5–6× on cost but nothing on
   reach.
6. **Per-cell `p`** — scoped below, and **gated no** by §8. Kept as a scoping
   record; the structural finding in it is worth having if paper II's HDG+ family
   ever revives the question.

Two side quests this turned up, both worth doing regardless of AMR, and both
now merged:

* PR **#13** — restart use-after-free, live today with no adaptivity involved.
* PR **#14** — `getDerivative`. `PyRunner` still binds nothing for `sigma`,
  `lambda` or `aux`, which will matter for `nAux > 0` and for the scalars.

## Per-cell degree — scoped, and gated *no*

The design, at the depth the earlier sections were measured to. Nothing here is
built and — after §8 — **nothing here should be built for adaptivity**. Read "The
gate" immediately below for the measurement that closed it. What is kept is the
scoping: it is an accurate account of what per-cell degrees would cost, and the
structural finding in it is the useful part if the question ever returns for
another reason.

### The gate — asked, and answered: no

**§8 ran this and per-cell `p` does not clear it.** The measurement is there; the
result is that on Shestakov the whole error is `0.0487 h0`, set by the width of
the one cell touching the singularity and nothing else — so the smooth region's
degrees, which are all a per-cell rule could economise, were measured to buy
*nothing at all* (identical errors to every digit across 1, 3 and 5 outer cells at
`k = 5`, and across 5 to 40 outer cells at `k = 2`). A few percent of DOF, against
the 3× this section asked for.

What cleared it instead was `h`: a hand-graded mesh at **14900×** the accuracy of a
uniform one at matched DOF, and the loop that would find such a mesh needs §7's
sensor rather than the accuracy indicator §3's loop used. That is the next thing to
build, and it is a much smaller change than what follows here.

**So the rest of this section is kept as a scoping record, not as a plan.** It is
still the right analysis of what per-cell degrees would cost, and the structural
finding below — that the global condensed system is `k`-free — is worth having
whenever this comes back, e.g. for paper II's HDG+ family, which `TODO` records as
blocked on the same single-`k` assumption. It should not be started for adaptivity.

### What would have justified beginning

Global `k` reached 2.8e-9 at 90 DOF in two iterations and 3060 physics visits
(§6). Adaptive `h` at 128 DOF reached 2.0e-6 for 16672. So the *global* degree
already collects most of what is available on the benchmarks in this tree, and
per-cell degrees have to beat it rather than beat uniform refinement.

There is exactly one case here that global `k` cannot serve, and it is the one to
measure: **Shestakov**, where §6 recorded the error falling 19× and stopping as
`k` went 2 to 12 on 10 cells, against eleven orders for `AdjointPoster`.
`x^{4/3}` caps the regularity, so raising `k` uniformly buys almost nothing —
while §7 showed the sensor localising that singularity to **one cell** with a
2.1e24 ratio across the domain. A degree raised everywhere *except* there, with
that cell refined instead, is the classical hp case and this tree has a benchmark
sitting in it.

So the gate is a spike, in Python, with no core change — the same way Phase 0
ran. Per-cell degrees cannot be emulated directly through `Grid_points`, but the
*payoff* can be bounded without them: solve Shestakov over a sequence of
`(mesh, global k)` pairs chosen by hand from the sensor's ranking — refine only
the singular cell, raise `k` globally — and compare error against DOF and against
physics visits with the best global-`k` result at matched cost. A hand-chosen
sequence is at least as good as any rule would pick, so what comes out is an
upper bound on per-cell `p`. **If the bound is under about 3× it is not worth 200
changed sites**, and per-cell degree should stay unbuilt with that number recorded here.

The reason this is a bound rather than the answer: a graded mesh at uniform `k`
spends degrees on the smooth cells that per-cell `p` would spend only where they
pay, so the emulation is *pessimistic* on cost and optimistic on nothing. If it
cannot show a win, per-cell `p` will not either.

### Why it is cheaper than the note it replaces said

The previous version of this line called it "the ~320-site blocker". That count
is real but it is the wrong measure, and the structural fact underneath is much
more favourable:

**The global condensed system does not involve `k` at all.** `K_global`,
`L_global` and `HGlobalMat` are `nVars*(nCells+1)` square (`SystemSolver.cpp:280-284`),
each cell contributes through fixed `2x2` blocks (`:1092`, `:536`), and a cell
touches exactly two faces whatever its degree. So `solveHDGJac`'s static
condensation, `solveJacEq`'s Woodbury pass for the global scalars, and the size
and sparsity of the system actually solved are **untouched** by per-cell degrees.
That is the property HDG is chosen for, and it holds in this code rather than
merely in the literature.

What varies is per-cell and dynamically sized already: `MX` is
`Eigen::MatrixXd M(localDOF, localDOF)` built inside the cell loop
(`SystemSolver.cpp:327`, `:1483`), `CE_vec` is `localDOF x 2*nVars` (`:348`),
`CG_cellwise` is `2*nVars x localDOF` (`:493`). None of those needs a new shape,
only a `localDOF` that is a function of the cell.

Two more pieces are already per-cell and were not expected to be:

* **`DGApproxImpl` holds a `std::vector` of one `VectorWrapper` per cell**, built
  as `block_data + i * stride` of length `k + 1` (`DGApprox.hpp:34-46`). The
  per-cell views exist; they are simply all given the same length and a constant
  stride. Replacing that with an offset table is a change to two constructors and
  `Map`, not to every reader.
* **`Integrator`'s weight cache is already keyed on `(order, Interval)`** —
  `std::map<std::pair<unsigned int, Interval>, Vector>` (`PyIntegrator.hpp:16`).
  It can hold several orders at once today. Only `cachedOrder`, the scalar
  `invalidateIfStale` compares against, assumes one.

### What actually blocks it

Counted rather than estimated. `grep -c '(k + 1)\|(k+1)'` over the non-test tree:
**162 in `SystemSolver.cpp`, 29 in `Matrices.cpp`, 15 in `DGSoln.hpp`, 10 in
`AdjointVectors.cpp`, 1 in `Basis.hpp`** — 217 sites, against 909 for the looser
`k *+ *1` pattern that also catches `2*k+1` and friends. Most are a local
`(k+1)` inside a cell loop and become `(k_i+1)`; the ones that are not are the
work:

1. **`DGSolnImpl` holds `const Index k` and `const BasisType Basis` by value**
   (`DGSoln.hpp:19`). Per-cell needs a vector of orders and a vector of bases —
   cheap, since `BasisType::getBasis(Order)` is a flyweight, but `getBasis()`
   returning a reference into the `DGSoln` is a documented use-after-free trap
   (CLAUDE.md) and a per-cell version multiplies the ways to get it wrong.
2. **`getDoF()` is a closed-form formula assuming uniform `k`**
   (`DGSoln.hpp:26-33`), and the same expression is duplicated at
   `Solver.cpp:132`, `NetCDFIO.cpp:299`, `MaNTA.cpp:127` and `PyRunner.cpp:73` —
   the duplication the AMR plan already flagged. It becomes a prefix sum, and
   **that sum wants one owner** rather than five.
3. **`localDOF` is one `size_t` member** (`SystemSolver.hpp:774`) used as a
   constant stride at `:935`, `:938` and `:1108`. It becomes an offset table, and
   `Y + nCells * localDOF` — where `lambda` starts — becomes the table's last
   entry. Getting this wrong is the failure CLAUDE.md names as the most common way
   to break the solver silently, so it wants its own test before anything else
   moves.
4. **The restart format carries one `PolyOrder`** as an `NcInt` in the grid group
   (`NetCDFIO.cpp:152`, read back at `SolverConfig.cpp:319`). It needs a per-cell
   array, and the scalar should stay for a uniform run so existing files keep
   loading — the round-trip regression cases compare bit for bit.
5. **`Postprocessing` builds its operators once per order** in
   `initialiseMatrices`, and `Superconvergent` evaluates the physics on `k+2` star
   nodes per cell. Both become per-order sets. `printSources` already picks its
   basis and stride from the flag for exactly this reason and would need the same
   treatment per cell.
6. **`Integrator::invalidateIfStale`** — see above; the map is fine, the scalar
   `cachedOrder` is not.

### Order of work

Each step is landable and testable on its own, and the first three are worth
having whether or not per-cell degrees are ever built:

1. **One owner for the DOF count.** Replace the five copies of the layout formula
   with a single function, still returning the uniform answer. Pure refactor, and
   it is what makes everything after it a change in one place.
2. **An offset table behind `localDOF`**, filled with the uniform values. The
   layout is unchanged, bit for bit, and the regression suite is the guard;
   what changes is that the stride is now looked up rather than multiplied.
3. **A per-cell order vector in `DGSoln`**, all entries equal. Same again: no
   behaviour change, and `copy()`'s existing refusal on a different order becomes
   a refusal on a different *vector*.
4. **Let the entries differ.** This is the real step, and it is where `MX`,
   `CE_vec`, `CG_cellwise` and the physics evaluation loops become per-cell. The
   global solve does not move (see above), which is what makes it finite.
5. **The hp rule**, on top of the sensor already on this branch: Woopen §4.3's
   switch — decay rate `s` above a threshold means raise `p`, below it means
   refine `h`. §7 measured `s < 4` separating the two benchmarks here, and
   recorded honestly that the margin is 1.6× and that at `k = 2` neither
   indicator separates anything.

### What must refuse rather than guess

* **Spatial adjoint parameters.** `G_p` is `(ng * nCells * (k+1), np)`, indexed by
  node, so varying degrees redefine how many parameters there are. `Superconvergent`
  already throws for this at `SystemSolver.cpp:1690` and `runAdaptiveDegree` throws
  for the same reason; a third instance of the same objection wants the same answer.
* **A restart whose per-cell orders do not match the run's** — until a transfer
  exists for it. #16 projects across a *uniform* degree change; the nesting
  argument it rests on is per cell, so it extends to per-cell degrees on the same
  mesh but not to a remesh.

### Why not `h` — **retracted for Shestakov by §8**

This section argued that `h` was not the sequel. On Park and `AdjointPoster` that
still holds and the measurements behind it are unchanged: §3 has Park at 1.35×
error for 3.2× the visits and `AdjointPoster` at 1.8–2.6× for 2.8×, and §6 has one
degree bump beating the whole h-adaptive machinery by seven orders on the latter.
Both are smooth problems, and `p` is the right lever on a smooth problem.

**On Shestakov it is wrong, and §8 measures it as wrong by 14900×.** The error
argued from here — §3's 27% DOF saving for 44% more visits — was the
*equidistribution loop's* result, and §3's own second measurement trap says why
that loop underperformed: the accuracy indicator cannot see this error, because it
is one global mode living in `sigma` and the indicator finds where error is *made*
rather than where it lands (rank correlation +0.41). Reading that as a statement
about `h` rather than about the indicator was the mistake, and it survived here
because no hand-graded mesh had been tried.

What §8 measures instead: `error = 0.0487 h0`, set by the width of the one cell
touching the singularity and nothing else, so grading geometrically towards it
converts an algebraic `O(1/N)` into an exponential. §7's sensor localises that cell
with a 2.1e24 margin, which is what the indicator could not do. The rule needs no
target-size formula, and it is the *smaller* of the two changes rather than the
narrower use this paragraph imagined.
