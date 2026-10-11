# MeshAdaptation from non-uniform meshes — design

**Status:** built, all three phases. `docs/adaptivity.rst` describes the
result and `MESH-REFINEMENT.md` §14 has the measurement. **Date:** 2026-10-08.

The build departs from this design in five places, each forced by that
measurement:
- **A graded start is not regraded as a recipe.** Like an explicit start,
  each rough end cell is split into a layer of the configured size, nothing
  else moves, and the count grows. Two recipe versions were built and measured
  first:
  - Squaring the ratio moved every face of the layer. On Shestakov with a 0.2
    layer that widened the cell holding the source edge at x = 0.1, and the
    regrade came out 15–19× worse than degree adaptation alone.
  - Moving bulk cells into the layer kept the faces but left layers too shallow,
    up to 7000× worse on the wall layer.

  The split reaches the same wall cell as squaring, without moving anything.
- **The neighbour test has a guard.** It also needs the end's own rate below
  half the measurable ceiling. Without it, smooth ends whose top mode sits just
  above the floor read up to 1.33×, against 1.36× for the mildest singular end,
  and no margin separates them.
- **The margin is 0.2, not 0.1.** Guarded, smooth graded ends read at most
  1.00, so anything in `[0, 0.36)` works.
- **The wall-cell floor is 1e-5, not 1e-6.** At 1e-6 the sensor cannot tell
  discretisation error from a singularity.
- **The exact-function numbers below were fitted to `P_j` coefficients.**
  MaNTA's basis is orthonormal, which shifts rates by about +0.4 at `k = 4`
  (3.98, not 3.58). The wall/neighbour ratios become 1.69 / 1.42 / 1.16, not
  1.77 / 1.47 / 1.18. The conclusions are unchanged.

`MeshAdaptation` runs p → h → p from a *uniform* sampling mesh. It refuses
`GradedGridBoundary` and `GridPoints`, and since the restart fix on
`fix/nonuniform-ladder-restart` the driver also refuses a non-uniform mesh it is
handed directly, which is how a restart that keeps its file's mesh reaches it.
This designs lifting both refusals in three phases: a sensor change both
phases need (0), graded recipe meshes (1), then explicit meshes (2).

The constraints come from `MESH-REFINEMENT.md` and are not up for redesign:

* **§8:** on a mesh graded towards a singularity the error is `0.0487 h0`. It is
  set by the width `h0` of the wall cell and by nothing else, so moving cells
  beats adding them. The error is monotone in the grading, with no optimum, so
  the rule is to grade as hard as the solver tolerates.
* **§3:** the per-cell accuracy indicator (`u* − u_h`) is **blind on
  Shestakov**, where the error is one global mode in `sigma`. Only the modal
  sensor (§7) finds the place. Nothing below localises with the indicator.
* **§9:** the ceiling is the solver, at `h0/span` around 1e-6 to 1e-7. A
  grading that fails to solve is a rejected step.
* **`gradingLayerFraction`:** grading only inside the flagged end cell keeps
  the faces the sampling mesh had. Moving Shestakov's face at `x = 0.1` raised
  the error from 1.34e-6 to 1.58e-2.

## What changes when the cells are not alike

The sensor's decay rate `s` is the fitted exponent of the per-cell Legendre
spectrum, at `k = 4`, as a function of cell width `h`. These were computed for
this design from exact functions, not solver output: each was interpolated at
the `k + 1` Chebyshev points of the first kind, as MaNTA's basis is, converted
to Legendre coefficients, and fitted for `log|û_j| = c − s log j` over
`j = 1..k`:

| function, cell at | h = 0.2 | 0.1 | 0.05 | 0.01 | 1e-3 | 1e-4 |
|---|---|---|---|---|---|---|
| `x^(4/3)` at the wall (singular) | 3.58 | 3.58 | 3.58 | 3.58 | 3.58 | 3.58 |
| `(4 − 3x²)^(1/3)` at `x = 1` (the pinch benchmark) | 4.30 | 5.37 | 6.59 | 9.75 | 14.5 | 16.5 |
| `cos 2x` at `x = 0` (smooth) | 5.86 | 7.00 | 8.15 | 10.8 | 13.6 | 1.58* |

\* Round-off. My quick fit floors at `1e-14`, where `cellSmoothness` floors at
`(k+1)·eps` and skips floored modes, so this one cell is not evidence about
MaNTA's sensor. It is evidence that very narrow smooth cells reach the floor,
which Phase 0 must check with the real sensor.

Three consequences:

1. **A singular cell's rate does not depend on its width.** The spectrum of
   `x^α` on `[0, h]` is `h^α` times its spectrum on `[0, 1]`, so the rate is
   identical. The sensor therefore keeps flagging a singular end however hard
   it is graded: **it says where to grade, never when to stop.**
2. **A smooth cell reads smoother as it narrows.** So today's comparison,
   `interior median / end rate`, is biased once widths vary. Take a mesh graded
   at one end: the narrow layer cells raise the interior median, and the wide
   end cell at the *other* end reads rough by comparison. On the pinch profile,
   width alone gives 9.75 / 4.30 = **2.27**, which is above the threshold of 2.0.
   That is a false positive caused by nothing but the mesh.
3. **Inside a geometric layer, the wall-to-neighbour ratio depends only on the
   grading ratio `r`, not on `h0`.** For `x^(4/3)`: 1.77 at `r = 0.5`, 1.47 at
   0.3, 1.18 at 0.1, identical at `h0` = 1e-2 and 1e-4. This follows from
   self-similarity. For a smooth function the narrower wall cell reads
   *smoother* than its neighbour, so the ratio is below 1. That sign difference
   is a usable signal.

## Phase 0 — a width-aware decision (prerequisite for 1 and 2)

Generalise `gradingDecision(cells, k, threshold)` to take the cell widths, and
choose the reference for each end separately:

* **Width-matched peers.** Use the median over interior cells within a factor
  of 2 of the end cell's width, when there are at least 3 of them. On a uniform
  mesh every interior cell qualifies, so **every verdict on a uniform mesh is
  unchanged**. That is pinned by running the existing `MeshAdaptationTests`
  cases unmodified.
* **Otherwise, a sign test against the inward neighbour.** This applies when
  the end cell is narrower than its neighbour by at least 1.5×, i.e. a graded
  end. The end is singular when `s_neighbour / s_end > 1 + margin`. A smooth
  function gives a ratio below 1 there, because the narrower cell is smoother,
  so the test is a sign test plus a margin rather than a calibration. The
  margin must sit below 1.18, the `r = 0.1` singular value, so `r` below about
  0.1 is refused or clamped.
* **Otherwise undecidable:** leave that end alone and say so. This covers a
  wide end with no peers, as on an arbitrary explicit mesh.

**Measurement before merging.** The exact-function numbers above must be
reproduced on solver output, i.e. `u_h` rather than `u`:
- problems: Shestakov (singular), Park/AdjointPoster (smooth), the `n = 2.5`
  wall layer (§12), and the thermodiffusive pinch (smooth, branch point 0.155
  outside the wall: the nearest thing to a false-positive trap in the tree);
- meshes: uniform, graded Lower/Upper/Both at `r` = 0.1, 0.3 and 0.5, and two
  random explicit meshes;
- degrees: `k` = 3–6.

Acceptance:
- no smooth problem graded on any mesh;
- every singular end flagged at every grading depth;
- uniform verdicts bit-identical to today;
- the `cellSmoothness` floor behaving at `h/span` = 1e-6.

The `margin` is set from that table, and it is the one number this phase
introduces.

## Phase 1 — `GradedGridBoundary` as the starting mesh

The refusal of `GradedGridBoundary` with `MeshAdaptation` is lifted. The
configured grading becomes the sampling mesh, and **the output is still a
recipe**: the same `GridSize`, with `GradingEnd`, `GradingCells`, `GradingRatio`
and the layer fraction re-chosen. That keeps it compatible with restarts and
with `GridLadder`, which rescales recipes.

Sample at `k0 ≥ 3` on the graded mesh, decide each end with Phase 0, then act:

| end | sensor says | action |
|---|---|---|
| graded | singular | **Grade harder** at the same count: lower `r` so `h0` shrinks, holding the layer fraction and cell split. This is the "split, repeat" of §8, one step per run. |
| graded | smooth | Keep it. The user asked for it, and grading costs little at a smooth end. Report it. |
| ungraded | singular | Add that end (`Lower` → `Both`), splitting cells by the existing fill rules. |
| ungraded | smooth | Nothing. |

Failure handling reuses `MeshAdaptationAttempts`, softening `r` back towards
the *configured* ratio, never past it. The fallback is therefore the starting
mesh, which is known to solve rather than merely assumed to. The degree loop
then runs on the winner, as now.

**Depth is one step per run, as it is from a uniform start today.** More steps
per run is follow-up work, and the same for both starts: fit `C` in
`err ≈ C h0` from two levels using the *global* `u* − u_h` estimate, then pick
`h0 = tol / C`, clamped at `h0/span ≥ 1e-6`. That needs the global estimate to
track the true error on Shestakov. §3 only shows the *per-cell* split is wrong
there, so the global estimate must be measured before it is relied on.

## Phase 2 — explicit meshes (`GridPoints`, a kept restart mesh, a previous run's output)

**The contract is that every given boundary stays.** A list does not say why
its faces are where they are, and moving one is what cost Shestakov four
orders. So the fixed-budget redistribution that wins on recipes is **not
offered**: it can only be had by deleting user faces.

The operation instead **subdivides the flagged end cell into a geometric
layer**:
- The new mesh is the given boundaries plus `m − 1` inserted points inside the
  wall cell. The count grows by `m − 1`, and `h0` becomes `H r^(m−1)`.
- `m` is `GradingCells` when given, otherwise as many as cost no more rounds
  (the `PhysicsParallelism` fill), otherwise 4. With `r = 0.3` that is a 37×
  smaller wall cell for three extra cells.
- A new pure function, `subdividedEndPoints(Grid, end, m, r)`, sits beside
  `gradedMeshPoints` and is tested the same way:
  - every original boundary present bit for bit;
  - monotone;
  - `h0` exact.

Already-graded explicit ends get the same treatment. A previous
`MeshAdaptation` output, resumed from its restart file with no `GridSize`, is
split again inside its wall cell, which nests. The §8 iteration therefore
becomes **restart → adapt → restart**: each pass is a separate run,
warm-started from a converged state, the most robust form available against
§9's ceiling. The mesh each pass ends on is written to the restart file, which
the restart fix now keeps, and is logged as a `GridPoints` line that can be
pasted into a config.

Configuration changes:
- the `GridPoints` refusal and the driver's uniformity guard are replaced by
  this mode;
- `GridLadder` stays refused with `MeshAdaptation`, as now;
- no new keys: `GradingCells` and `GradingRatio` mean "how to split the wall
  cell".

## Not covered, and why

* **Interior features.** The sensor reads only the two end cells, so a face a
  user placed at an interior source edge is kept (Phase 2) but never refined.
  Interior grading is a different sensor question.
* **Several variables.** `runAdaptiveMesh` reads variable 0 only. For the
  coupled benchmarks (`n`–`T`, `Ti`–`Te`) the natural rule is the roughest
  variable per end. That is independent of this design and should be done
  first if a multi-channel problem needs grading.
* **Per-cell `p`.** Gated "no" by §8; nothing here reopens it.

## Order and cost

Phase 0 is mostly measurement: one sweep script, and the rule change behind a
test that the uniform verdicts are identical. Phase 1 is small once Phase 0
lands, because it is a lifted refusal plus the decision table over the
existing attempt loop. Phase 2 is the largest: a count-changing mesh
operation, the fill, restart output, and a pasteable `GridPoints` log line.
Each phase is independently useful and mergeable: Phase 1 alone covers every
`GradedGridBoundary` user, and Phase 2 alone covers resuming a previous
adaptation.
