Adaptivity: the indicators, and the p → h → p driver
====================================================

MaNTA can choose its own polynomial degree and its own mesh grading. This page
says what the three quantities it measures actually *are*, which one drives which
decision, and why the sequence runs in the order it does.

Everything here was measured on the benchmarks under ``python-examples/``;
``MESH-REFINEMENT.md`` in the repository root carries the numbers, the retractions
and the failures. Where a claim below has a margin, it is stated.

Every new mesh and every new degree is a new evaluation plan for the physics case,
so a case adapted this way either declares that it follows one in place or runs
with ``RebuildPhysicsOnRegrid = true``; otherwise the run is refused before it
starts. See :ref:`evaluation-plans`.

.. _adaptivity-quantities:

The three quantities
--------------------

All three are per cell, and they answer different questions. Mixing them up is the
easiest mistake here, so the distinction comes first: **two of them say where the
error is, and one says whether spending more degree will help.**

.. list-table::
   :header-rows: 1
   :widths: 22 20 58

   * - Quantity
     - Kind
     - Answers
   * - Accuracy indicator :math:`E_K`
     - *a posteriori*
     - How large is the error in this cell, now?
   * - Modal energy fraction :math:`S_K`
     - *a priori*
     - What share of this cell's content is in its finest mode?
   * - Decay rate :math:`s`
     - *a priori*
     - How fast does this cell's spectrum fall — i.e. how smooth is the solution
       here?

The accuracy indicator :math:`E_K`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Capasso *et al.* equation (15). Per cell,

.. math::

   E_K^2 = \frac{\lVert u^* - u_h \rVert^2_{L^2(K)}}{\lvert K \rvert}

the gap between the solution and its own postprocessing, divided by the cell width
so that cells of different sizes are comparable. :math:`u^*` is the degree-\
:math:`(k+1)` reconstruction described in :doc:`superconvergence`, built on every
run with :math:`k \ge 1`.

This is what ``DegreeAdaptation`` drives from, aggregated over the mesh. It is
*a posteriori*: it measures the error that is there, using no knowledge of the
solution beyond the two approximations MaNTA already has.

.. warning::

   :math:`E_K` **needs** ``Superconvergent = true``. The whole quantity rests on
   :math:`u^*` being the better of the two approximations, and plain interpolatory
   HDG loses that. Measured on Park, the Spearman rank correlation between
   :math:`E_K`'s per-cell ranking and the true per-cell error's is **+0.11 to
   +0.40 with the flag off** against **+0.98 to +1.00 with it on** — barely better
   than chance against essentially exact. ``DegreeAdaptation`` therefore turns the
   flag on, and refuses a configuration that asks for it off.

The modal energy fraction :math:`S_K`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Persson & Peraire's smoothness sensor, as used by Woopen §4.3 and identical to
Capasso equation (13). Writing :math:`w_H` for the :math:`L^2` projection of a
cell's solution onto :math:`P_{k-1}`,

.. math::

   S_K = \frac{(w - w_H,\, w - w_H)_K}{(w,\, w)_K}

In a Legendre basis that projection just drops the top coefficient, so with
:math:`\lVert P_j \rVert^2 = 2/(2j+1)` the whole thing collapses to a ratio of
weighted squares of the modal coefficients :math:`\hat u_j`:

.. math::

   S_K = \frac{\hat u_k^2 / (2k+1)}
              {\sum_j \hat u_j^2 / (2j+1)}

No quadrature, and no factor of :math:`\lvert K \rvert` — the :math:`h/2` from the
reference map cancels — so it is directly comparable across a non-uniform mesh.

.. note::

   **MaNTA does not threshold** :math:`S_K` **, and the reason is measured.** It is
   an energy *share*, so it has no fixed scale and any threshold on it is a
   per-degree quantity. Between a smooth benchmark and a singular one the
   separation is **389× at k = 4 and only 2.2× at k = 2**; on one unchanged
   function :math:`S_K` moves from 3.3e-3 to 3.8e-11 as :math:`k` goes 2 → 6,
   nearly eight orders, while the decay rate below moves 2.04 → 5.39. Persson &
   Peraire's own :math:`S^* \sim 1/k^4` is a per-degree calibration of exactly this
   kind, and it is tuned for shock capture — it sits orders of magnitude above
   anything seen here. :math:`S_K` is reported by the sensor and is useful for
   inspection; the decisions are driven from the rate.

.. _adaptivity-decay-rate:

The decay rate :math:`s`
~~~~~~~~~~~~~~~~~~~~~~~~

**This is the quantity the grading decision is made from, so it is worth being
precise about.** After Mavriplis: within each cell, take the modal coefficients
:math:`\hat u_j` of the solution, and fit a straight line to their magnitudes on
log–log axes against the mode number,

.. math::

   \log \lvert \hat u_j \rvert = c - s \log j,
   \qquad j = 1 \ldots k

by least squares. :math:`s` is the fitted slope, negated so that **larger means
smoother**. The name is literal: it is the exponent of the algebraic rate at which
the spectrum decays, since the fit asserts
:math:`\lvert \hat u_j \rvert \sim j^{-s}`.

Why an exponent rather than a magnitude: **its scale is fixed**. A function with an
:math:`x^{a}` singularity has Legendre coefficients falling like
:math:`j^{-(a+1)}`, so :math:`s` estimates :math:`a+1` — a property of the
*solution*, not of the discretisation, the units, or the degree. That is what lets
a rule like "rougher than the interior by 2×" mean the same thing at every degree
and on every problem, where a rule on :math:`S_K` or :math:`E_K` cannot.

The theory is checkable and checks out: for :math:`\lvert x \rvert^{4/3}` the
predicted rate is :math:`j^{-7/3}`, i.e. :math:`s = 2.333`, and the sensor measures
**2.4047** — 3% agreement, which is the strongest evidence in the tree that the fit
measures decay rather than merely producing a number.

Two values are reported rather than fitted, and they are the two ends of the range:

``infinity``
   The top mode is at the round-off floor, so the cell's solution is exactly
   representable below degree :math:`k` — a constant, or any polynomial the space
   already contains. Nothing left to resolve, and the decision rule reads this as
   perfectly smooth. This is the normal answer on a problem whose steady state is
   low-order, such as ``jardin-critical-gradient``.

   It is not larger than every finite rate, though, and the decision rule does not
   treat it so. The largest rate that can be *measured* at degree :math:`k` is the
   one whose top mode sits on the floor, :math:`s_{\max} = -\log((k+1)\epsilon) /
   \log k` — 31.5 at :math:`k = 3`, 24.8 at 4, 16.3 at 8 — and a cell just above
   the floor fits a rate just under it. Both are round-off, so every rate is
   compared capped at :math:`s_{\max}`.

``zero``
   Only the top mode is above round-off, so the spectrum has no decay in it at all.
   As rough as the sensor can report.

.. _adaptivity-floor:

Two traps in the fit, both hit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Neither is obvious and both change the sign of the answer, so they are recorded
here rather than only in the source.

**A coefficient at the round-off floor must be skipped, not fitted.** A function
that is even about a cell's centre has *every odd Legendre coefficient identically
zero*. Pinning those at the floor and running the least-squares line through the
resulting alternating sequence gives :math:`s = -8.3` at :math:`k = 6` for
:math:`\lvert x \rvert^{4/3}` — a **negative** rate, i.e. the exact opposite of the
truth, on the sharpest feature in the tree. Skipping them gives the 2.4047 quoted
above.

**The floor is** :math:`(k+1)\varepsilon` **of the cell's largest coefficient, not**
:math:`\varepsilon`. Each :math:`\hat u_j` is a length-\ :math:`(k+1)` dot product,
so its round-off is bounded by :math:`(k+1)\varepsilon` times the scale of the
data. At one epsilon exactly one structural zero survived the filter
(:math:`\hat u_1 / \text{scale} = 2.56 \times 10^{-16}`) and that single mode caused
the whole defect above. The two populations are eleven orders apart, so the exact
position in the gap does not matter — but being on the correct side of it does.

.. _adaptivity-low-k:

Why the decision needs :math:`k \ge 3`
--------------------------------------

``MeshAdaptation`` **refuses** ``PolynomialDegree < 3``. Not a warning, because
below it the verdict is not merely uncertain — it is *reversed*.

Measured with the rule described below, on a uniform mesh of 10 cells:

.. list-table::
   :header-rows: 1
   :widths: 34 16 16 16 16

   * - Problem
     - k = 2
     - k = 3
     - k = 4
     - k = 5
   * - Shestakov — *wants grading*
     - Lower, 2.69×
     - Lower, 3.09×
     - Lower, **6.80×**
     - Lower, 6.44×
   * - Park — *wants none*
     - **Lower, 3.63×**
     - uniform, 0.99
     - uniform, 1.19
     - uniform, 1.01
   * - Jardin — *wants none*
     - uniform, 0.97
     - uniform, 1.00
     - uniform, 0.97
     - uniform, 1.15

At :math:`k = 2` Park's roughness ratio *exceeds* Shestakov's, so **no threshold
works**: any rule that grades the singular problem also grades the entire one, and
harder. From :math:`k = 3` the two populations separate — 3.09–6.80 against
0.97–1.19 — so anything in (1.2, 3.0) works, and at :math:`k \ge 4` there is a
factor of 5.7 of headroom.

The mechanism is specific, and it is why raising the floor fixes it. The first
cell's spectra:

.. list-table::
   :header-rows: 1
   :widths: 28 18 18 18 18

   * -
     - :math:`\hat u_1`
     - :math:`\hat u_2`
     - :math:`\hat u_3`
     - :math:`\hat u_4`
   * - Shestakov
     - 4.76e-02
     - 6.81e-03
     - 1.90e-03
     - 9.72e-04
   * - Park
     - 7.92e-03
     - 2.63e-03
     - **7.91e-06**
     - 1.14e-06

Shestakov falls by 7.0×, 3.6×, 2.0× — slow and algebraic all the way out. Park
falls by 3.0× and then **332×**: its spectrum has a *knee*, and the honest decay
only begins at :math:`j = 3`. A two-point fit over :math:`j = 1, 2` sees the knee
and reports :math:`\log(2.63/7.92)/\log 2 = 1.575`, which is what was measured.
Nothing about roughness.

.. important::

   **The knee is not an accident of that problem — it is what a Neumann axis
   does.** Park is even about :math:`x = 0` and the axis is zero-flux, so the
   solution is locally flat there and its *linear* content on the first cell is
   suppressed relative to its quadratic content. Every MaNTA problem with a
   zero-flux axis has that, **on the very cell a driver is interrogating**. So the
   failure at :math:`k = 2` is not random bad luck on a smooth problem: it is
   systematic, in the direction of a false positive, exactly where the question is
   being asked.

.. _adaptivity-h0:

What grading buys, and what limits it
-------------------------------------

On Shestakov's problem — an :math:`x^{4/3}` singularity at the axis — the error was
measured to be

.. math::

   \text{relative } L^1 \text{ error} = 0.0487 \, h_0

in :math:`h_0`, the width of the cell touching the singularity, and to depend on
**nothing else** — not the cell count, and not whether the mesh got there by being
uniform or by being graded. The constant held to 0.5% across a 4600× range of
:math:`h_0` and across both mesh families.

That is what makes this a *redistribution* rather than a refinement, and it is the
whole argument for the driver:

.. list-table::
   :header-rows: 1
   :widths: 46 18 18 18

   * - At ``PolynomialDegree = 5``
     - cells
     - DOF
     - rel. error
   * - uniform
     - 10
     - 60
     - 4.91e-03
   * - **graded, same budget**
     - **10**
     - **60**
     - **3.29e-07**
   * - uniform, four times the budget
     - 40
     - 240
     - 1.22e-03

**Quadrupling the DOF buys 4.0×. Redistributing them buys 14900×**, for 1.5× the
physics evaluations. Extrapolating the law, matching the graded result on a uniform
mesh needs about 148000 cells.

.. warning::

   **The ceiling is the time integrator, not the discretisation, and it arrives
   early.** Past roughly :math:`h_0 / \text{span} = 10^{-6}`, IDA's corrector starts
   failing at :math:`\lvert h \rvert =` ``MinStepSize``, whose default is ``1e-7``;
   lowering it to ``1e-12`` bought exactly one further level. Below about
   :math:`10^{-7}` nothing got through — not ``MinStepSize``, not
   ``SuppressAlgebraicError``, not the tolerances, not either continuation mode —
   and below :math:`10^{-8}` ``IDACalcIC`` cannot build a consistent initial
   condition at all. Of 52 hand-built gradings, 15 failed, non-monotonically in
   everything: two meshes differing only in how many cells sat outside the layer
   gave one success and one failure.

   The law that makes grading worth doing was still holding at the last mesh that
   converged, so a run that dies here has **not** run out of accuracy to gain.
   ``MeshAdaptation`` treats a failed grading as a rejected step for that reason.

.. _adaptivity-driver:

The driver: p → h → p
---------------------

``MeshAdaptation = true`` runs the whole sequence. The order is *forced* by two
independent measurements, from opposite directions:

* grading only **pays** from :math:`k \ge 4` — at :math:`k = 2` the :math:`h_0` law
  breaks, and an 11× reduction in :math:`h_0` bought 1.19×, because the cells that
  are not the innermost stop resolving the singularity well enough to be
  negligible;
* the grading **decision** is only correct from :math:`k \ge 3`, per
  :ref:`adaptivity-low-k`.

So:

**1. p — one solve at** ``PolynomialDegree`` **on** ``GridSize`` **uniform cells.**
This is both the degree floor the decision needs and the sample it is read from.
You choose both numbers: the floor is 3, and 10 cells is already a reasonable
resolution for the sensor to work from.

**2. h — decide, and regrade at the same cell count.** Each end cell's decay rate
is compared against the **median over the interior**, and the end is graded if it
is rougher by ``MeshAdaptationThreshold`` (default 2.0). Comparing against the
interior rather than a fixed number is what makes the test scale-free and blind to
a uniform lack of resolution — which is the degree loop's business, not this one's.

Rates are compared capped at :math:`s_{\max}`, the largest measurable at the
degree. Without the cap a resolved problem gets graded. On Jardin's linear steady state at 10 cells and
:math:`k = 3`, every interior cell's top mode falls under the round-off floor and
reports :math:`\infty`, while the wall cell's lands just above it, at about 30. Taken
at face value that end is infinitely rougher; capped, the ratio is 1.05 and the
mesh stays uniform.

The regraded layer is **the end cell of the sampling mesh** unless
``LowerBoundaryFraction`` or ``UpperBoundaryFraction`` is given for that end. That
cell is the region the sensor judged rough, and keeping its inner face keeps
whatever the sampling mesh had resolved by putting a face there. Shestakov's
source switches off at :math:`x = 0.1`, where the uniform 10-cell mesh has a face.
Grading within the first cell takes the error at :math:`k = 3` from 1.36e-02 to
1.34e-06. Grading over a fifth of the domain instead moves that face off the
kink, and the error *rises*, to 1.58e-02.

The regraded mesh keeps ``GridSize`` cells. ``GradingCells = 0`` means *as many as
the budget allows*, i.e. ``GridSize - 1``, which minimises :math:`h_0`; note this
differs from what ``GradingCells = 0`` means on the manual ``GradedGridBoundary``
path, where it is half the grid. Here exactly one end is being graded and the error
is known to be :math:`0.0487 h_0`, so filling the layer is the right default —
measured, 9 cells of 10 beat 5 of 10 by 48×.

**The graded solve starts from the sample.** Its state is projected onto the
graded mesh, the way a restart onto a different mesh is, and its first
pseudo-transient step is the one the sample finished at -- so it does not climb
the SER ramp again. On the wall-layer case in ``MESH-REFINEMENT.md`` that took
the graded solve from 19 continuation steps to 2, for the same answer to
:math:`10^{-10}`. If the warm start fails, the same mesh is solved cold before
anything is softened. The degree loop's levels carry the step the same way, from
the level before.

**A grading that fails to solve is a rejected step.** The ratio is softened towards
1 and retried, up to ``MeshAdaptationAttempts`` times (default 4); if none
converges the run continues on the uniform mesh and says so at ``WARNING``. Given
the failure rate above, a driver without this would die on a third of the problems
it was pointed at.

**3. p — the degree loop.** ``runAdaptiveDegree`` on whichever mesh won, to
``DegreeTolerance`` by Giorgiani's rule. See :ref:`degree-adaptation`. Its first
level is the solve stage 1 or 2 already made on that mesh at ``PolynomialDegree``,
so a run whose degree is already good enough costs the sampling solve plus, if it
graded, one more -- and nothing is solved twice.

.. code-block:: toml

   [configuration]
   PolynomialDegree = 4          # the floor is 3; 4 puts two decaying modes in the fit
   GridSize = 10                 # the budget, and it is not increased
   LowerBoundary = 0.0
   UpperBoundary = 1.0

   MeshAdaptation = true         # implies DegreeAdaptation, and so Superconvergent
   SteadyStateSolve = true       # steady solves only

   MeshAdaptationThreshold = 2.0
   GradingRatio = 0.3
   # LowerBoundaryFraction = 0.1 # the layer, if the lower end is chosen; by
                                 # default the first cell of the sampling mesh

A run reports each stage:

.. code-block:: text

   Mesh adaptation: sampling at k = 4 on 10 uniform cells
     decay rate: lower end 2.85, interior median 19.4, upper end 19.2; measurable up to 24.8
     roughness vs interior: lower 6.80x, upper 1.01x, threshold 2.00x -> grade Lower
     layer: 0.1 of the domain, the sampling mesh's end cell
     attempt 1: ratio 0.3, narrowest cell 6.561e-06 of the domain
   Mesh adaptation: adapting the degree on the graded mesh

What it is worth, measured end to end on a problem whose steady state is
:math:`u = x - x^{4/3}` — an :math:`x^{4/3}` singularity at the axis, and linear in
:math:`u` so that Newton reaches it in one step. Ten cells throughout, so the DOF
budget is fixed and the mesh is the only thing that differs:

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * -
     - :math:`h_{\min}/h_{\max}`
     - rel. :math:`L^1`
     - gain
   * - uniform, no adaptation
     - 1.00
     - 9.43e-03
     - —
   * - p only (``DegreeAdaptation``)
     - 1.00
     - 4.04e-03
     - 2.3×
   * - **p → h → p**
     - **1.6e-05**
     - **4.26e-05**
     - **221×**

**p alone buys 2.3× and then stops** — that is the regularity cap, and it is the
same wall measured on Shestakov, where the error fell 19× from :math:`k = 2` to
:math:`k = 12` and then nothing. Grading at the same budget buys 95× *on top of*
that. Both halves of the sequence are load-bearing, and the ``h`` half is the
larger one. The decision on that problem reads a lower-end decay rate of 1.20
against an interior median of 7.83 and an upper end of 9.63 — 6.51× rougher at the
axis, three times the margin it needed.

And on Shestakov's problem itself, which is where every grading measurement above
came from — ``python-examples/shestakov-nonlinear`` with :math:`n_b = 0.05`:
**10 cells at k = 3, relative error 1.36e-02 uniform against 1.34e-06 graded —
10,000×**, again at the same cell count, for 2.7× the physics evaluations of the
uniform solve. ``python-examples/paper-figures/adaptivity.py`` reproduces it,
along with Park and Jardin left uniform at no cost.

.. note::

   That run was impossible until the continuation loop's error handling was
   corrected, and the correction is worth knowing about because it was a single
   return code. ``KINSol`` returning ``KIN_MXNEWT_5X_EXCEEDED`` (−7) — "five
   consecutive Newton steps hit the maximum length" — was treated as fatal, where
   ``KIN_MAXITER_REACHED`` was already treated as *this dt was too ambitious, damp
   and retry*. Under pseudo-transient continuation those are the same signal, and
   the second response is the right one for both.

   What that one line was worth on Shestakov, at k = 2:

   .. list-table::
      :header-rows: 1
      :widths: 30 35 35

      * - cells
        - −7 fatal
        - −7 a rejected step
      * - 5, 8, 12, 20, 25, 30, 40, 50
        - **fail**
        - converge
      * - 10
        - converge
        - converge
      * - 6, 15
        - fail
        - fail *(differently — see below)*

   And with ``Superconvergent`` on, which ``MeshAdaptation`` requires, every
   combination of k = 2…5 at 10 and 20 cells went from failing to converging.

   The diagnosis was that **every schedule lever was inert**:
   ``PseudoTransientMaxStep`` from 1 to infinity, the SER rate and floor, an initial
   ``dt`` from 1e-4 to 1e3, and the tolerance over eight orders all failed at the
   *same* continuation step with the *same* residual. A failure that does not move
   when the schedule moves was never a schedule failure.

   Two counts still fail — 6 and 15 — but with a different and softer symptom: the
   loop exhausts its 200 continuation steps rather than hitting a hard ``KINSol``
   error. Both are rescued by putting a cell boundary on the source kink at
   :math:`x = 0.1` through ``GridPoints``, which drops each to the 10-cell error
   exactly. Note that cell-boundary alignment does *not* predict failure in general:
   5, 8, 12 and 25 cells are all misaligned and all converge, so the kink is
   implicated in those two cases without being the whole rule.

Restrictions, all refused rather than warned about:

* ``PolynomialDegree >= 3`` — see above.
* Steady solves only. Inherited from ``DegreeAdaptation``, and for the same reason:
  each stage would otherwise take the previous stage's final state as its initial
  condition and integrate the interval again.
* Not with ``GridPoints``, and not with ``GradedGridBoundary`` — both of those
  already determine the mesh, so one of them would silently lose.

.. note::

   **Scope, stated plainly.** The threshold was chosen from a gap between two
   populations of one and two problems. That is enough to establish the mechanism
   and the ordering; it is not a calibration. A singularity milder than
   :math:`x^{4/3}` would sit closer to the smooth population, and the honest
   response is that the threshold is a configuration key with this measurement
   beside it rather than a constant buried in the source.

   Per-cell degrees — a different degree in each cell — were scoped and **declined**:
   measured at 217 ``(k+1)`` sites in the core for a few percent of DOF, because the
   smooth region's degrees were measured to buy nothing at all.
   ``MESH-REFINEMENT.md`` has that analysis, which stays useful if the question
   returns for another reason.

.. _adaptivity-parallelism:

Machines that evaluate N points at once
---------------------------------------

``PhysicsParallelism = N`` describes a machine on which a batch of :math:`M`
physics points takes :math:`\lceil M/N \rceil` rounds whatever :math:`M` is: a
flux model spread over :math:`N` GPUs or MPI ranks, say. A level whose batches
leave the last round part empty is paying for evaluations it does not make, so
every controller **fills** the level it chooses: having chosen it as it always
has, it takes the largest level beyond it that costs no more rounds. The extra
accuracy is free in physics time, and it often saves a later level.

What a level costs is read from its evaluation plan (:ref:`evaluation-plans`) —
the plan a solver at that level would make, not a formula for it. Each batched
site the run repeats counts as calls × :math:`\lceil \text{batch}/N \rceil`,
separately for what happens per residual, per Jacobian build and per continuation
step, and a level fills another only if it costs no more in each. Pointwise hooks
are not counted, because MaNTA calls them one point at a time; nor is anything
that happens once per run.

``DegreeAdaptation``
   After capping the rule's step at ``MaxDegreeIncrement``, raises :math:`k` to the
   largest degree that costs no more rounds, up to ``MaxPolynomialDegree``. The cap
   bounds how far one estimate may send the loop; degrees that cost nothing extra
   to evaluate are not a bet on the estimate, so they are not held to it.

``MeshAdaptation``
   Grades as many cells as cost no more rounds than the sample's. The extra cells
   go into the layer at the ratio that keeps the wall cell the sample's count would
   have had, for as long as that ratio stays at or below 1/2, and the rest into the
   bulk. At a fixed ratio each added layer cell would shrink the wall cell by the
   ratio again, which below the layer's own width loses accuracy; past 1/2 the
   layer's next cell would be narrower than the wall cell. ``GradingCells`` holds
   the layer where it is, and then every extra cell goes to the bulk.

``DegreeLadder`` / ``GridLadder``
   Each intermediate rung is filled — in degree, then in cells, along whichever of
   the two the ladder steps — but never past the configured resolution, and a rung
   filled up to the level after it is dropped.

The configured level itself is the one no controller chooses, so it is only warned
about: if a larger ``PolynomialDegree`` (up to ``MaxPolynomialDegree``) or
``GridSize`` would cost no more rounds, the run says so before it starts.

Measured on the :math:`n = 2.5` wall layer of ``MESH-REFINEMENT.md`` §12 —
5 cells at :math:`k = 4`, ``DegreeTolerance = 1e-6``:

.. list-table::
   :header-rows: 1

   * - ``PhysicsParallelism``
     - graded mesh
     - degrees solved
     - relative :math:`L^\infty`
   * - 1
     - 5 cells, 4 in the layer at 0.3
     - 4, 6, 7, 8, 9
     - 1.8e-5
   * - 64
     - 10 cells, 6 in the layer at 0.486
     - 4, 10
     - 9.3e-6

Both have the same wall cell, 5.4e-3 of the domain. Note what is *not* free:
filling raises the per-cell linear algebra, which grows like :math:`k^3`, and the
number of cells the trace solve spans. It is the right trade when the physics is
what a run waits for, which is the only situation in which the key is worth
setting.
