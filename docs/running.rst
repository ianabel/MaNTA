Running MaNTA, and its output
=============================

.. code-block:: sh

   ./MaNTA myproblem.conf

Output file names
-----------------

Output names default to the **stem of the configuration file**, and
``OutputFilename`` overrides that on both surfaces.

.. warning::

   Whichever name is used, the files land in the **current working directory**:
   any directory part is dropped. ``./MaNTA runs/case7.conf`` writes
   ``./case7.nc``, not ``runs/case7.nc``, and setting
   ``OutputFilename = "out/case7"`` still writes ``./case7.nc``. Two drivers
   running in different directories under the same base name will overwrite each
   other.

A run writes:

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - File
     - When
     - Contents
   * - ``<stem>.nc``
     - ``WriteOutput``
     - The solution at every output time. See the layout below.
   * - ``<stem>.restart.nc``
     - ``WriteOutput``
     - The final state, in enough detail to resume from.
   * - ``<stem>.dat``
     - ``WriteDatFile``
     - Plain-text columns for gnuplot.
   * - ``<stem>.dydt.dat``, ``<stem>.res.dat``
     - ``WriteDebugDatFiles`` **and** a ``PHYSICS_DEBUG`` build
     - The time derivative and the residual, for debugging.

The netCDF layout
-----------------

Top-level scalar variables record the shape of the problem — ``nVariables`` and
its companions — and a ``Grid`` group holds ``CellBoundaries`` and ``PolyOrder``.

Then, **one group per variable**, named by the physics case's
``getVariableName``, each containing four fields sampled at ``OutputPoints``
positions for every output time:

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Field
     - Meaning
   * - ``u``
     - The variable itself.
   * - ``q``
     - Its derivative :math:`\partial_x u`.
   * - ``sigma``
     - The flux. Remember this is :math:`-\hat\sigma`; see
       :ref:`the sign convention <sign-convention>`.
   * - ``u_star``
     - The postprocessed solution in :math:`P_{k+1}`. Present whenever
       :math:`k \ge 1`, whether or not ``Superconvergent`` is set. See
       :doc:`superconvergence`.

Auxiliary variables get a group each, named by ``getAuxVarName``. Global scalars
are written as time series, named by ``getScalarName``. A physics case can add
whatever else it likes through the ``initialiseDiagnostics`` and
``writeDiagnostics`` hooks, which is how derived quantities end up in the same
file.

.. _coupled-output:

What a coupled run adds
~~~~~~~~~~~~~~~~~~~~~~~

With a :doc:`field model <field_coupling>` attached, both ``<stem>.nc`` and
``<stem>.restart.nc`` gain **one further group**, named by the model's spec
(``FieldModelSpec::name``, defaulting to ``Field``), carrying:

* one **time series per field unknown**, named, described and given units by the
  spec's ``dofs`` — :math:`\psi` has no :math:`x` dependence, so it is written
  the way a global scalar is;
* one **spatial variable per geometry slot**, from the spec's ``geometry``,
  sampled at the same ``OutputPoints`` positions as ``u`` at every output time —
  these *are* functions of :math:`x`, so they are written the way ``u`` is;
* a group attribute ``label``, the spec's name for the spatial coordinate its
  geometry is expressed against. MaNTA does not interpret it; recording it is
  what lets a reader know what the run's :math:`x` meant.

The restart file additionally records ``RestartData/nField``, which is how many
of the trailing entries of ``Y`` are :math:`\psi`. It is written on every run,
so an uncoupled file says ``nField = 0`` rather than leaving a reader to
distinguish "no field model" from "written before the field block existed"; a
file predating this is read back as zero, which is the truth about it. Resuming
a coupled run needs both the file and a config naming a ``FieldModel`` that
declares the same number of unknowns, and a mismatch is reported by name rather
than as an ``nVars``/``nAux``/``nScalars`` disagreement.

Everything above is gated on ``WriteOutput``, like the rest of the netCDF and
restart output. A run with no field model writes exactly what it wrote before
the feature existed — byte for byte, apart from that one ``nField = 0`` in the
restart file.

With ``solveAdjoint`` set, the file also carries the scalars ``ng``, ``np`` and
``np_boundary``, one ``G<i>`` value per objective, and the groups ``G<i>_p`` and
``G<i>_boundary`` holding the gradient with respect to each named parameter.

.. note::

   ``WriteAdjoints()`` is currently commented out in ``Solver.cpp``, so a run
   does not in fact serialise the adjoint *state*. The gradients themselves are
   available through ``Runner.getAdjointGradients``; see :doc:`adjoints`.

Reading the output
------------------

Any netCDF reader will do. From Python:

.. code-block:: python

   from netCDF4 import Dataset

   nc = Dataset("case7.nc")
   u = nc.groups["Density"].variables["u"][:]      # (time, position)
   x = nc.groups["Grid"].variables["CellBoundaries"][:]

``Tools/`` contains a plotting script that reads these files directly.

Restarting
----------

A run leaves a ``<stem>.restart.nc`` unless ``WriteOutput`` is false. To resume,
point a config file at it:

.. code-block:: toml

   [configuration]
   restart = true
   RestartFile = "case7.restart.nc"
   t_initial = 0.5
   t_final = 1.0

The grid comes from the restart file, so ``Grid_size``, ``Grid_points``,
``Lower_boundary`` and ``Upper_boundary`` are ignored on this path.

``Polynomial_degree`` is **not** ignored. It defaults to the degree the file was
written at, and setting it to something else resumes the run at that degree
instead, projecting the stored state onto the new space:

.. code-block:: toml

   [configuration]
   restart = true
   RestartFile = "case7.restart.nc"   # written at Polynomial_degree = 2
   Polynomial_degree = 3              # resume at 3

Refining loses nothing — a degree-*k* element polynomial lies inside the
degree-(*k*\ +1) space, so the projection reproduces it exactly. Coarsening
discards what the lower space cannot hold, and nothing else: projections onto
nested spaces compose, so a state taken from *k* down to *k*\ −1 lands precisely
where a cold run at *k*\ −1 would. Both are pinned to round-off by
``a_restart_at_a_higher_degree_reproduces_the_state_exactly`` and its sibling.
Equal degrees take a straight copy, bit for bit, as they always have.

A degree change is logged at ``WARNING``, in both directions — the key is
required of every config, so a mismatch is as likely to be a config copied from
elsewhere as a deliberate request.

.. note::

   The *mesh* still cannot be changed on a restart. The nesting argument above
   is per cell and needs the same cell boundaries, and a physics case is
   constructed with the grid where nothing hands it the degree, so changing the
   mesh changes what the case was built against.

The **field model is not ignored either**: it is named by the config as usual,
and its declared ``nFieldDOF`` is checked against the ``nField`` the file
records before anything is read into it. Resuming a coupled run therefore needs
the same ``FieldModel`` line the original run had.

A restart written by a steady solve carries ``dYdt = 0``, which is the defining
property of the state it holds. It used to carry the ``t_initial`` derivative
instead — ``solveSteadyState`` damps through a scratch vector and never wrote
back to the one the restart file is built from — so a resumed run started from a
solution and a time derivative that did not belong together.

.. warning::

   **Keep** ``delta_t`` **at a size the resumed run could actually take as its
   first step.** On a restart, ``IDASetInitStep`` is given ``delta_t`` as the
   **first step**, on the reading that a resumed run should continue at the same
   step it left off with. A cadence chosen for output frequency rather than for
   stability can therefore hand IDA a first step approaching the whole remaining
   integration, and the error test then fails repeatedly until it gives up with
   ``IDA_ERR_FAIL`` (-3). Setting ``dt0`` overrides it.

   This warning used to have a second half, about ``t_initial == delta_t``
   failing with ``IDA_ILL_INPUT`` (-22) and ``tout1 too close to t0``. That was
   a MaNTA bug — ``IDACalcIC`` was handed the *interval* where SUNDIALS wants
   the first output *time* — and it is fixed; the two agree only at
   ``t0 = 0``, which is where every fixture in the tree starts, so no cold run
   ever noticed.

.. warning::

   Restarting is **fragile at tight tolerances**, and more so with ``nAux > 0``.
   The regression suite exercises three restart round trips, each at the tightest
   tolerance that completes; below that, ``IDACalcIC`` can fail to converge on
   the resumed run. If a restart fails where the original run succeeded, loosen
   the tolerances before looking for a deeper cause.

   Such a failure now reports itself: ``IDACalcIC could not complete``. It did not
   used to — the return value was overwritten before it was checked, so a failed
   initial-condition calculation carried on into the time loop with whatever
   partial state IDA had reached, and the symptom appeared later and elsewhere.

   :ref:`suppress-algebraic-error` makes this *worse*, not better: it is the
   accuracy of the algebraic components that a restart resumes from.

.. _coupled-field-sweeps:

What a coupled run reports
--------------------------

A run with ``FieldModel`` set and ``FieldSolve = iterative`` — the default —
prints one extra line beside the step and Jacobian counts at the end::

   Coupled field sweeps                  :1043 over 342 solves (0 exact fallbacks)

The ratio is what to read. A sweep is one transport solve; the exact Schur
complement is :math:`\texttt{nField}+1` of them. So the iterative path is
*cheaper* only when the mean sweep count is below :math:`\texttt{nField}+1`, and
on every fixture in this tree it is not — see :doc:`field_coupling` for the
numbers. Measured in a real integration the sweep runs 2.5 to 3.6 iterations per
Jacobian solve.

.. important::

   **A nonzero fallback count is a cost report, not an error.** A sweep that
   exhausts ``FieldSolveMaxSweeps`` escalates to the exact Schur solve, so the
   answers are correct; the run is simply paying for both, at
   :math:`\texttt{nField}+1` transport solves on top of the sweeps it already
   spent. The run says so:

   .. code-block:: text

      WARNING: 17 of 342 coupled Jacobian solves exhausted FieldSolveMaxSweeps = 20
      and fell back to the exact Schur solve, at 6 transport solves each. The answers
      are correct; the run is paying for both. Raise FieldSolveMaxSweeps, or set
      FieldSolve = exact and skip the sweeps.

   Nothing latches that decision, so a genuinely divergent coupling pays both on
   *every* Jacobian solve for the whole run. If the count is a large fraction of
   the solve count, ``FieldSolve = exact`` is the cheaper answer.

The escalation runs in **both** directions, forward and adjoint, which is what
makes ``FieldSolve`` a cost choice and never an accuracy one. The adjoint has its
own cap, ``FieldSolveMaxAdjointSweeps``, defaulting to 100 against the forward
20: the transposed iteration has the same spectrum — it *is* the transpose — but
always runs at :math:`c_j = 0`, where the spectral radius is largest, so it is
strictly the harder direction. Five field unknowns have been measured needing
13–38 sweeps on isolated right-hand sides, which is why inheriting the forward
cap would under-serve it.

.. note::

   The counts are per run. ``initialize`` zeroes them, alongside the field
   model's own ``resetForRun``, so a second run on a reused solver reports its
   own numbers rather than a cumulative total.

Two failures worth being able to read
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``initialize`` refuses a **field DOF declared differential whose residual row
carries no time derivative**, naming the DOF. Left to IDA that is an
``IDA_LINESEARCH_FAIL`` (-13): ``IDA_YA_YDP_INIT`` holds every differential
*value* fixed, so a row that reaches no unknown it may move is irreducible and
the backtracking loop runs to exhaustion — a message about the linesearch for a
defect in the declaration.

A field model that **cannot evaluate at the state it is handed** should throw
from ``FieldResidual``. That is caught and reported to IDA as a *recoverable*
error, so the step is retried with a smaller ``h`` rather than the run failing.

.. _steady-state-solver:

Reaching a steady state
-----------------------

When only the final state matters, ``SteadyStateSolve = true`` asks for a steady
solve and ``SteadyStateSolver`` chooses how to reach one. Naming a
``SteadyStateTolerance`` asks for the same thing and sets the tolerance with it;
either arms it, and giving both uses the tolerance.

.. warning::

   ``SteadyStateSolver`` alone does **not** arm a steady solve. It names the
   *method*, and the method is only consulted once termination is armed — so a
   config setting ``SteadyStateSolver = "Newton"`` and nothing else time-marches,
   which is not what it looks like. That is why ``SteadyStateSolve`` exists:
   arming used to be a side effect of choosing a tolerance, so asking for a
   steady solve meant having an opinion about how tight it should be.

Steady-state termination is also armed by ``run_ss()``, which supplies its own
fallback tolerance, and is ignored by a plain ``run(tFinal)``, where the
transient *is* the answer.

``PseudoTransient`` (the default)
   Pseudo-transient continuation, after Kelley and Keyes. A backward-Euler mass
   term is kept purely as damping and the pseudo-time step is sized from the
   *residual* rather than from a local error estimate: on each accepted step
   ``dt`` grows by ``||F_prev||/||F_now||`` (Switched Evolution Relaxation, with
   a floor of 2), and a step that increases the residual is rejected outright —
   the state is restored and ``dt`` cut by four. ``PseudoTransientInitialStep``
   sets the first step, defaulting to ``initialTimestep`` and then ``delta_t``;
   ``PseudoTransientMaxStep`` caps it.

   The schedule itself is adjustable. On an accepted step,

   .. math::

      dt \leftarrow dt \cdot \max\!\left[
         \left( \frac{\|F_\mathrm{prev}\|}{\|F_\mathrm{now}\|} \right)^{r},\ f
      \right]

   with ``PseudoTransientSERRate`` the exponent :math:`r` (default 1) and
   ``PseudoTransientSERFloor`` the floor :math:`f` (default 2).

   ``r`` says how hard ``dt`` leans on the residual reduction: 0 ignores it and
   grows at the floor alone, 1 is plain SER, above 1 is more aggressive. It may
   not be negative — that would shrink ``dt`` as the residual falls.

   ``f`` is the least ``dt`` may grow on a step that made progress, and is why
   plain SER is not enough on its own: the ratio is only as large as the
   residual reduction, and the reduction is only as large as ``dt`` allows, so a
   conservative first step is self-perpetuating. ``f = 1`` is "no floor", i.e.
   plain SER; below 1 is refused, since this branch runs only when the residual
   fell and so the ratio already exceeds 1 — a smaller floor could never bind.

   Both are worth reaching for when a problem sits at either extreme. Measured
   on the unit-test diffusion problem from ``dt = 1e-3``, in physics
   evaluations:

   .. list-table::
      :header-rows: 1

      * - ``r``
        - ``f``
        - evaluations
      * - 1
        - 2 *(defaults)*
        - **552**
      * - 1
        - 1 *(plain SER)*
        - 3540
      * - 2
        - 1
        - 1704

   The converged state is identical in every case; only the number of
   continuation steps differs. Growing ``dt`` faster is not free, though — on a
   degenerate flux such as ``shestakov-nonlinear``'s it is what makes the inner
   solve start rejecting steps, which is the trade-off these two expose.

``Newton``
   The same code with an infinite first step, so the damping term is absent from
   the outset and this is Newton's method on the steady problem. Cheapest when
   it works; it has no globalisation, so it wants a decent starting state. A
   rejected step drops it to a finite ``dt`` and it continues as continuation.

``TimeMarch``
   The original behaviour: integrate with IDA until ``dY/dt`` falls below
   ``SteadyStateTolerance``. Slower, and the only one of the three that selects
   a solution *branch* by following the physics — which matters for a problem
   with more than one steady state, such as a transport model with a barrier
   bifurcation.

Measured on the benchmarks under ``python-examples/``, in the units
``PERFORMANCE.md`` asks for — evaluations of the physics per point, for an
answer identical in every digit printed:

.. list-table::
   :header-rows: 1

   * - benchmark
     - ``TimeMarch``
     - ``PseudoTransient``
     - ``Newton``
   * - ``park-convergence``
     - 113
     - **19**
     - **11**
   * - ``jardin-critical-gradient``
     - 176
     - **92**
     - 117
   * - ``shestakov-nonlinear``
     - **283**
     - 705
     - 731

Park's own solver reaches that state in 9–15 iterations, which is where
``Newton`` lands. The last row is the counter-example and is why ``TimeMarch``
stays: that problem's flux ``D0 q^3/u^2`` is degenerate, the mass term
continuation exists to shed is what was damping it, and as ``dt`` grows the inner
solve starts rejecting steps. Its ``run.conf`` therefore pins ``TimeMarch``.

.. note::

   A steady solve is also *better* for adjoints, not merely cheaper. The adjoint
   state method assumes ``F(y, p) = 0``; ``TimeMarch`` only approximates that
   through a ``dY/dt`` threshold, while these two enforce it. On the
   ``test_adjoint.py`` fixture the gradient moves from ``2e-5`` to ``2e-8``
   against a finite-difference reference — the same accuracy as integrating far
   past the transient, for a fraction of the work.

   Explicitly time-dependent data — boundary values, sources — is frozen at
   ``t_initial``. There is no time axis to evaluate it on.

Output from a steady solve
~~~~~~~~~~~~~~~~~~~~~~~~~~

``PseudoTransient`` and ``Newton`` do not advance time, so there is no elapsed
time to report and no series of timeslices to write. A run in either mode
produces **exactly two** slices:

.. list-table::
   :header-rows: 1

   * - ``t``
     - what it holds
   * - ``0`` (strictly, ``t_initial``)
     - the initial condition, written by ``initialize()``
   * - ``1``
     - the converged steady state

**The second stamp is a label, not a time.** It is
``SystemSolver::STEADY_STATE_TIME``, fixed at ``1.0`` whatever ``t_initial`` and
``t_final`` are, and it carries no physical meaning: every time-dependent input
was frozen at ``t_initial``, as above. A fixed label is what separates the
answer from the initial condition in the file — the alternatives do not, since
the solver's clock never leaves ``t_initial`` on this path, and ``run_ss()``
calls the solver with ``tFinal = 0``. So read a steady run's answer as *the last
slice*, and do not difference the two for a rate of change.

``WriteDatFile`` output follows the same shape: two blocks, the second being the
converged state.

``TimeMarch`` is unaffected — it writes a slice per output cadence at real times
and stops when ``dY/dt`` falls below the tolerance, so its last slice carries the
time the state was actually reached.

**A failed steady solve writes its last state too**, at the same ``t = 1``,
before the error propagates — which is exactly the run whose state is worth
looking at. On the "ran out of continuation steps" path that is the last
*accepted* iterate; after a hard ``KINSol`` failure it is whatever KINSOL left.
Note two things about it: ``dYdt`` is still the ``t_initial`` derivative, since
only a converged solve zeroes it, so a diagnostic hook differentiating it there
is reading the initial condition's rate of change; and **nothing in the file
distinguishes a failed last slice from a converged one** — the exception, the
logged error and the exit status do.

What the solve did
~~~~~~~~~~~~~~~~~~

Every steady solve prints what it is about to do — mode, mesh, tolerance,
starting ``||F||`` and ``dt``, and the SER settings — then how it ended. That is
unconditional, matching ``TimeMarch``, which prints a line per output slice and
three IDA totals at the end.

``SteadyStateDiagnostics = true`` adds the cost:

.. code-block:: text

   Steady solve: PseudoTransient on 6 cells at k = 3, tolerance 1e-10
     initial ||F|| = 4.74756, dt = 0.05, SER rate 1, floor 2, max step inf
     converged: ||F|| = 1.67044e-13 after 5 continuation steps.
   Steady solve statistics -- converged
     continuation steps      : 5  (0 rejected)
     KINSOL Newton iterations: 35
     residual evaluations    : 46  (of which KINSOL: 40)
     Jacobian builds         : 7  (KINSOL asked for 7)
     Jacobian solves         : 35

It is off by default because a steady solve is often run in a loop by an
optimisation driver, where a block per solve is noise. It prints on **failure**
as well, labelled with which failure — that is the case it is most useful in.

Reading the numbers:

* **continuation steps** are outer iterations, one ``KINSol`` call each. A rejected one restored the state and cut ``dt``; a run that rejects most of its steps is one whose ``dt`` is growing faster than the problem will take, so lower ``PseudoTransientSERFloor``.
* **residual evaluations** are the true total, KINSOL's own plus the merit function's — the latter costs one per continuation step plus one on entry, and is the number ``PERFORMANCE.md`` cares about.
* **Jacobian builds against solves** is the useful split. A build assembles and factorises the per-cell blocks; a solve is the static condensation that reuses them. KINSOL keeps a factorisation across Newton iterations, so on a nonlinear problem builds are far fewer — ``AdjointPoster`` above pays 7 against 35. On a *linear* problem each inner solve converges in one iteration and the two coincide.

The same numbers are available programmatically from
``SystemSolver::lastSteadyStats()``, filled in whether or not they were printed
and whether or not the solve converged.

Per continuation step
~~~~~~~~~~~~~~~~~~~~~

``SteadyStateStepDiagnostics = true`` reports each ``KINSol`` invocation as it
returns, one row per continuation step:

.. code-block:: text

   Steady solve: PseudoTransient on 6 cells at k = 3, tolerance 1e-10
     initial ||F|| = 4.74756, dt = 0.05, SER rate 1, floor 2, max step inf
     step          dt       ||F||  iters    res    jac  solves  outcome
        0   5.000e-02   9.359e-01     13     15      2      13  accepted
        1   2.536e-01   1.105e-01     11     13      2      11  accepted
        2   2.148e+00   2.108e-03      7      9      1       7  accepted
        3   1.126e+02   7.974e-07      3      5      1       3  accepted
        4   2.977e+05   1.653e-13      1      3      1       1  accepted
     converged: ||F|| = 1.6533e-13 after 5 continuation steps.

That is the same solve the totals above describe — 35 Newton iterations, 46
residual evaluations, 7 Jacobian builds, 35 solves — and the point of the table
is what the totals cannot say: **the cost is all in the first two steps.** 24 of
the 35 Newton iterations go on getting ``||F||`` from 4.7 to 0.11, and the last
three steps together cost 5. A solve that took twenty cheap steps and one that
took three expensive ones report similar totals and want opposite things done to
them, and only the trace distinguishes the two.

The two flags are **independent** and compose: either can be had on its own. A
trace is the more specialised request, so it is not nested inside the summary.

Reading the columns:

* ``dt`` is the pseudo-time step the call was *damped with*, before SER updates it — so the row shows what was tried, not what will be tried next.
* ``||F||`` is the **steady** residual after the call, which is not the norm ``KINSol`` converged. KINSOL sees the damped residual, and a small enough ``dt`` makes that small whatever the state; the merit function re-evaluates at ``dt = infinity``. That extra evaluation is why ``res`` exceeds KINSOL's own count by exactly one per step.
* ``iters``, ``res``, ``jac``, ``solves`` are that step's Newton iterations, residual evaluations, Jacobian builds and Jacobian solves. They sum to the totals, with one documented offset: the merit function is evaluated once *before* the loop, so ``sum(res) + 1`` is the total. ``sum(jac)`` and ``sum(solves)`` are equalities — nothing builds or solves outside the loop.
* ``outcome`` is ``accepted`` when the step reduced ``||F||``, ``rejected`` when it was rolled back and ``dt`` cut, and ``FAILED (n)`` for the ``KINSol`` return that ends the solve. A failing row is printed before the exception propagates, which is the case the trace is most useful in — and ``||F||`` is ``nan`` there, because no steady residual was evaluated after that call.

``SystemSolver::lastSteadyStepStats()`` returns the same records as a vector, in
order, **filled whether or not they were printed** — so a driver can have the
trace without putting it through ``stdout``. It is cleared at the top of every
``solveSteadyState``, so it describes one solve rather than the solver's history,
which matters for ``PyRunner``: it runs many solves on one object.

Over a whole run
~~~~~~~~~~~~~~~~

One run holds one steady solve, except under :ref:`degree adaptation
<degree-adaptation>`, which builds a solver per level and solves at each. There
the totals above are *per level*, and ``SteadyStateDiagnostics`` adds a run total
after the last one:

.. code-block:: text

   Degree adaptation totals -- 4 levels, one steady solve each
     continuation steps      : 15  (0 rejected)
     KINSOL Newton iterations: 71
     residual evaluations    : 105  (of which KINSOL: 86)
     Jacobian builds         : 19  (KINSOL asked for 19)
     Jacobian solves         : 71

That is the number to compare against a fixed-degree run, since the whole bet of
adapting the degree is that the coarse levels are cheap enough to be worth
paying for. It is printed even at one level, where it duplicates that level's
own block: a log whose shape depends on how many levels a run happened to take
is one nothing can read mechanically.

.. note::

   Before this was fixed, a ``PseudoTransient`` or ``Newton`` run wrote **only**
   the ``t = 0`` slice: every output call lived inside the time loop, which
   those two modes skip. The converged state still reached ``getSolution()`` and
   the ``.restart.nc``, so the Python API was correct and only the files were
   wrong. A physics case's ``writeDiagnostics`` hook is called from the same
   place and so was never called at all, which is why a steady run's diagnostic
   groups were empty.

The inner solve for both modes is **KINSOL**, driving the same static
condensation IDA does, so MaNTA links ``sundials_kinsol`` whichever mode a run
selects — see :doc:`install` if the build stops at ``kinsol/kinsol.h``.

Controlling the inner solve
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Four keys reach KINSOL. They apply to ``PseudoTransient`` and ``Newton`` alike —
pseudo-transient continuation *is* Newton on a damped residual — and not at all
to ``TimeMarch``, which never builds a KINSOL object. Every default reproduces
what the code did when these were hardcoded, so an unconfigured run is unchanged.

``NewtonJacobianReuse``
   How many Newton iterations may share one Jacobian factorisation (KINSOL's
   ``msbset``). ``1`` is full Newton; larger is modified Newton. **This is the
   setting the** ``jac`` **and** ``solves`` **columns above measure**, and the
   section below is about why it is worth setting per case.

``NewtonMaxIterations``
   Newton iterations one ``KINSol`` call may take before handing back to the
   continuation loop. The default is **20 against KINSOL's own 200**, deliberately:
   an inner solve only has to make progress, because SER re-damps and tries again
   from a better ``dt``. Raise it for ``SteadyStateSolver = "Newton"``, where
   there is no outer loop to fall back on.

``NewtonStepTolerance``
   KINSOL's scaled-step test. A ``KINSol`` below it returns
   ``KIN_STEP_LT_STPTOL`` — which the continuation loop treats as *ordinary* and
   answers by damping, not as a failure. So raising it makes inner solves give up
   sooner and ``dt`` be cut more eagerly: a continuation-schedule control wearing
   a tolerance's clothing. Zero leaves KINSOL's ``uround^(2/3)`` ≈ 3.7e-11.

``NewtonScaling``
   ``Unit`` (default) or ``ErrorWeights``. KINSOL's convergence tests are on
   *scaled* quantities, so unit scaling makes them dimensional: on a case carrying
   densities near 1e19 beside temperatures near 1e3, one ``SteadyStateTolerance``
   means something different for each variable and the largest dominates.
   ``ErrorWeights`` fills the vectors from the same ``1/(rtol|y| + atol)`` weights
   IDA's WRMS norm uses, refreshed every continuation step because they depend on
   the state and the state moves a long way. One honest limitation: KINSOL takes
   separate ``u_scale`` and ``f_scale`` and both get the same vector here, as they
   already did when both were ones — the residual does not carry the solution's
   units, so a properly derived ``f_scale`` would be a different vector.

.. _jacobian-reuse-measurement:

What Jacobian reuse actually trades
"""""""""""""""""""""""""""""""""""

The three costs are not comparable, and the ordering is what makes this worth a
key rather than a constant:

* **A Jacobian solve is always cheap.** It is a static condensation against a
  factorisation that already exists.
* **A Jacobian assembly is at least as expensive as a residual evaluation**, and
  the ratio is set by *your physics case*, not by the solver. A case whose flux
  is differentiable — hand-written derivatives, ``AutodiffTransportSystem``, JAX —
  pays value-and-gradient against value, which is more but not by much. A case
  whose Jacobian comes from finite-differencing expensive flux calls pays many
  flux evaluations per assembly, and **assemblies then dominate the run**.

So raising ``NewtonJacobianReuse`` trades assemblies away for extra Newton
iterations, and each of those costs a residual evaluation plus a (cheap) solve.
Which side wins is a property of how your flux model is differentiated. **That is
the whole reason this is configurable**, and it is why there is no default that is
right for every case.

The default of 10 is KINSOL's. At the cheap-Jacobian end of the range it is
conservative — measured on ``AdjointPoster``, an analytic flux, at k = 3, driving
the residual to 1e-10:

.. list-table::
   :header-rows: 1

   * - ``NewtonJacobianReuse``
     - 800 cells
     - builds
     - solves
     - residual evals
   * - 1 (full Newton)
     - **3.39 s**
     - 15
     - 15
     - 26
   * - 2
     - 3.64 s
     - 10
     - 17
     - 28
   * - 5
     - 4.94 s
     - 8
     - 25
     - 36
   * - 10 (default)
     - 6.21 s
     - 7
     - 32
     - 43
   * - 20
     - 8.30 s
     - 5
     - 45
     - 56

Full Newton is 1.8× faster *there*, and the gap widens with the mesh — at 200
cells it is 0.18 s against 0.21 s. ``AdjointPoster`` differentiates cheaply, so it
sits at the end of the range where assemblies are nearly free.

Measured on the three benchmarks under ``python-examples/``, in the units
``PERFORMANCE.md`` uses — calls into the ``TransportSystem``, split into flux
evaluations (residual cost) and derivative evaluations (assembly cost):

.. list-table::
   :header-rows: 1

   * - case
     - mode
     - reuse
     - flux calls
     - derivative calls
   * - Park
     - ``TimeMarch``
     - —
     - 1504
     - 400
   * - Park
     - ``PseudoTransient``
     - 10 / 1
     - 176
     - 64
   * - Park
     - ``Newton``
     - 10 / 1
     - **128**
     - **48**
   * - Jardin
     - ``TimeMarch``
     - —
     - 4256
     - 1088
   * - Jardin
     - ``PseudoTransient``
     - 10
     - 2560
     - 384
   * - Jardin
     - ``PseudoTransient``
     - 1
     - 544
     - 320
   * - Jardin
     - ``Newton``
     - 10
     - 3264
     - 480
   * - Jardin
     - ``Newton``
     - 1
     - **416**
     - **256**
   * - Shestakov
     - ``TimeMarch``
     - —
     - 7808
     - 1216
   * - Shestakov
     - ``PseudoTransient``
     - 10
     - *fails*
     - *fails*
   * - Shestakov
     - ``PseudoTransient``
     - 1
     - 1920
     - 704
   * - Shestakov
     - ``Newton``
     - 10
     - *fails*
     - *fails*
   * - Shestakov
     - ``Newton``
     - 1
     - **1792**
     - **640**

Three things to take from it.

**Park does not care, and that is the control.** Its ``chi`` is constant, so the
flux is linear in the unknowns, every inner solve converges in one Newton
iteration, and there is never a second iteration to reuse a Jacobian across. A
setting that changed Park's numbers would be evidence of a bug, not of tuning.

**On the two nonlinear cases, reuse is not a trade — it loses on both axes.**
Jardin under ``Newton`` costs 416 flux and 256 derivative calls at reuse 1
against 3264 and 480 at reuse 10. Fewer assemblies *per iteration* bought so many
extra iterations that the total assembly count went up as well. The trade
described above is real only while the Jacobian is stable enough that a stale one
still points somewhere useful; on a strongly nonlinear problem it is not, and the
extra iterations are pure loss.

**At the default, Shestakov does not converge at all**, in either steady mode,
returning ``KIN_MXNEWT_5X_EXCEEDED``. A Jacobian ten iterations old gives a bad
enough direction that the step clamp fires five times running. At reuse 1 both
modes converge, and ``PseudoTransient`` beats ``TimeMarch`` four to one — which
reverses the note in ``../shestakov-nonlinear/`` that continuation costs 2.5× what
time marching does. That measurement was taken at the default and is a statement
about ``msbset``, not about pseudo-transient continuation.

So the honest summary is that KINSOL's default of 10 suits neither of MaNTA's
nonlinear benchmarks, and on one of them it is the difference between converging
and not. It is left in place only because the cost model above says the opposite
case exists: a physics case whose Jacobian is finite-differenced from expensive
flux calls pays far more per assembly than these do, and would rather have the
iterations. **If a steady solve is slow or will not converge,**
``NewtonJacobianReuse = 1`` **is the first thing to try.**

.. _degree-adaptation:

Choosing the polynomial degree
------------------------------

``DegreeAdaptation = true`` picks the global polynomial degree by solving,
measuring how well resolved the answer is, and re-solving at a higher degree
until it is good enough:

.. code-block:: toml

   [configuration]
   SteadyStateSolver = "Newton"
   SteadyStateTolerance = 1.0e-10
   Polynomial_degree = 2          # where to start
   DegreeAdaptation = true
   DegreeTolerance = 1.0e-9       # relative L2 error to reach
   MaxPolynomialDegree = 12       # where to give up
   MaxDegreeIncrement = 3         # most degrees to add at once (the default)

On ``AdjointPoster`` at 6 cells that is four solves — ``k`` = 2, 5, 8, 10 —
taking the estimated error 2.1e-3, 8.6e-6, 1.6e-8, 2.0e-10.

The estimate is the gap between the solution and its own postprocessing,
:math:`E_K^2 = \|u^* - u_h\|^2_{L^2(K)} / |K|` per cell (Capasso *et al.*
eq. 15). Every level reports two aggregates:

.. code-block:: text

   k = 2: relative L2 error 2.103e-03 (variable 0), absolute 1.244e-03,
          worst cell 2.535e-03 at cell 1

The **relative L2 error** drives the decision, because it is the quantity the
benchmarks quote. The **worst cell** is the binding constraint on a *single*
global degree, and is the one to look at if the loop converges while some corner
of the domain is plainly unresolved — a graded mesh, not a higher degree, is the
answer to that.

The degree rises by Giorgiani's rule, :math:`\Delta k = \lceil \log_b(E/\epsilon)
\rceil`, with :math:`b` = ``DegreeAdaptationBase`` between 10 and 100. A larger
base is a more aggressive assumption about what one extra degree buys, so it
asks for *fewer* of them. The rule assumes no convergence *order* at all, which
is deliberate: :math:`u^*`'s observed rate is not dependable enough to calibrate
against — see :doc:`superconvergence`.

``MaxDegreeIncrement`` then caps each step, at 3 by default. The rule is free to
ask for a large jump from a coarse first solve — the run above asked for +7 from
``k`` = 2 — and taking it would clear most of the budget without reporting
anything on the way. The cap costs solves and buys a legible trajectory; when it
binds, the run says so::

   raising k from 2 to 5 (the rule asked for +7, capped at +3)

Four things worth knowing:

* **It implies** ``Superconvergent = true``. The whole estimate rests on
  :math:`u^*` being the better of the two approximations, which is only assured
  with the superconvergent scheme on. Setting ``Superconvergent = false``
  alongside it is refused rather than silently overridden; leave the key out and
  it is enabled for you.
* **Steady solves only.** ``SteadyStateSolver = "TimeMarch"`` is refused: the
  estimate cannot separate spatial from temporal error, and the transfer between
  levels drops the BDF history.
* **The tolerance is relative**, to each variable's own :math:`L^2` norm, so one
  number means the same thing for variables in different units. It is floored by
  ``Absolute_tolerance``, which is what stops a solution that is *identically
  zero* — ``LinearDiffusion`` with zero Dirichlet data at both ends, say — from
  dividing one round-off by another and climbing to the ceiling for nothing.
* **The ceiling warns rather than failing.** Reaching ``MaxPolynomialDegree``
  without meeting the tolerance leaves the best available answer in the output
  and logs a warning. A run that stopped there did not converge, whatever the
  files look like — and it may be telling you something: an endpoint
  singularity is exactly what raising the degree cannot fix. ``NonlinDiffTest``
  climbs to the ceiling because its steady state behaves like
  :math:`(1-x)^{1/3}` at the upper boundary, and the estimate falls only like
  :math:`1/k`. A graded mesh is the answer to that, not a higher degree; the
  *worst cell* line tells you where to put one.

Each level writes output, so the files left behind are the final level's. Levels
after the first start from the previous one's solution, projected onto the new
space; see `Restarting`_ for what that projection costs, which for refining is
nothing.

.. note::

   The degree is **global**. Per-cell degrees are a much larger change than they
   look — ``DGSolnImpl`` holds one ``k`` and one basis by value, and there are
   some 320 ``(k+1)`` sites in the core — and the measurements in
   ``MESH-REFINEMENT.md`` say most of the available win does not need them.

   Adaptation is also refused alongside spatial adjoint parameters, for the same
   reason ``Superconvergent`` already is: those are indexed by node, so changing
   the degree changes how many parameters there are.

.. _suppress-algebraic-error:

Dropping the algebraic rows from the error test
-----------------------------------------------

``SuppressAlgebraicError = true`` calls SUNDIALS' ``IDASetSuppressAlg``, which
takes ``sigma``, ``q``, ``lambda`` and ``phi`` out of IDA's local error test and
leaves only ``u`` and the differential scalars in it. It is **off by default**,
and this is a trade rather than an improvement.

What it buys. Any problem whose flux grows steeply — a degenerate diffusivity, a
critical-gradient model near its kink — puts a large ``sigma`` into an error test
that no reduction in step size can satisfy, because the offending component is
algebraic. The signature is an ``IDA_ERR_FAIL`` (-3) whose ``dsm`` in the
SUNDIALS log is *identical* as ``h`` shrinks, with the Newton converging happily
each time. Setting this key is the direct fix, and it also dissolves the
``Absolute_tolerance`` floor that otherwise makes ``atol <= 1e-7`` fail on the
first step of such a problem. Measured on the benchmarks under
``python-examples/``, it costs 13–44% *fewer* calls into the physics and moves
the answer of a direct run by nothing at five significant figures.

What it costs. ``sigma``, ``q``, ``lambda`` and ``phi`` are then controlled only
by the Newton tolerance, and two things read them:

* **Restart files serialise the whole DOF vector.** A round trip that agreed to
  ``1.9e-6`` degrades to ``8.6e-4`` with this on — see the warning above, which
  this key makes worse rather than better.
* **``phi`` is a physics quantity when ``nAux > 0``**, not merely an
  intermediate. The ``AuxVarTest`` regression case drifts 1.0% against its
  reference with this on, past its 0.84% tolerance.

So it is the right key for a hard steady-state or transient solve whose output is
``u``, and the wrong one if you intend to restart from the result or care about
the auxiliary variables. ``python-examples/shestakov-nonlinear`` is a problem
that cannot be integrated at all without it — though that turns out to depend on
the initial condition rather than on the formulation, which is worth knowing
before reaching for this key. A start whose flux is badly scaled needs it; the
same problem from a physically scaled start does not. That example's
``ANALYSIS.md`` measures both.

The three phases
----------------

``runSolver`` is a convenience wrapper around three steps that can also be driven
separately — which is what lets a caller stop between building the initial
condition and integrating:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Step
     - What it does
   * - ``initialize``
     - Allocates the SUNDIALS objects, builds the initial condition, runs
       ``IDACalcIC``, takes its corrected result back, and opens the output files.
       So the :math:`t_0` slice of every output is the **corrected** initial
       condition — the state the time loop actually starts from, rather than the
       guess handed to ``IDACalcIC``. The two differ only in the algebraic fields
       (:math:`q`, :math:`\sigma`, the auxiliary variables, and :math:`u^\star`
       through :math:`q`); :math:`u` is differential, so ``IDACalcIC`` holds it
       fixed and it is the same either way.
   * - ``integrate(tFinal)``
     - The time loop, then the adjoint solve if requested, then the final netCDF
       and restart output.
   * - ``destroySundials``
     - Frees all of it. Idempotent, and safe to call with no preceding
       ``initialize``.

Reusing a solver
~~~~~~~~~~~~~~~~

**The three phases can be run again on the same object**, and a reused solver
gives the same answer a fresh one would — *bit for bit*. That is pinned by
``a_second_integration_on_one_solver_matches_a_fresh_one`` in
``Tests/UnitTests/SolverLifecycleTests.cpp``, at exactly zero tolerance.

The tolerance is the point rather than a flourish. A second integration used to
fail outright, with ``IDASolve`` returning ``IDA_ERR_FAIL`` on its first step,
and two defects had to combine to produce it: ``id`` was left all zeros, so IDA
was told the whole system was algebraic and ``IDACalcIC``'s return value was
discarded when it failed; and ``initialiseMatrices``, which ``initialize`` skips
when it has already run, filled the boundary arrays at a hardcoded
:math:`t = 0`, so a second run solved its initial :math:`\mathrm{d}y/\mathrm{d}t`
out of the *previous* run's final-time boundary values. Once the first was
fixed, the second run completed and looked right — and was wrong in the eleventh
digit. An approximate comparison would not have caught it.

.. note::

   Anything that reuses a solver rests on that test, so do not relax it to
   something approximate. Note also that ``initialize`` skips
   ``initialiseMatrices`` when the solver is already initialised: anything that
   function computes *once* must either be genuinely run-independent or be
   refreshed per run.

The Python ``Runner`` is unaffected either way — ``configure`` builds a fresh
solver every time — which is why the loop-over-runs pattern in the optimisation
drivers was always safe.

If you hold on to the solution after a run, note that the object mapping the
live SUNDIALS vector dangles once the run is freed; the separately owned
``yJac``/``dydtJac`` views are the ones that outlive a solve, and they are valid
after ``initialize`` as well as after ``integrate``.
