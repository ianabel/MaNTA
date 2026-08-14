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

The grid and polynomial degree come from the restart file, so the corresponding
config keys are ignored on this path.

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

.. _steady-state-solver:

Reaching a steady state
-----------------------

When only the final state matters, ``SteadyStateSolver`` chooses how to get
there. It applies whenever steady-state termination is armed — ``run_ss()``, or a
config carrying ``SteadyStateTolerance`` — and is ignored by a plain
``run(tFinal)``, where the transient *is* the answer.

``PseudoTransient`` (the default)
   Pseudo-transient continuation, after Kelley and Keyes. A backward-Euler mass
   term is kept purely as damping and the pseudo-time step is sized from the
   *residual* rather than from a local error estimate: on each accepted step
   ``dt`` grows by ``||F_prev||/||F_now||`` (Switched Evolution Relaxation, with
   a floor of 2), and a step that increases the residual is rejected outright —
   the state is restored and ``dt`` cut by four. ``PseudoTransientInitialStep``
   sets the first step, defaulting to ``initialTimestep`` and then ``delta_t``;
   ``PseudoTransientMaxStep`` caps it.

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
     - 212
     - **127**
     - 152
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

The inner solve for both modes is **KINSOL**, driving the same static
condensation IDA does, so MaNTA links ``sundials_kinsol`` whichever mode a run
selects — see :doc:`install` if the build stops at ``kinsol/kinsol.h``.

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
