Running MaNTA, and its output
=============================

.. code-block:: sh

   ./MaNTA myproblem.conf

Output file names
-----------------

.. warning::

   Output names come from the **stem of the configuration file**, and the files
   land in the **current working directory** regardless of any path in the config
   file's own name. ``./MaNTA runs/case7.conf`` writes ``./case7.nc``, not
   ``runs/case7.nc``. There is no config key that changes this — the TOML
   interface does not read ``OutputFilename`` at all.

A run writes:

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - File
     - When
     - Contents
   * - ``<stem>.nc``
     - always
     - The solution at every output time. See the layout below.
   * - ``<stem>.restart.nc``
     - always
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

Every run leaves a ``<stem>.restart.nc``. To resume, point a config file at it:

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
     - Allocates the SUNDIALS objects, builds the initial condition, opens the
       output files, runs ``IDACalcIC``.
   * - ``integrate(tFinal)``
     - The time loop, then the adjoint solve if requested, then the final netCDF
       and restart output.
   * - ``destroySundials``
     - Frees all of it. Idempotent, and safe to call with no preceding
       ``initialize``.

.. warning::

   **A second integration on the same solver object does not work.** ``IDASolve``
   fails with ``IDA_ERR_FAIL`` on the first step of the second run. Calling
   ``initialize`` again after ``destroySundials`` *does* work and rebuilds the
   initial condition; it is completing a second time loop that fails. This is
   undiagnosed. It does not affect the Python ``Runner``, whose ``configure``
   builds a fresh solver every time, which is exactly why the loop-over-runs
   pattern in the optimisation drivers is safe.

If you hold on to the solution after a run, note that the object mapping the
live SUNDIALS vector dangles once the run is freed; the separately owned
``yJac``/``dydtJac`` views are the ones that outlive a solve, and they are valid
after ``initialize`` as well as after ``integrate``.
