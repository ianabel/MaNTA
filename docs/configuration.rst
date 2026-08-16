Configuration
=============

There are two ways to configure a run:

* a **TOML file** given to the ``MaNTA`` binary, described here and read by
  ``runManta`` in ``MaNTA.cpp``;
* a **Python dict** passed to ``Runner.configure``, read by ``PyRunner.cpp``.

They are the **same set of keys**, declared once in ``ConfigSchema.cpp`` and read
by both through ``loadSolverConfig``. A key has the same name, the same type and
the same default whichever way it arrives. The few places the two surfaces still
differ are deliberate, and are listed in :ref:`config-divergences` below.

``./MaNTA --list-options`` prints the current schema — every key, its type, its
default and a line of description — straight from the table this page is written
from.

A minimal file
--------------

.. code-block:: toml

   [configuration]

   TransportSystem = "LinearDiffusion"

   Polynomial_degree = 3
   Grid_size = 30
   Lower_boundary = -1.0
   Upper_boundary =  1.0

   t_final = 1.0
   delta_t = 0.1

   Relative_tolerance = 1.0e-3
   Absolute_tolerance = 1.0e-3

   [DiffusionProblem]

   Kappa = 1.0
   SourceStrength = 3.5

``[configuration]`` is required and holds everything below. A physics case reads
its own parameters from its own table beside it — the name of that table is the
case's business, not the solver's.

.. note::

   **An unknown key in** ``[configuration]`` **is an error**, and the message
   names the nearest key in the schema::

      ERROR: Unknown configuration key 'Superconvergnet'. Did you mean 'Superconvergent'?

   That sweep covers ``[configuration]`` only. A physics case's own table is the
   case's business and is not checked against anything.

   A missing required key is an error too, and every one missing is reported in
   the same message rather than one per run.

Problem definition
------------------

.. list-table::
   :header-rows: 1
   :widths: 26 12 62

   * - Key
     - Default
     - Meaning
   * - ``TransportSystem``
     - *required, file only*
     - Name of the physics case to instantiate, as registered by
       ``REGISTER_PHYSICS_IMPL``. An unrecognised name is an error. A ``Runner``
       is handed the object instead, so passing this key to ``configure`` is an
       error rather than being ignored.
   * - ``PhysicsPlugins``
     - ``[]``, *file only*
     - Shared objects to ``dlopen`` before instantiating, for cases built
       outside this tree. See :doc:`out_of_tree`.
   * - ``Polynomial_degree``
     - *required*
     - The degree :math:`k` of the per-cell polynomial basis; each field carries
       :math:`k+1` coefficients per cell. ``Superconvergent`` requires
       :math:`k \ge 1`.
   * - ``Grid_size``
     - *required*
     - Number of cells.
   * - ``Lower_boundary``, ``Upper_boundary``
     - *required*
     - The ends of the domain. Required unless ``Grid_points`` is given or the
       run is a restart, both of which supply the cell boundaries themselves.
   * - ``Grid_points``
     - ``[]``
     - Explicit cell boundaries, as an array. Supersedes ``Lower_boundary``,
       ``Upper_boundary`` and ``Grid_size``.
   * - ``zeroFlux``
     - ``false``
     - What a **Neumann** boundary condition constrains: with this off the case's
       boundary value is imposed on the gradient :math:`q` (zero-gradient), with
       it on it is imposed on the flux :math:`\sigma` (zero-flux). Dirichlet
       boundaries are unaffected either way. It is a *global* flag, and it is now
       implemented as one: a Neumann end is assembled as a
       :ref:`mixed condition <mixed-boundaries>` with :math:`d = 1` when it is set
       and :math:`b = 1` when it is not, so the per-variable, per-end version of
       the same choice is available to a case through its own spec.
   * - ``FieldModel``
     - ``""``, *file only*
     - Name of a registered magnetic-field model to couple to, as registered by
       ``REGISTER_FIELD_MODEL_IMPL``. Absent — the default — means no coupling,
       and an uncoupled run's ``.nc`` is bit-for-bit what it was before the
       feature existed. Its ``.restart.nc`` is not, by exactly one scalar:
       ``RestartData/nField`` is written on every run, and reads back as ``0``.
       An unrecognised name is an error listing what *is* registered;
       note that no field model is registered in this tree yet. Like
       ``TransportSystem`` this is a ``ProblemSelection`` key, so passing it to
       ``Runner.configure`` is an error rather than being ignored. See
       :doc:`field_coupling`.
   * - ``High_Grid_Boundary``
     - ``false``
     - Refine the grid towards both ends instead of spacing cells uniformly.
   * - ``Lower_Boundary_Fraction``, ``Upper_Boundary_Fraction``
     - ``0.2``
     - With ``High_Grid_Boundary``, the fraction of the domain at each end that
       is refined.

Time integration
----------------

.. list-table::
   :header-rows: 1
   :widths: 26 12 62

   * - Key
     - Default
     - Meaning
   * - ``t_final``
     - *required in a file*
     - Time to integrate to. Optional in a dict, where ``Runner.run(tFinal)``
       supplies it; ``Runner.run()`` with no argument uses this key and raises
       if it was not given.
   * - ``t_initial``
     - ``0.0``
     - Time the run starts at. ``tZero`` is a deprecated spelling of this and
       still works, with a warning.
   * - ``delta_t``
     - *required*
     - **The output interval, not the timestep.** The loop advances ``tout`` by
       ``delta_t`` and asks IDA for the solution there; IDA chooses its own
       internal steps and takes as many as it needs. ``delta_t`` also seeds the
       initial step guess for ``IDACalcIC``.
   * - ``Relative_tolerance``
     - ``1e-3``
     - IDA's relative tolerance.
   * - ``Absolute_tolerance``
     - ``1e-3``
     - IDA's absolute tolerance. Either a single value or an array with one
       entry per variable.
   * - ``MinStepSize``
     - ``1e-7``
     - Steps smaller than this end the run.
   * - ``initialTimestep``
     - ``0.0``
     - First step to attempt. Zero lets IDA pick one.
   * - ``AggressiveTimesteps``
     - ``false``
     - Let IDA grow the step by up to 10× rather than 2× between steps. Useful
       when the transient is not the interesting part, at the cost of making IDA
       more likely to overshoot and retry. ``aggressiveTimesteps`` is a
       deprecated spelling of this and still works, with a warning.
   * - ``SteadyStateTolerance``
     - *unset*
     - If present, the run terminates when :math:`\mathrm{d}y/\mathrm{d}t` falls
       below this rather than at ``t_final``. It is the key's **presence** that
       arms that, not its value, on both surfaces. ``Runner.run_ss()`` arms it
       whether or not the key was given, and falls back to ``1e-3``.
   * - ``ObjectiveDecreaseTolerance``
     - ``0.0`` — off
     - If nonzero, the run is abandoned before the time loop when
       :math:`\mathrm{d}G/\mathrm{d}t < -` this at the initial condition. For an
       optimisation sweep that turns a step which was going to make the objective
       worse into the cost of initialisation alone. Requires ``solveAdjoint``,
       since the adjoint problem is what defines :math:`G`. Absolute, not
       relative — it carries the units of the objective over time, so there is no
       number worth defaulting to and zero means "off". A negative value is an
       error rather than a quiet "off". See :doc:`adjoints`.
   * - ``tau``
     - ``1.0``
     - The HDG stabilisation parameter. Constant across the domain.

Output
------

.. list-table::
   :header-rows: 1
   :widths: 26 12 62

   * - Key
     - Default
     - Meaning
   * - ``OutputFilename``
     - the config file's stem
     - Base name for every file the run writes. A dict has no file to take a
       name from, so this key is **required** by ``Runner.configure``.
   * - ``OutputPoints``
     - ``301``
     - Number of **spatial** points at which the solution is sampled for output.
       Independent of ``Grid_size``: the solution is a polynomial per cell, so it
       can be sampled as finely as you like.
   * - ``WriteOutput``
     - ``true``
     - Write ``<stem>.nc`` and ``<stem>.restart.nc``. Turn it off for a run whose
       result is read out of the process rather than off disk — an optimisation
       sweep evaluating an objective, say.
   * - ``WriteDatFile``
     - ``false``
     - Also write the plain-text gnuplot output ``<stem>.dat``.
   * - ``WriteDebugDatFiles``
     - ``false``
     - Also write ``<stem>.dydt.dat`` and ``<stem>.res.dat``. Additionally
       requires a ``PHYSICS_DEBUG`` build (``make DEBUG=on``).

``<stem>`` throughout is ``OutputFilename``. The two ``.dat`` options are
deliberately **not** nested under ``WriteOutput``: they are opt-in already, so a
config that asks only for ``WriteDatFile`` gets what it asked for.

Output lands in the **current working directory** whatever directory
``OutputFilename`` names — only the file-name part of it is used, so
``OutputFilename = "runs/case7"`` writes ``./case7.nc``. See :doc:`running`.

Restarting
----------

.. list-table::
   :header-rows: 1
   :widths: 26 20 54

   * - Key
     - Default
     - Meaning
   * - ``restart``
     - ``false``
     - Resume from a restart file instead of building an initial condition.
   * - ``RestartFile``
     - ``<stem>.restart.nc``
     - Which file to resume from. Unlike the output names, a path given here is
       used as it stands.

.. note::

   On a restart the grid and the polynomial degree are read from the restart
   file, not from the config file — ``Grid_size``, ``Polynomial_degree``,
   ``Lower_boundary`` and ``Upper_boundary`` are ignored on that path. Changing
   them in the config will not change the resumed run.

Adjoints and superconvergence
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 12 62

   * - Key
     - Default
     - Meaning
   * - ``solveAdjoint``
     - ``false``
     - Build the adjoint problem from the physics case and solve for
       :math:`\mathrm{d}G/\mathrm{d}p` after the forward integration. See
       :doc:`adjoints`.
   * - ``Superconvergent``
     - ``false``
     - Switch the residual and Jacobian to the superconvergent interpolatory
       scheme. Requires :math:`k \ge 1`, and is incompatible with spatial adjoint
       parameters. See :doc:`superconvergence`.

Coupling to a magnetic-field model
----------------------------------

These four are read whether or not a model is attached, and do nothing at all
without one — ``FieldModel``, under *Problem definition* above, is what turns
the coupling on.

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Key
     - Default
     - Meaning
   * - ``FieldSolve``
     - ``"iterative"``
     - How the coupled Jacobian is solved. ``iterative`` is block Gauss–Seidel
       between the transport and field blocks with Irons–Tuck acceleration, at
       one transport solve per sweep. ``exact`` forms the Schur complement onto
       the field block, at :math:`\texttt{nField}+1` transport solves per
       Jacobian solve.
   * - ``FieldSolveTolerance``
     - ``1e-8``
     - Where the sweep stops — **one key, two tests**, because the two
       directions have different things available to measure. Forward, it is a
       relative *change*: the sweep stops once
       :math:`\|\delta\psi\| \le \texttt{tol}\,\|\psi\|` for the unaccelerated
       iterate. In the adjoint it is a relative *backward error*: the field row
       of the transposed system holds identically at every iterate, so the
       residual of the pair returned is exactly
       :math:`A_2^{T}\,\delta z_\psi`, and the sweep stops once that is below
       ``tol`` times the norm of the right-hand side. Both are scale
       equivariant, so neither has an absolute floor, and neither returns an
       under-converged answer — reaching the cap escalates to the exact solve.
   * - ``FieldSolveMaxSweeps``
     - ``20``
     - Sweep cap for the forward solve. Reaching it escalates to the exact
       solve; it does not return an under-converged answer.
   * - ``FieldSolveMaxAdjointSweeps``
     - ``100``
     - The same cap for the adjoint solve. Separate, and larger, because the
       adjoint always runs at :math:`c_j = 0` where the coupling is stiffest —
       five field unknowns have been measured needing 13–38 sweeps.

.. important::

   **``FieldSolve = exact`` is a verification tool, not a slow mode.** It is what
   makes the coupled system checkable by finite-differencing the residual and
   requiring :math:`J\,\delta y = g`, and it is the oracle the iterative path is
   measured against.

   And the choice between them is a **cost** choice, never an accuracy one: the
   sweep escalates to the exact solve in both directions rather than guessing, so
   ``iterative`` can be slower than ``exact`` and can never be less accurate. On
   every fixture in this tree it *is* slower — see :doc:`field_coupling` for the
   break-even and the measured numbers.

.. The label is `config-divergences` for historical reasons: it named a section
   listing how the two readers disagreed, and docs/python.rst links to it.

.. _config-divergences:

One schema, two surfaces
------------------------

Both readers work from the same table. It is declared once, in
``ConfigSchema.cpp``, and read by ``loadSolverConfig`` (``SolverConfig.cpp``)
through a ``ConfigSource`` — ``TomlConfigSource`` for a file,
``DictConfigSource`` for a dict — after which one function,
``applySolverConfig``, applies the result to the solver. So a key has the same
name, the same type and the same default whichever surface it arrives on, and
an option cannot exist on one and not the other.

Deprecated spellings
~~~~~~~~~~~~~~~~~~~~

Two keys used to have a different name on each side, which is the sort of thing
one table prevents and two tables did not. Both old spellings still work, and
warn::

   WARNING: Configuration key 'tZero' is deprecated; use 't_initial'. Both are accepted for now.

.. list-table::
   :header-rows: 1
   :widths: 34 34 32

   * - Canonical
     - Deprecated
     - Which reader wanted it
   * - ``t_initial``
     - ``tZero``
     - ``Runner.configure``
   * - ``AggressiveTimesteps``
     - ``aggressiveTimesteps``
     - ``Runner.configure``

Giving a key under both spellings at once is an error rather than a silent
preference for one of them.

What is still asymmetric
~~~~~~~~~~~~~~~~~~~~~~~~

Three things, all deliberate:

* ``TransportSystem`` and ``PhysicsPlugins`` select the physics case. A
  ``Runner`` is handed the object instead, so passing either to ``configure``
  is an **error naming the reason**, rather than being accepted and ignored.
* ``t_final`` is required in a config file, which has nothing else to supply the
  end time with. It is optional in a dict: ``Runner.run(tFinal)`` overrides it,
  and ``Runner.run()`` with no argument uses it — a driver legitimately runs one
  configuration to many end times.
* ``OutputFilename`` falls back to the config file's stem when it is read from a
  file, and is required in a dict, where there is no file to take a name from.

The three ``PythonModule`` keys are a fourth, smaller case: they are read by the
``manta`` command rather than by the solver, and are in the schema so that a
config file carrying them is not rejected. See :doc:`out_of_tree`.

How a bad configuration is reported
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A configuration error is an exception, not a return code:

* ``manta.run("myrun.conf")`` raises ``ValueError``;
* ``Runner.configure`` raises ``RuntimeError``, which is what it has always
  raised — a driver already catching it goes on working;
* the ``MaNTA`` binary catches the same exception, prints one line to standard
  error and exits 1.

Conditions that are not about the configuration's *contents* keep reporting as
they did. ``runManta`` — the binary, and ``manta.run`` — logs an error and
returns 1 for a config file that does not exist, an unrecognised
``TransportSystem``, or a restart file that will not open; that number is the
binary's exit status and ``manta.run``'s return value. ``Runner.configure``,
which has no exit status to hand back, raises ``RuntimeError`` for the restart
file too.

Finally, ``./MaNTA --list-options`` prints the schema as it actually stands —
every key, its type, its default and a line of description. Prefer it to this
page when the two disagree, and fix the page.
