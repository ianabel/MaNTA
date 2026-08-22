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
   * - ``ConsistentICTolerance``
     - ``0.0`` — off
     - If nonzero, ``IDACalcIC`` is skipped when the initial state's weighted
       residual is already below this. It exists because ``IDACalcIC`` costs two
       Jacobian builds and two solves even when it has nothing to do, which is
       the usual case for a warm start. **Read** :ref:`warm-starts` **before
       setting it**: no single value was safe across the cases measured, and the
       default is that measurement rather than caution.
   * - ``SteadyStateSolve``
     - ``false``
     - Terminate when :math:`\mathrm{d}y/\mathrm{d}t` becomes small rather than at
       ``t_final``, using the default tolerance. Note that ``SteadyStateSolver``
       alone does **not** do this — it names the method, which is only consulted
       once termination is armed.
   * - ``SteadyStateTolerance``
     - *unset*
     - The same, naming the tolerance. It is the key's **presence** that arms
       termination, not its value, on both surfaces; giving it alongside
       ``SteadyStateSolve`` simply sets the tolerance.
       ``Runner.run_ss()`` arms it whether or not either key was given, and falls
       back to ``1e-3``.
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

   On a restart the grid is read from the restart file, not from the config
   file — ``Grid_size``, ``Grid_points``, ``Lower_boundary`` and
   ``Upper_boundary`` are ignored on that path.

   ``Polynomial_degree`` is honoured. It defaults to the degree the file was
   written at, and a different value resumes at that degree, projecting the
   stored state across the change and logging a warning. See
   :doc:`running` for what refining and coarsening each cost.

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
       parameters. See :doc:`superconvergence`. ``DegreeAdaptation`` turns this
       on; setting it to ``false`` alongside that is refused rather than
       overridden, so leave it out if you want the default.
   * - ``SteadyStateSolve``
     - ``false``
     - Run to a steady state with the default tolerance. ``SteadyStateTolerance``
       does the same and names the tolerance; either arms it, and giving both
       uses the tolerance. Without one of them a run time-marches, whatever
       ``SteadyStateSolver`` says.
   * - ``DegreeAdaptation``
     - ``false``
     - Choose the global polynomial degree by solving, estimating the error from
       :math:`u^* - u_h`, and re-solving at a higher degree. Steady solves only.
       See :ref:`degree-adaptation`.
   * - ``DegreeTolerance``
     - ``1e-6``
     - Target for ``DegreeAdaptation``: the estimated :math:`L^2` error relative
       to the solution's own :math:`L^2` norm, worst variable, floored by
       ``Absolute_tolerance``.
   * - ``MaxPolynomialDegree``
     - ``10``
     - Ceiling on the degree ``DegreeAdaptation`` may reach. Reaching it warns
       and returns the best result available rather than failing.
   * - ``MaxDegreeIncrement``
     - ``3``
     - Most degrees a single step may add. Giorgiani's rule can ask for a large
       jump from a coarse first solve; capping it keeps the loop taking steps it
       can report on. Minimum 1 — zero would leave it re-solving a degree it is
       not allowed to raise.
   * - ``DegreeAdaptationBase``
     - ``10``
     - How much error one extra degree is assumed to buy, in
       :math:`\Delta k = \lceil \log_b(E/\epsilon) \rceil`. Between 10 and 100;
       larger is more aggressive and so asks for fewer degrees.

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
