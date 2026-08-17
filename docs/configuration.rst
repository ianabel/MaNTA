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

   PolynomialDegree = 3
   GridSize = 30
   LowerBoundary = -1.0
   UpperBoundary =  1.0

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
   * - ``PolynomialDegree``
     - *required*
     - The degree :math:`k` of the per-cell polynomial basis; each field carries
       :math:`k+1` coefficients per cell. ``Superconvergent`` requires
       :math:`k \ge 1`.
   * - ``GridSize``
     - *required*
     - Number of cells.
   * - ``LowerBoundary``, ``UpperBoundary``
     - *required*
     - The ends of the domain. Required unless ``GridPoints`` is given or the
       run is a restart, both of which supply the cell boundaries themselves.
   * - ``GridPoints``
     - ``[]``
     - Explicit cell boundaries, as an array. Supersedes ``LowerBoundary``,
       ``UpperBoundary`` and ``GridSize``.
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
   * - ``GradedGridBoundary``
     - ``false``
     - Grade the grid *geometrically* towards the ends of the domain instead of
       spacing cells uniformly. See :ref:`graded-meshes` — on a problem whose
       solution is singular at an end this is worth orders of magnitude at an
       unchanged cell count. Replaces ``High_Grid_Boundary``, which spaced the same
       boundary layers by a cosine rule; that spelling still works and warns, but
       **the mesh it builds has changed**.
   * - ``GradingRatio``
     - ``0.3``
     - Width ratio between neighbouring cells in a graded layer, strictly between
       0 and 1. Smaller grades harder.
   * - ``GradingCells``
     - ``0``
     - Cells in *each* graded layer; at least 2, and few enough to leave one cell
       outside them. ``0`` means a third of ``GridSize`` per layer for ``"Both"``,
       half for a single end.
   * - ``GradingEnd``
     - ``"Both"``
     - Which end ``GradedGridBoundary`` refines into: ``"Both"``, ``"Lower"`` or
       ``"Upper"``.
   * - ``LowerBoundaryFraction``, ``UpperBoundaryFraction``
     - ``0.2``
     - The fraction of the domain at each end that is refined. Only the one
       belonging to a graded end is read, so the other may be left alone.

.. _graded-meshes:

Grading the mesh towards the ends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``GradedGridBoundary`` puts ``GradingCells`` cells over the layer of width
``LowerBoundaryFraction × (UpperBoundary − LowerBoundary)`` against each graded
end, each ``GradingRatio`` times the width of its inward neighbour, and spaces the
remaining cells uniformly over what is left. ``GradingEnd`` chooses which ends
get a layer; the default is both.

.. warning::

   The default, ``GradingEnd = "Both"``, is what ``High_Grid_Boundary`` meant and
   is not tuned to any particular problem — it is a boundary-layer mesh, not an
   optimised one. On Shestakov's problem, where the feature is at the lower end
   alone, the defaults at ``GridSize = 10`` give a **worse** answer than a uniform
   mesh (1.65e-02 against 4.91e-03) for two reasons worth knowing: half the graded
   cells are spent at the regular upper end, and the four remaining bulk cells no
   longer put a boundary on the source kink at :math:`x = 0.1`. Cell alignment with
   a kink is worth a factor of 4–7 on its own.

   So grading is a tool to aim, not a switch to flip. Name the end, and choose
   ``LowerBoundaryFraction`` so a cell boundary lands on any interior feature.

Grading one end, which is the configuration the measurements below were taken on:

.. code-block:: toml

   [configuration]
   GridSize = 10
   GradedGridBoundary = true
   GradingEnd = "Lower"
   GradingCells = 9              # nine cells inside the layer, one outside
   GradingRatio = 0.3
   LowerBoundaryFraction = 0.1   # the layer is the lower 10% of the domain

On ``[0, 1]`` that gives cell boundaries at 0, 6.56e-06, 2.19e-05, 7.29e-05,
2.43e-04, 8.10e-04, 2.70e-03, 9.00e-03, 0.03, 0.1, 1 — ten cells, the narrowest
against the axis and 137000 times narrower than the widest.

**The cell touching a graded end is the whole point, and its width is**

.. math::

   h_0 = \mathtt{fraction} \times (\mathtt{Upper} - \mathtt{Lower})
         \times \mathtt{GradingRatio}^{\,\mathtt{GradingCells} - 1}

Note this is *not* a pure geometric progression. That closing cell runs all the
way to the end, so it is :math:`1/(1-r)` wider than continuing the progression
would give; the first width ratio inside the layer is :math:`(1-r)/r` and every
later one is :math:`1/r`. That is what makes :math:`h_0` the clean expression
above, and it is what
``a_graded_mesh_puts_the_layer_cells_in_a_geometric_progression`` pins.

**Why bother.** On Shestakov's problem — an ``x^{4/3}`` singularity at the axis —
the error was measured to be :math:`0.0487\,h_0` and to depend on **nothing else**,
not even on the cell count, over a 4600× range of :math:`h_0` and across both
uniform and graded meshes. At a fixed 10 cells and ``PolynomialDegree = 5``, i.e.
an unchanged 60 degrees of freedom, that takes the relative error from 4.91e-03 to
3.29e-07: a factor of **14900**, for 1.5× the physics evaluations. For comparison,
*quadrupling* the budget to 40 uniform cells buys 4.0×. Redistributing cells is
worth far more here than adding them. ``MESH-REFINEMENT.md`` has the measurements.

.. note::

   **This replaced** ``High_Grid_Boundary``, which put a cosine-spaced layer
   against each end. That spelling is a deprecated alias of
   ``GradedGridBoundary`` and still loads, keeping its layer widths and its
   one-third-of-the-cells-per-layer split — but the cells *inside* each layer are
   now spaced geometrically, so **a file that used it will produce a different
   answer than it did before**. The run says so at ``WARNING``. Give
   ``GridPoints`` to reproduce a specific old mesh exactly.

Three things to expect:

* **A polynomial degree of about 4 or more is a prerequisite.** At ``k = 2`` the
  law above breaks once :math:`h_0` is small — an 11× reduction bought 1.19× —
  because the cells that are *not* the innermost stop resolving the singularity
  well enough to stay negligible.
* **The time integrator is the ceiling, not the discretisation.** Past roughly
  :math:`h_0 / \mathrm{span} = 10^{-6}` IDA's corrector starts failing at
  ``|h| = MinStepSize``, whose default is ``1e-7``; lowering it to ``1e-12`` bought
  one further level. Below about :math:`10^{-7}` nothing got through — neither
  ``MinStepSize``, nor ``SuppressAlgebraicError``, nor the tolerances, nor either
  continuation mode — and below :math:`10^{-8}` ``IDACalcIC`` cannot build a
  consistent initial condition at all. A warning is logged past
  :math:`10^{-6}` saying so, because the failure that follows points at IDA rather
  than at the mesh.
* **Grading into the upper end is slightly less exact than into the lower**, and
  irreducibly so: the boundary next to ``UpperBoundary`` is a number near it, so
  the narrowest cell's *width* is a difference of nearly equal numbers and carries
  a relative error of about :math:`\varepsilon / h_0`. It is 3e-11 at the grading
  above and only matters at far harder ones.

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
     - If present, the run terminates on reaching a steady state rather than at
       ``t_final``. It is the key's **presence** that arms that, not its value, on
       both surfaces. ``Runner.run_ss()`` arms it whether or not the key was
       given, and falls back to ``1e-3``. What it is compared against depends on
       ``SteadyStateSolver``: :math:`\mathrm{d}y/\mathrm{d}t` under ``TimeMarch``,
       and a mesh-independent weighted norm of the steady residual under the two
       continuation modes — see :ref:`steady-merit-function`.
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
       Independent of ``GridSize``: the solution is a polynomial per cell, so it
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
   file — ``GridSize``, ``GridPoints``, ``LowerBoundary`` and
   ``UpperBoundary`` are ignored on that path.

   ``PolynomialDegree`` is honoured. It defaults to the degree the file was
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
       parameters. See :doc:`superconvergence`.

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
