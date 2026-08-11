Configuration
=============

There are two ways to configure a run, and they are **not the same interface**:

* a **TOML file** given to the ``MaNTA`` binary, described here and read by
  ``runManta`` in ``MaNTA.cpp``;
* a **Python dict** passed to ``Runner.configure``, read by ``PyRunner.cpp``.

Most keys are common to both. Those that are not, and those whose names or
defaults differ, are collected in :ref:`config-divergences` below — check there
before assuming a key carries across.

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

.. warning::

   **Unknown keys are silently ignored.** Every optional key is read through
   ``toml::find_or``, so a misspelled or obsolete key is inert rather than an
   error. If a setting appears to have no effect, check its spelling and its
   capitalisation first: ``AggressiveTimesteps`` and ``aggressiveTimesteps`` are
   different keys on different interfaces, and neither warns about the other.

Problem definition
------------------

.. list-table::
   :header-rows: 1
   :widths: 26 12 62

   * - Key
     - Default
     - Meaning
   * - ``TransportSystem``
     - *required*
     - Name of the physics case to instantiate, as registered by
       ``REGISTER_PHYSICS_IMPL``. An unrecognised name is an error. Not used by
       the Python interface, which is handed the object directly.
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
     - The ends of the domain.
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
     - *required*
     - Time to integrate to.
   * - ``t_initial``
     - ``0.0``
     - Time the run starts at.
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
     - ``1e-2``
     - IDA's absolute tolerance. Either a single value or an array with one
       entry per variable. Note the default differs between the two interfaces —
       see :ref:`config-divergences`.
   * - ``MinStepSize``
     - ``1e-7``
     - Steps smaller than this end the run.
   * - ``AggressiveTimesteps``
     - ``false``
     - Let IDA grow the step by up to 10× rather than 2× between steps. Useful
       when the transient is not the interesting part, at the cost of making IDA
       more likely to overshoot and retry.
   * - ``SteadyStateTolerance``
     - *unset*
     - If present, the run terminates when :math:`\mathrm{d}y/\mathrm{d}t` falls
       below this rather than at ``t_final``.
   * - ``ObjectiveDecreaseTolerance``
     - *unset*
     - If present, the run is abandoned before the time loop when
       :math:`\mathrm{d}G/\mathrm{d}t < -` this at the initial condition. For an
       optimisation sweep that turns a step which was going to make the objective
       worse into the cost of initialisation alone. Requires ``solveAdjoint``,
       since the adjoint problem is what defines :math:`G`. Absolute, not
       relative — it carries the units of the objective over time, which is why
       there is no default. See :doc:`adjoints`.
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
   * - ``OutputPoints``
     - ``301``
     - Number of **spatial** points at which the solution is sampled for output.
       Independent of ``Grid_size``: the solution is a polynomial per cell, so it
       can be sampled as finely as you like.
   * - ``WriteDatFile``
     - ``false``
     - Also write the plain-text gnuplot output ``<stem>.dat``.
   * - ``WriteDebugDatFiles``
     - ``false``
     - Also write ``<stem>.dydt.dat`` and ``<stem>.res.dat``. Additionally
       requires a ``PHYSICS_DEBUG`` build (``make DEBUG=on``).

netCDF output is written unconditionally and is not configurable here; the file
names come from the *config file's stem*. See :doc:`running`.

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
     - Which file to resume from.

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

.. _config-divergences:

Differences between the TOML file and ``Runner.configure``
----------------------------------------------------------

The Python interface takes a dict rather than a file. It is not a wrapper around
the TOML reader, and the two lists of keys have drifted:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * -
     - TOML file
     - ``Runner.configure`` dict
   * - initial time
     - ``t_initial``
     - ``tZero``
   * - aggressive stepping
     - ``AggressiveTimesteps``
     - ``aggressiveTimesteps``
   * - final time
     - ``t_final``, a required key
     - not a key — an argument to ``run(tFinal)``
   * - physics case
     - ``TransportSystem``, required
     - not a key — the object is passed to ``configure``
   * - ``Absolute_tolerance`` default
     - ``1e-2``
     - ``1e-3``
   * - output file name
     - not read at all; names derive from the config file's stem
     - ``OutputFilename``, **required**
   * - explicit cell boundaries
     - not supported
     - ``Grid_points``, a list of boundaries; ``Lower_boundary`` and
       ``Upper_boundary`` are then unused
   * - ``WriteOutput``
     - not read (output is always written)
     - supported, default ``true``
   * - ``zeroFlux``, ``initialTimestep``
     - not read
     - supported
   * - ``SteadyStateTolerance``
     - absent means "integrate to ``t_final``"
     - always present, default ``1e-3``; ``run_ss`` is what uses it
   * - ``ObjectiveDecreaseTolerance``
     - absent means the gate is off
     - always present, default ``0.0``, which also means off; a negative value is
       an error rather than a quiet "off"

Everything else is shared, with the same name and the same default:
``restart``, ``RestartFile``, ``High_Grid_Boundary``,
``Lower_Boundary_Fraction``, ``Upper_Boundary_Fraction``, ``Polynomial_degree``,
``Grid_size``, ``Lower_boundary``, ``Upper_boundary``, ``tau``, ``delta_t``,
``Relative_tolerance``, ``MinStepSize``, ``OutputPoints``, ``solveAdjoint``,
``WriteDatFile``, ``WriteDebugDatFiles`` and ``Superconvergent``.

The dict interface is declarative — ``PyRunner.cpp`` opens with a table of
``Parameter`` entries carrying ``.required`` and ``._default`` — so that table is
the authority on the Python side, as ``MaNTA.cpp`` is on the TOML side.
