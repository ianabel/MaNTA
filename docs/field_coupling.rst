.. _field-coupling:

Self-consistent magnetic fields
===============================

A transport model that is meant to be predictive cannot take its magnetic
geometry as given: the pressure profile it computes changes the equilibrium, and
the equilibrium changes the transport coefficients. MaNTA lets a **field model**
join the system as a coupled algebraic block, so that the geometry the physics
reads and the profiles that determine it are solved for together, in one Newton
iteration, rather than in an outer loop.

Nothing about this is on unless a run asks for it. With no field model attached
— which is every run that predates this feature and every configuration that
does not set ``FieldModel`` — the solution vector, the Jacobian solve, the
adjoint solve and the output are unchanged, bit for bit.

The coupled system
------------------

A field model supplies ``nFieldDOF`` unknowns :math:`\psi` and one residual row
each,

.. math::

   R_m(\psi, \dot\psi, u, q, \sigma, \phi, t) = 0,
   \qquad m = 1 \ldots \texttt{nFieldDOF}

together with a map from :math:`\psi` to the ``nGeometry`` *geometry slots*
:math:`g_s(\psi, x, t)` that the transport physics reads. So the coupling is
two-way and both directions go through a declared interface: the field rows may
read the transport solution, and the transport hooks may read the geometry, but
neither reaches into the other's unknowns directly.

Written as blocks, the Jacobian is

.. math::

   \begin{pmatrix} A & A_1 \\ A_2 & B \end{pmatrix}
   \begin{pmatrix} \delta y \\ \delta\psi \end{pmatrix}
   = \begin{pmatrix} r_y \\ r_\psi \end{pmatrix}

with :math:`A` the HDG operator MaNTA already static-condenses,
:math:`A_1 = \partial(\text{transport residual})/\partial\psi`,
:math:`A_2 = \partial R/\partial y`, and :math:`B = \partial R/\partial\psi +
\alpha \, \partial R/\partial\dot\psi` the model's own block. See
:ref:`field-solve` below for how that is solved.

:math:`A_1` is a chain rule and the two factors come from opposite sides of the
interface:

.. math::

   \frac{\partial(\text{row})}{\partial \psi_m}
     = \sum_s \frac{\partial(\text{row})}{\partial g_s}
              \frac{\partial g_s}{\partial \psi_m}

The first factor is the physics case's — the three hooks in
:ref:`geometry-derivatives` — and the second is the field model's
``dGeometry_dpsi``. A case that does not read geometry leaves all three
unimplemented and contributes an identically zero block, which is correct rather
than approximate: it does not couple.

Degrees of freedom
------------------

The field unknowns are appended to the solution vector **after** the global
scalars, so nothing before them moves:

.. code-block:: text

   [ sigma | q | u | aux ]   per cell, for each cell in turn
   [ lambda ]                all face traces
   [ mu ]                    all global scalars
   [ psi ]                   the field model's unknowns

``DGSoln::getDoF()`` accounts for all of it and is the authority on the length —
the formula used to be open-coded in three places, and a copy that did not know
about ``nField`` wrote a *short* restart file whose recorded ``nDOF`` matched the
uncoupled formula, so the truncated file read back as consistent.

Each field DOF declares whether it is differential or algebraic, and that
declaration reaches ``IDASetId``. Declaring a DOF differential when its residual
row carries no :math:`\dot\psi` is refused at ``initialize()``, by name — left to
IDA it is an ``IDA_LINESEARCH_FAIL`` (-13), a message about the linesearch for a
defect in the declaration. See :doc:`running`.

The geometry slots are **not** unknowns. They are a function of
:math:`(\psi, x)` evaluated at the physics nodes and cached per residual, in the
same standing as :math:`\hat\sigma`. A physics case reads slot ``s`` through
``State::geom(s)``.

Writing a field model
---------------------

A model derives from ``FieldModel`` (``FieldModel.hpp``) and, like a physics
case, **declares itself as data**: the only constructor takes a
``FieldModelSpec``, which is validated there, so a part-built model cannot
exist.

.. code-block:: cpp

   class MyEquilibrium : public FieldModel
   {
   public:
       MyEquilibrium(toml::value const &config, Grid const &grid)
           : FieldModel(buildSpec()) { /* read your own table from config */ }

       static FieldModelSpec buildSpec()
       {
           FieldModelSpec s;
           s.dofs     = {{"psi_axis", "poloidal flux on axis", "Wb", false}};
           s.geometry = {{"V_prime", "flux-surface volume element", "m^3"}};
           s.label    = "rho_toroidal";   // what this model's x means
           s.name     = "Equilibrium";    // its netCDF group
           return s;
       }
       ...
   };

``label`` names the spatial coordinate the geometry is expressed against. MaNTA
does not interpret it — the model declares its own coordinate and supplies the
metric on it — but it is recorded in the output so a run says what its :math:`x`
meant. ``name`` is the netCDF group psi and the geometry slots are written to,
and defaults to ``Field``.

The hooks
~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Hook
     - What it must do
   * - ``FieldResidual``
     - The constraint rows. Receives ``psi``, ``dpsidt``, the transport solution
       sampled on the physics nodes as a ``GlobalState``, the abscissae, and one
       quadrature weight per node — so :math:`\int f \, \mathrm{d}x` is
       ``weights.dot(f_at_nodes)``. **Use those weights rather than a rule of
       your own**; see the same warning for ``ScalarG`` in
       :doc:`physics_interface`.
   * - ``Geometry``
     - The metric at one point, length ``nGeometry``, arriving zeroed.
   * - ``dGeometry_dpsi``
     - Shape ``(nGeometry, nFieldDOF)``, arriving zeroed.
   * - ``FieldResidualPrime``
     - Every field row's derivative at once, in the shape ``ScalarGPrime`` uses:
       ``dR`` against the transport DOFs, ``dRdpsi`` and ``dRddpsidt`` against
       the model's own block. Reporting every row at once is deliberate — it is
       what lets a model that solves a coupled system internally do so once.
   * - ``InitialFieldValue``
     - The starting guess for psi. Not called on a restart, where psi is resumed
       from the file.
   * - ``resetForRun``
     - Discard anything cached for one run. See below.

Four more — ``updateFieldJacobian``, ``applyB``/``applyBTranspose`` and
``solveB``/``solveBTranspose`` — have defaults that store :math:`B` densely and
factorise it with a partial-pivot LU. That is right for a small block and is
what the manufactured test models use. **A model with a large or structured
block overrides all four**, and that is the seam a real Grad–Shafranov solver
plugs into: MaNTA never needs :math:`B` itself, only the ability to apply and
invert it in both directions.

Two things a model author is likely to get wrong
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**``dRdot`` cannot be filled today, and leaving it zero is correct.**
``FieldResidual`` receives ``states`` and no ``states_dot`` — unlike
``ScalarG``, which takes both — so a field row has no way to depend on the
transport time derivatives in the first place. The slot exists because the
coupling assembly already weights it by :math:`\alpha`. What you must **not** do
on finding it unfillable is put :math:`\partial R/\partial\dot\psi` there
instead; that belongs in ``dRddpsidt``, and written into ``dRdot`` it lands in
the coupling row at the wrong degrees of freedom entirely, with nothing to say
so.

**``resetForRun`` is not optional if the model caches anything.**
``SystemSolver::initialize`` skips ``initialiseMatrices`` when the solver has
already been initialised, so anything computed once per object rather than once
per run is stale on the second run — the trap that made a reused solver take its
initial condition from the previous run's final state. ``resetForRun`` is called
from the unconditional part of ``initialize`` for exactly that reason, and
``a_coupled_solver_reused_matches_a_fresh_one_bit_for_bit`` is what stands
between a model that ignores it and a second run that completes, looks
plausible, and is wrong.

Failing gracefully
~~~~~~~~~~~~~~~~~~

A model that **cannot evaluate at this state** — no x-point, a boundary that has
left the domain — must throw from ``FieldResidual``. ``static_residual`` catches
it and returns 1, which IDA treats as a *recoverable* error and retries with a
smaller step. Returning a nonsense value instead turns a recoverable step into a
wrong answer.

Registering and selecting one
-----------------------------

Registration follows the physics-case pattern exactly:
``REGISTER_FIELD_MODEL_HEADER(T)`` in the header,
``REGISTER_FIELD_MODEL_IMPL(T)`` in the ``.cpp``. A model only appears if its
object file is linked in — nothing references it directly, so a missing entry is
a link-line problem with no compile error — and a duplicate name throws rather
than being quietly dropped. An unknown name in a config throws with the list of
what *is* registered.

A run selects one by name:

.. code-block:: toml

   [configuration]
   FieldModel = "Equilibrium"

.. note::

   **No field model is registered in this tree.** The two that exist
   (``ManufacturedField`` and ``ManufacturedFieldVector``) are test fixtures
   under ``Tests/UnitTests`` and are deliberately unregistered, so there is
   nothing for ``FieldModel`` to name in the shipped binary yet. The whole
   coupling is exercised by the unit suite instead; see ``Tests/README.md``.

.. _field-solve:

How the coupled Jacobian is solved
----------------------------------

``FieldSolve`` picks between two routes to the same answer.

``exact``
   Form the Schur complement onto :math:`\psi` by applying the transport inverse
   to every column of :math:`A_1`. That is :math:`\texttt{nField} + 1` transport
   solves per Jacobian solve, so it is a **verification tool**: it is what makes
   the coupled system checkable by the ``SolveJacTests`` method — finite
   difference the residual, require :math:`J \, \delta y = g` — and it is the
   oracle the iterative path is compared against.

``iterative`` (the default)
   Block Gauss–Seidel between the transport and field blocks, with Irons–Tuck
   acceleration, at one transport solve per sweep. It stops once the relative
   change in :math:`\psi` is below ``FieldSolveTolerance``.

.. important::

   **``iterative`` is a cost choice, never an accuracy one.** A sweep that
   exhausts its cap escalates to the exact Schur solve rather than returning an
   under-converged answer, and it does so in **both** directions — forward and
   adjoint. So the mode can be *slower* than ``exact`` in the worst case, paying
   the sweeps and then the exact solve on top, and can never be less accurate.

The break-even is

.. math::

   \#\text{sweeps} < \texttt{nField} + 1

because a sweep costs one transport solve and the exact solve costs
:math:`\texttt{nField}+1`, and **no fixture in this tree is on the winning side
of it**: iterative is about 1.5× more expensive than exact at
:math:`\texttt{nField} = 1` and 2.2–6.3× at :math:`\texttt{nField} = 5`, for the
same answer. The iterative path is a bet on the regime this feature exists for,
:math:`N_\text{magnetics} \gg N_\text{HDG}`, where the exact solve's
:math:`\texttt{nField}+1` transport solves and its :math:`O(\texttt{nField}^3)`
dense factorisation are hopeless. Nothing in the tree exercises that regime yet.

Adjoints
--------

The coupled adjoint is the transpose, block for block: the elimination runs the
other way round and the Schur complement onto :math:`\psi` becomes

.. math::

   \left( B^T - A_1^T A^{-T} A_2^T \right) z_\psi &= G_\psi - A_1^T A^{-T} G_y \\
   A^T z_y &= G_y - A_2^T z_\psi

which is why ``FieldModel`` declares ``applyBTranspose`` and ``solveBTranspose``
beside the forward pair: a model supplying only one direction cannot be silently
accommodated. ``FieldSolveMaxAdjointSweeps`` is separate from
``FieldSolveMaxSweeps`` and defaults larger — 100 against 20 — because the
adjoint always runs at :math:`c_j = 0`, where the coupling is stiffest.

Two limits are structural rather than untested, and both are zero today by
construction rather than by assumption:

* **An objective whose integrand reads ``State::geom`` directly loses its
  :math:`\mathrm{d}G/\mathrm{d}\psi` term.** ``AdjointProblem`` reports
  :math:`\partial g/\partial u`, :math:`\partial q`, :math:`\partial\sigma` and
  :math:`\partial\phi`, and geometry is not among them.
* **A field model cannot depend on an adjoint parameter**, so
  :math:`\partial R/\partial p` is zero.

Both are recorded in ``TODO`` and beside ``G_field`` in ``SystemSolver.hpp``.

Current restrictions
--------------------

* **``nScalars > 0`` with a field model is refused**, at ``setFieldModel``. The
  reason is a disagreement between two branches rather than a missing feature:
  the non-superconvergent ``dSources_dScalars`` coupling builds its ``State``
  from ``DGSoln::evalOnNode``, which has no geometry rows, while its
  superconvergent twin reads states that already carry them — so a case reading
  geometry there would work with ``Superconvergent = true`` and read out of
  bounds with it off.
* **``Superconvergent = true`` with spatial adjoint parameters** throws, as it
  does without a field model: the star node set would redefine how many
  parameters there are.
* **A field model cannot be written in Python.** ``FieldModel`` has no pybind11
  class, and ``FieldModel`` is a ``ProblemSelection`` key, so it is an error in
  a ``Runner.configure`` dict. A coupled run is a config-file run.

Two dimensions, and beyond
--------------------------

What is implemented is the *coupling*, not an equilibrium solver: MaNTA supplies
the block structure, the coupled Jacobian and adjoint solves, the DOF
bookkeeping and the serialisation, and a model supplies the physics. A 2-D
Grad–Shafranov solve, or a DESC stellarator equilibrium, plugs in by overriding
the four ``B`` operations above so that MaNTA never forms the model's Jacobian
at all. Neither exists yet.
