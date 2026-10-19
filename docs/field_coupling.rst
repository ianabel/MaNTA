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
adjoint solve and the ``.nc`` output are unchanged, bit for bit. The one
exception is the restart file, which gains ``RestartData/nField`` on every run,
coupled or not, so that a reader can tell "no field model" from "written before
the field block existed"; see :ref:`coupled-output`.
Measured over all 14 ``Tests/RegressionTests/*.conf``: every ``.nc`` byte
identical, every ``.restart.nc`` differing by exactly that one
``int nField = 0``.

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

.. note::

   **Names become netCDF names, and are checked as such where the check can say
   something useful.** ``validate()`` refuses a name netCDF would reject — one
   containing ``/`` or a control character, one padded with whitespace, one not
   beginning with a letter, digit or underscore — and refuses a DOF and a
   geometry slot that share a name, because the two lists go into *one* group.
   ``setFieldModel`` additionally refuses a group name that collides with a
   transport variable's group, an auxiliary variable, or one of the names the
   solver writes itself (``Grid``, ``RestartData``, ``x``, ``t``,
   ``nVariables``) — netCDF-4 gives groups and variables a single namespace, so
   those are collisions too.

   All of these are otherwise an ``NcBadName`` or ``NcNameInUse`` thrown from
   inside ``ncGroup.cpp`` at the *first write*, naming netCDF's own source and a
   line number rather than the spec that is wrong — and, for a long run, hours
   after the point where it was knowable.

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

The theta pinch
~~~~~~~~~~~~~~~

``ThetaPinchEquilibrium`` is the field model registered in this tree, and
``ThetaPinchDiffusion`` the transport case written against it. Together they
are a diamagnetic theta pinch in a flux-conserving wall:
``docs/superpowers/specs/2026-10-19-theta-pinch-field-model-design.md`` is the
design.

* The transport label is the normalised axial flux :math:`x = \Phi/\Phi_a`, so
  surfaces move in :math:`r` while their labels do not, and the equilibrium
  reaches the transport purely as metric coefficients on a fixed grid.
* Force balance is :math:`p + B^2 = C` on every surface (units with
  :math:`2\mu_0 = 1`, so :math:`p` is the local beta), and the wall fixes the
  area it encloses, :math:`\int_0^1 V' \,\mathrm{d}x = \pi a^2`. That
  integral is what determines :math:`C`.
* The unknowns are :math:`V' = \mathrm{d}A/\mathrm{d}x` as a cubic B-spline on
  the model's own grid of ``Intervals`` intervals, plus :math:`C`; the rows are
  Galerkin against the same splines. The geometry slots are :math:`V'`,
  :math:`\mathrm{d}V'/\mathrm{d}x`, :math:`\langle|\nabla x|^2\rangle` and
  :math:`B`.
* The pressure crosses as an auxiliary variable of the transport case, named
  by index (``PressureAux``), so the model knows nothing else about the
  transport. The case evolves :math:`N = V' n`, which keeps continuity exactly
  conservative while :math:`V'` moves.
* The block is banded with a one-row, one-column border, and the model
  overrides all four ``B`` operations with Eigen's ``SparseLU``. So ``nField``
  is set by ``Intervals`` independently of the transport's degree and mesh.

.. code-block:: toml

   [configuration]
   TransportSystem = "ThetaPinchDiffusion"
   FieldModel = "ThetaPinchEquilibrium"

   [ThetaPinchEquilibrium]
   Intervals = 32        # nField = Intervals + 4
   PressureAux = 0

   [ThetaPinchDiffusion]
   Temperature = 0.3
   D0 = 0.5
   Gamma = 2.0           # D = D0 (Bref / B)^Gamma

The two manufactured models under ``Tests/UnitTests`` (``ManufacturedField``
and ``ManufacturedFieldVector``) remain deliberately unregistered: they exist to
be checked against closed forms, not to be run.

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
:math:`\texttt{nField}+1`, plus an :math:`O(\texttt{nField}^3)` dense
factorisation of its Schur complement. Which side of it a problem is on is a
property of the coupling, and the tree has both:

* The manufactured fixtures, with :math:`\texttt{nField}` of 1 and 5, are on
  the losing side: iterative is about 1.5× more expensive than exact at
  :math:`\texttt{nField} = 1` and 2.2–6.3× at :math:`\texttt{nField} = 5`.
* The theta pinch is on the winning side at every resolution measured, because
  its sweep contracts strongly (spectral radius 0.10–0.19 at every
  :math:`c_j`) and takes about 7.5 sweeps per Jacobian solve whatever
  :math:`\texttt{nField}` is. On 16 cells at :math:`k = 4`, one BLAS thread,
  best of three:

  .. list-table::
     :header-rows: 1

     * - ``Intervals``
       - ``nField``
       - ``exact``
       - ``iterative``
       - speed-up
     * - 8
       - 12
       - 252 ms
       - 196 ms
       - 1.3×
     * - 32
       - 36
       - 583 ms
       - 228 ms
       - 2.6×
     * - 128
       - 132
       - 3948 ms
       - 496 ms
       - 8×
     * - 512
       - 516
       - 49185 ms
       - 1772 ms
       - 28×

  with the same step sequence in both modes and no fallbacks. That is the
  regime the iterative path exists for, :math:`N_\text{magnetics} \gg
  N_\text{HDG}`, where the exact solve's cost grows as
  :math:`\texttt{nField}^3` and the sweep's not at all.

The Irons–Tuck step extrapolates along the last increment, which is exact once
the sweep has settled into its dominant eigendirection and unjustified before.
A strongly non-normal iteration matrix can amplify and turn an increment several
times over before contracting -- the theta pinch's adjoint sweep grows its first
increment tenfold, at a cosine of :math:`-0.27` to the one before, while its
spectral radius is 0.14 -- so the step is taken only when two consecutive
increments are aligned to :math:`|\cos| \ge 0.9`, and the plain sweep
otherwise. The divergent couplings the acceleration exists for give
:math:`|\cos| > 0.99`.

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

``FieldSolveTolerance``, though, is shared, and it does not mean the same thing
in the two directions. Forward it bounds the relative *change* in :math:`\psi`
between sweeps; here it bounds a relative *backward error*. The field row above
holds identically at every iterate, so the residual of the pair the sweep
returns is exactly :math:`A_2^T \, \delta z_\psi` in the first row and nothing in
the second — which makes :math:`\|A_2^T \delta z_\psi\|`, compared against the
norm of the right-hand side, the backward error itself rather than a proxy for
it. That derivation is why the iterate *accepted* has to be the unaccelerated
``solveBTranspose`` output: returning an extrapolated :math:`z_\psi` leaves the
field row off by :math:`B^T \mu \, \delta`, and the stopping test silently
becomes an estimate.

**An objective may read the geometry.** :math:`\psi` reaches nothing outside
the field rows except through the geometry slots, so the explicit share of
:math:`G = \sum_j w_j\, g(\text{geom}(\psi, x_j))` is

.. math::

   \frac{\partial G}{\partial \psi_m} = \sum_j w_j \sum_s
     \left(\frac{\partial g}{\partial g_s}\right)_j
     \frac{\partial g_s}{\partial \psi_m}(x_j)

over the nodes :math:`G` is summed on, with the weights it is summed with. An
objective reports :math:`\partial g/\partial g_s` pointwise, through
``dgFn_dgeom`` or the geometry rows of ``dg``'s output -- the solver owns the
weights and ``dGeometry_dpsi``, so a case reports a derivative of :math:`g` and
never of :math:`G`. An objective that reads no geometry leaves them zero, which
is the right answer. ``AutodiffAdjointProblem::addGWithGeometry`` takes an
objective of :math:`(x, u, q, \sigma, \phi, \text{geom})` and differentiates
it for itself.

On a coupled run the solver evaluates :math:`G`, and its explicit
:math:`\partial G/\partial p`, on its own geometry-filled physics-node states
(``AdjointProblem::GFnAtNodes`` and ``dGFndpAtNodes``), since nothing else can
fill their geometry rows. The objective estimate and the ``"Objective"`` steady
test carry :math:`G_\psi \cdot \delta\psi` beside the cells' and the
scalars' shares.

One limit is structural rather than untested: **a field model cannot depend on
an adjoint parameter**, so :math:`\partial R/\partial p` is zero because there
is no hook a model could fill. It is recorded in ``TODO``.

Current restrictions
--------------------

* **``nScalars > 0`` with a field model is refused**, at ``setFieldModel``. The
  reason is a disagreement between two branches rather than a missing feature:
  the non-superconvergent ``dSources_dScalars`` coupling builds its ``State``
  from ``DGSoln::evalOnNode``, which has no geometry rows, while its
  superconvergent twin reads states that already carry them — so a case reading
  geometry there would work with ``Superconvergent = true`` and read out of
  bounds with it off.
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
