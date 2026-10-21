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
* The unknowns are :math:`V' = \mathrm{d}A/\mathrm{d}x` as a :math:`C^2` cubic
  B-spline whose knots are the transport mesh's faces, each cell split into
  ``IntervalsPerCell`` equal intervals (default 1), plus :math:`C`; the rows are
  Galerkin against the same splines. The geometry slots are :math:`V'`,
  :math:`\mathrm{d}V'/\mathrm{d}x`, :math:`\langle|\nabla x|^2\rangle` and
  :math:`B`.
* The pressure crosses as an auxiliary variable of the transport case, named
  by index (``PressureAux``), so the model knows nothing else about the
  transport. The case evolves :math:`N = V' n`, which keeps continuity exactly
  conservative while :math:`V'` moves.
* The block is banded with a one-row, one-column border, and the model
  overrides all four ``B`` operations with Eigen's ``SparseLU``. ``nField`` is
  :math:`n_\text{cells} \times` ``IntervalsPerCell`` :math:`+ 4`: independent
  of the transport's degree, so a restart onto another degree carries
  :math:`\psi` verbatim, but not of its mesh, so a restart onto another mesh is
  refused by name.

**Why the knots follow the transport mesh, and why the spline stays smooth
across it.** The pressure arrives as the transport's piecewise polynomial
:math:`p_h`, which the rows integrate exactly cell by cell, and :math:`p_h`
jumps by :math:`O(h^{k+1})` at every transport face. A spline much finer than
the mesh resolves those jumps, which puts spikes of about the jump over the knot
spacing into :math:`\mathrm{d}V'/\mathrm{d}x` at the transport nodes beside
each face -- and :math:`\mathrm{d}V'/\mathrm{d}x` goes straight into the flux.
Measured on the manufactured theta pinch (``ThetaPinchMMSTests.cpp``):

* one or two intervals per cell converge at :math:`k+1` in :math:`N` for
  :math:`k = 1, 2, 3`, and four per cell at :math:`k = 1` drops to order
  :math:`k`;
* IDA stalls at order one -- under ``FieldSolve = exact`` as much as
  ``iterative`` -- at two per cell on 64 cells at :math:`k = 3`, and at sixteen
  per cell on 8 at :math:`k = 6`; at one per cell the exact solve never has;
* letting :math:`V'` break where :math:`p_h` does (repeated knots at the faces)
  is worse at every :math:`k`, about order :math:`k`: :math:`V'` then inherits
  :math:`p_h`'s derivative error, :math:`O(h^k)`, cell by cell, where a smooth
  spline averages it away.

So one interval per cell is the default, and ``IntervalsPerCell`` is for a
high-degree run in which the spline's own :math:`O(h^4)` is what limits the
answer.

.. code-block:: toml

   [configuration]
   TransportSystem = "ThetaPinchDiffusion"
   FieldModel = "ThetaPinchEquilibrium"

   [ThetaPinchEquilibrium]
   IntervalsPerCell = 1  # nField = cells * IntervalsPerCell + 4
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
   acceleration, at one transport solve per sweep. It stops once a sweep changes
   the whole increment -- transport and field together -- by less than
   ``FieldSolveTolerance`` (default :math:`10^{-4}`) in the run's error-weighted
   RMS norm, the one IDA's tests are in, where its Newton iteration stops at
   0.33; or by less than a millionth of the increment, when that is larger.

**What that test is measured in, and against what, were both settled by what
each costs IDA**, on the manufactured theta pinch at :math:`10^{-9}/10^{-11}`
and on the theta pinch in ``ThetaPinchRunTests.cpp`` at
:math:`10^{-6}/10^{-8}`.

* *In the run's weights, over the whole increment.* A relative 2-norm test on
  :math:`\psi` alone leaves an error in the transport block, through
  :math:`A^{-1} A_1 \delta\psi`, that is negligible in the 2-norm and not in
  IDA's: :math:`\sigma` and :math:`q` near the axis sit near zero, where the
  weights are about ``1/atol``. Newton then converges only linearly, the
  leftover lands in IDA's local error estimate, and it holds order one at a
  tiny step: the manufactured theta pinch at :math:`k = 3`, 8 cells and
  :math:`10^{-9}/10^{-11}` took more than 20000 steps against the exact solve's
  213, and tightening that test to :math:`10^{-11}` did not cure it.
* *Absolute, and well inside Newton's own tolerance.* IDA estimates Newton's
  rate as :math:`\|\delta_m\| / \|\delta_0\|` in the same norm, fails a step
  above 0.9, and accepts a corrector after one iteration on the strength of the
  last rate, so the solve's error has to be small against a late update and
  against the first. At :math:`10^{-2}` the run test took 658 steps against the
  exact solve's 166. A bound relative to the increment does not fix that: at
  :math:`10^{-3}` of the increment a first update of size ~2 carries an error of
  :math:`2\times10^{-3}`, and :math:`k = 4` on 8 cells held order one for 29000
  steps against 214. At :math:`10^{-4}` absolute every case measured takes steps
  comparable to the exact solve's -- 166 against 166, 221 against 213, 278
  against 214 -- with the same answers.
* *...or a millionth of the increment.* ``IDACalcIC``'s first step, at
  :math:`c_j \sim 10^9`, has an increment of about :math:`4\times10^8`
  error-weight units, where :math:`10^{-4}` is :math:`10^{-13}` of it and
  twenty sweeps cannot reach it: every run escalated to the exact solve exactly
  once. The relative arm governs only above 100 units, where an error of
  :math:`10^{-6}` relative moves Newton's rate estimate by nothing it can see.

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
  the losing side: they run 2.0–2.8 sweeps per Jacobian solve against an
  :math:`\texttt{nField}+1` of 2 to 6.
* The theta pinch is on the winning side at every resolution measured, because
  its sweep contracts strongly (spectral radius 0.10–0.19 at every
  :math:`c_j`) and takes about 3.7 sweeps per Jacobian solve whatever
  :math:`\texttt{nField}` is. Its field unknowns follow the transport mesh, so
  :math:`\texttt{nField}` grows with the cell count, and the exact solve's
  :math:`\texttt{nField}+1` transport solves grow with it. ``ThetaPinch.conf``
  at :math:`k = 4` and one interval per cell, one BLAS thread, best of three:

  .. list-table::
     :header-rows: 1

     * - cells
       - ``nField``
       - ``exact``
       - ``iterative``
       - speed-up
     * - 8
       - 12
       - 151 ms
       - 103 ms
       - 1.5×
     * - 32
       - 36
       - 1288 ms
       - 338 ms
       - 3.8×
     * - 128
       - 132
       - 24553 ms
       - 2076 ms
       - 12×

  with the same step sequence in both modes and no fallbacks. The exact path
  costs :math:`O(n_\text{cells}^2)` per Jacobian solve and the sweep
  :math:`O(n_\text{cells})`, so the gap widens with the mesh.

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

The tolerance is separate too, ``FieldSolveAdjointTolerance`` (default
:math:`10^{-8}`), because the two directions measure different things. Forward
there is IDA's error test to measure against; here there is a gradient and no
error test, and the tolerance bounds a relative *backward error*. The field row above
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
