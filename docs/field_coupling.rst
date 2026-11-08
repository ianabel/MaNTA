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

A field model can be attached only to a case whose spec sets
``readsGeometry = true`` (see :ref:`geometry-derivatives`). ``setFieldModel``
refuses any other, naming the model: a field that cannot reach the solution
would be solved for at every Newton iteration and change nothing. It also
refuses a case whose ``geometryDerivatives`` is ``None``, since the coupled
Jacobian would be built on zero blocks; such a case runs its field as a
:ref:`black box <field-tier-0>` instead. With no field
model attached the declaration changes nothing — no geometry rows, no geometry
evaluation — which is what keeps every uncoupled run bit for bit what it was.

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
       :doc:`physics_interface`. A row that reads :math:`\partial u/\partial t`
       declares ``readsTransportTimeDerivatives`` in its spec and reads it as
       ``states[j].udot(v)``; see below.
   * - ``Geometry``
     - The metric at one point, length ``nGeometry``, arriving zeroed.
   * - ``dGeometry_dpsi``
     - Shape ``(nGeometry, nFieldDOF)``, arriving zeroed.
   * - ``FieldResidualPrime``
     - Every field row's derivative at once, in the shape ``ScalarGPrime`` uses:
       ``dR`` against the transport DOFs, ``dRdot`` against their time
       derivatives, ``dRdpsi`` and ``dRddpsidt`` against the model's own block. Reporting every row at once is deliberate — it is
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

A field row that reads the transport time derivatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A row may depend on :math:`\partial u/\partial t`, as a transport source may.
The model declares it in its spec, ``readsTransportTimeDerivatives = true``, and
then every ``GlobalState`` the solver hands it — to ``FieldResidual``,
``FieldResidualPrime``, ``dFieldResidual_dt`` and an adjoint problem's
``dFieldResidual_dp`` — carries :math:`\partial u/\partial t` at the nodes,
read pointwise as ``states[j].udot(v)``. Undeclared, the states carry none and
``udot`` reads zero. Only :math:`u`'s derivative is available, for the reason
the transport sources give: :math:`u` is the only transport quantity IDA treats
as differential.

The derivative goes in ``dRdot[f].Variable()``, and the coupling weights it by
:math:`\alpha` — so it is invisible to a Jacobian check at :math:`\alpha = 0`,
and to the adjoint, which is built there. Such a field is not a function of the
transport state, so it runs coupled: ``FieldCoupling = "auto"`` couples it, and
``"consistent"`` and ``"reduced"`` refuse it, as they refuse a differential
field unknown.

Two things a model author is likely to get wrong
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**:math:`\partial R/\partial\dot\psi` goes in ``dRddpsidt``, never in
``dRdot``.** ``dRdot`` is the derivative in the *transport* time derivatives;
written there, a derivative in the field's own lands in the coupling row at the
wrong degrees of freedom entirely, with nothing to say so — the coupling row is
only ever applied, never assembled or printed.

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

``FieldSolve`` picks between three routes to the same answer.

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

``krylov``
   GMRES -- SUNDIALS' SPGMR -- on the field increment, the system the sweep is
   a fixed-point iteration on:

   .. math::

      \left(I - B^{-1} A_2 A^{-1} A_1\right) \delta\psi
      = B^{-1}\left(r_2 - A_2 A^{-1} r_1\right),
      \qquad \delta x = A^{-1}(r_1 - A_1\,\delta\psi).

   One transport solve per iteration, and no product with :math:`A`, which is
   never assembled. Its residual is exactly the change one more sweep would
   make to :math:`\delta\psi`, and SPGMR is scaled by :math:`\psi`'s error
   weights, so it stops on the part of the sweep's test that binds -- the field's
   weights are what keep the sweep converging far below its tolerance. The pair
   is handed back as the sweep's is, :math:`\delta\psi` recomputed from
   :math:`\delta x` so the field row holds exactly. The Krylov dimension is
   ``FieldSolveMaxSweeps``, restarted up to four times; one that does not
   converge escalates to the exact solve. Irons–Tuck is a rank-one secant and
   removes one eigendirection of the sweep; GMRES removes them all, so this is
   the mode for a coupling whose sweep has several eigenvalues outside the unit
   circle. The adjoint of a ``krylov`` run is GMRES too, on the transposed
   system (:ref:`field-tier-1-adjoint`), escalating to the exact transposed
   solve.

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

**A field row may read a parameter** — a coil current, a boundary flux — and
the adjoint problem says so with ``dFieldResidual_dp(f, psi, dpsidt, states,
points, weights, t)``, taking ``FieldResidual``'s arguments and shaped as
``dScalarG_dp``: ``(np, 1)``, or ``(np, nPoints)`` with spatial parameters. The
gradient gains :math:`-z_{\psi,f}\, \partial R_f/\partial p`. The default is
an empty matrix, meaning the rows read no parameter: a ``FieldModel`` has no
parameters of its own, so the adjoint problem that knows which of its parameters
the model holds is the one place that can report the derivative. One that does
not report it gets zero for a parameter only the field rows read, beside a
correct :math:`G`.

.. _field-tier-0:

A field model as a black box
----------------------------

``FieldCoupling = "consistent"`` runs a model without differentiating through
it. What the transport needs from a field is one map, from the transport state
to the geometry at the physics nodes, and an equilibrium code driven as a black
box can supply that and nothing more: it has no residual in MaNTA's form and no
derivatives. Such a model derives from ``EquilibriumModel``
(``EquilibriumModel.hpp``) and supplies two things:

.. code-block:: cpp

   std::unique_ptr<Equilibrium> Solve(GlobalState const &states,
                                      std::vector<Position> const &points,
                                      Vector const &weights, Time t,
                                      Equilibrium const *warm);
   void Geometry(MatrixRef g, Equilibrium const &e,
                 std::vector<Position> const &points);

``Equilibrium`` is the model's own, opaque to the solver: a spline, a set of
Fourier coefficients, a handle on a file. ``Solve`` is handed the transport
state at the basis nodes, exactly as a field model's residual is, and ``warm``
is its own last answer in this run. ``Geometry`` then gives the slots at
whichever points the physics is about to be evaluated at, columns of ``g``.

The solver holds the answer as a **cache, not as unknowns**:

* Nothing joins the solution vector. The layout, the error weights and IDA's
  Newton norms are the uncoupled run's, and the restart file's ``nField`` is 0.
* The cache is **keyed on the transport state, byte for byte**, and on the
  time. A residual evaluated again at the same state -- KINSOL's merit
  function, a Jacobian build at the last residual's state -- costs no ``Solve``;
  any other state, however close, is a new one. The trace values are not part of
  the key, since a model is never handed them.
* The last equilibrium is the next ``Solve``'s warm start, and is **dropped at
  the start of every run**, with ``resetForRun`` called, so a solver reused for
  a second run starts from the same cold state as a fresh one.
* The **Jacobian carries no coupling**: the transport derivatives are taken at
  the geometry of the build's state, held fixed. Newton therefore converges
  linearly in the coupling rather than quadratically, which costs residual
  evaluations, each an equilibrium solve -- see the measurements below.
* ``initialize()`` hands the model a thousandth of the run's ``rtol`` and
  smallest ``atol`` through ``setTolerances``. An equilibrium converged no
  tighter than its own looser tolerance puts that error into every residual,
  where IDA reads it as local error; a model that cannot meet them returns
  ``false`` and is warned about, by name.
* The geometry slots are written to the model's output group at every slice,
  from the cached equilibrium at the slice's state.
* **No adjoint** is solved through it, and a run asking for one -- or for the
  ``"Objective"`` steady test, which needs one -- is refused. A gradient with
  the field frozen would be wrong by the field's share: a third of the
  closed-form gradient on the theta pinch.

**Every field model can run this way**, through ``FieldModelEquilibrium``,
which is what ``FieldCoupling = "consistent"`` attaches: its ``Solve`` is
Newton on the model's own rows with the transport state fixed, through the
model's ``updateFieldJacobian`` and ``solveB`` at :math:`\alpha = 0`, warm-started
from the last :math:`\psi` and iterated to round-off. A model with a
differential DOF is refused, since its field is not an instantaneous function
of the transport state. So the theta pinch runs either way, and the two agree:
on six cells at :math:`k = 3`, to :math:`8.8\times10^{-8}` in :math:`N` after a
transient at ``rtol`` :math:`10^{-6}`, and to :math:`4\times10^{-10}` at a steady
state.

.. _field-tier-1:

A black box coupled through its response
----------------------------------------

``FieldCoupling = "reduced"`` runs a model as an equilibrium, exactly as
``"consistent"`` does, and puts its *response* into the Jacobian. What tier 0's
Newton lacks is how the geometry moves when the transport state does,
:math:`G'(p)`. With it the coupled Jacobian is the transport one plus a
correction, with the transport unknowns alone:

.. math::

   J_\text{red} = A + A_g\, G'\, P,

where :math:`A` is the transport operator, :math:`P\,\delta x` the increment at
the basis nodes -- every component, the input ``Solve`` reads -- and
:math:`A_g` the case's derivatives in the geometry projected into the cell rows:
:math:`A_1` without its chain through :math:`\partial g/\partial\psi`, built
from the same evaluation (``geometryDerivatives``, ``Analytic`` or
``Differenced``, and refused for ``None``). An ``EquilibriumModel`` supplies
the response through

.. code-block:: cpp

   void applyResponse(MatrixRef dg, Equilibrium const &e, GlobalState const &states,
                      GlobalState const &dp, std::vector<Position> const &inputPoints,
                      Vector const &weights, Time t,
                      std::vector<Position> const &geometryPoints);

:math:`\delta g = G'(p)\,\delta p` at the equilibrium ``e``, at the physics
nodes. **The default is a forward difference of** ``Solve``, warm-started from
``e``: one equilibrium solve per call, at a step of ``responseStep()`` --
:math:`\sqrt{\epsilon}`, for an equilibrium converged to round-off -- relative to
the input. A model that converges only to a tolerance should return about its
square root, or override ``applyResponse`` with its own linearisation and
declare ``EquilibriumResponse::Model`` in its spec. Either way the equilibrium
work is free on the cost that matters; the case is asked for nothing it is not
asked for at tier 2.

``FieldModelEquilibrium`` supplies the exact response: the field rows hold
:math:`R(\psi(p); p) = 0`, so
:math:`\delta\psi = -(\partial R/\partial\psi)^{-1}(\partial R/\partial p)\,\delta p`
and :math:`\delta g = (\partial g/\partial\psi)\,\delta\psi` -- tier 2's blocks
with :math:`\psi` eliminated. So the theta pinch runs at tier 1 too, which is
what lets the reduced solve be checked against the coupled one.

``FieldSolve`` chooses how :math:`J_\text{red}` is solved, with the geometry
increment :math:`\delta g` in the role :math:`\delta\psi` plays at tier 2:

``iterative``
   The block Gauss–Seidel sweep, :math:`\delta x \leftarrow A^{-1}(r - A_g\,\delta g)`
   then :math:`\delta g \leftarrow G' P\,\delta x`, with Irons–Tuck on
   :math:`\delta g` and tier 2's stopping test. One transport solve and one
   response per sweep, and **one more transport solve at the end**:
   :math:`\delta x` is re-solved against the accepted :math:`\delta g`. At tier
   2 IDA is handed the accepted :math:`\delta\psi` itself; at tier 1 the
   transport increment is the whole Newton update, the field following it
   through the next residual's equilibrium solve, so it has to be solved
   against the best :math:`\delta g` the sweep has. Handing back the one
   before cost the theta pinch 199 steps against the exact solve's 163; with
   the re-solve it takes 163, in a third fewer sweeps. At its cap the sweep
   goes on to ``krylov``, since the exact solve here is a verification tool.

``krylov``
   GMRES on :math:`(I + G' P A^{-1} A_g)\,\delta g = G' P A^{-1} r`, as at
   tier 2. :math:`\delta g` is not in the solution vector and has no error
   weights from IDA, so it is given the run's rule for an algebraic variable,
   applied to the geometry slot by slot:
   :math:`1/(\mathrm{rtol}\,(|g| + \max|g_s|))`. A solve that does not
   converge is a failed linear solve -- IDA cuts its step -- never an iterate
   known to be unconverged.

``exact``
   :math:`(I + G' P A^{-1} A_g)\,\delta g = G' P A^{-1} r` assembled column by
   column, one transport solve and one response per geometry value at every
   physics node, and solved densely. It is what tier 1 is checked with.

A field with dynamics is refused: like the black box, tier 1 is a field that
is a function of the transport state.

.. _field-tier-1-adjoint:

The adjoint at tier 1
~~~~~~~~~~~~~~~~~~~~~

The adjoint solves :math:`J_\text{red}^T z = \partial G/\partial y`, where an
objective reading the geometry contributes through the response too:

.. math::

   \left(A^T + P^T G'^T A_g^T\right) z
   = G_y + P^T G'^T G_g,

:math:`G_g` being :math:`w_j\,\partial g/\partial(\text{geometry})` at the physics
nodes, as tier 2's :math:`G_\psi` is before its chain through
:math:`\partial g/\partial\psi`. It needs one thing a forward run does not, the
transposed response, which a model supplies through

.. code-block:: cpp

   void applyResponseTranspose(GlobalState &dp, Equilibrium const &e,
                               GlobalState const &states, Matrix const &dg,
                               std::vector<Position> const &inputPoints,
                               Vector const &weights, Time t,
                               std::vector<Position> const &geometryPoints);

:math:`\delta p = G'(p)^T \delta g`. **The default assembles** :math:`G'`: one
``applyResponse`` per component of the input -- :math:`3 n_\text{vars} +
n_\text{aux}` at each of the :math:`N(k+1)` input points -- and keeps the matrix
on the equilibrium, so it is built once per equilibrium whatever number of
objectives or iterations transpose it. Under the differenced response that is
one equilibrium solve per column, free on the measure the costs here are
counted in, and the only transpose a forward difference has.
``FieldModelEquilibrium`` supplies the exact transpose,
:math:`-(\partial R/\partial p)^T B^{-T} (\partial g/\partial\psi)^T \delta g`,
which costs no equilibrium solve at all.

The solve is tier 2's in the coupling variable :math:`\zeta`, the adjoint of the
geometry increment:

.. math::

   A^T z = G_y + P^T G'^T \zeta, \qquad \zeta = G_g - A_g^T z,

with :math:`A^T` the transposed transport operator ``assembleAdjointOperator``
builds and :math:`A_g^T` materialised beside :math:`A_1^T` and :math:`A_2^T`, so
that a test can zero it and see the gradient go wrong. ``iterative`` is the
transposed sweep with Irons–Tuck on :math:`\zeta`, stopping on the same relative
backward error ``FieldSolveAdjointTolerance`` bounds at tier 2; ``krylov`` is
GMRES on :math:`(I + A_g^T A^{-T} P^T G'^T)\,\zeta = G_g - A_g^T A^{-T} G_y`,
whose residual is that backward error; ``exact`` assembles the same system.
The sweep escalates to GMRES and GMRES to the exact solve, so an adjoint at
tier 1 is never taken from an unconverged solve. An adjoint on demand makes no
physics call, as at tier 2: the record of the last Jacobian build carries tier
1's blocks and its equilibrium.

With ``FieldModelEquilibrium``'s exact transpose the tier-1 adjoint is tier 2's
with :math:`\psi` eliminated, and the two agree: on the theta pinch the
transport block of :math:`z` to :math:`10^{-14}` under ``exact``, and steady
gradients to :math:`10^{-11}` under every solve, each within :math:`10^{-9}` of a
central difference of :math:`G`. The assembled transpose is a forward
difference and good to about :math:`10^{-7}` of :math:`G'`, which reaches the
gradient on the scale of its largest entry: :math:`1.1\times10^{-7}` of it
against tier 2.
``auto`` still couples a field model when the run solves an adjoint, since the
tier-2 adjoint needs no transposed response at all.

**Tier 1 costs the case what tier 2 does.** The response is field-side work
-- free on the measure that matters -- and ``A_g`` is the evaluation tier 2
already makes, so the steady solves spend exactly tier 2's calls and a
transient differs only by the steps each takes. ``ThetaPinchDiffusion`` on five
cells at :math:`k = 4`, ``rtol`` :math:`10^{-6}`, ``atol`` :math:`10^{-8}`, to
:math:`t = 1`, with the steady solves to ``SteadyStateTolerance``
:math:`10^{-8}` (the ``PseudoTransient`` counts are the same at
:math:`10^{-4}` and :math:`10^{-6}`); calls per physics point, values 1 and
derivatives 2, ``+g`` the analytic hooks priced as derivatives. The transient
and ``Newton`` columns reproduce the table under :ref:`field-coupling-mode` to
the call; its ``PseudoTransient`` column was taken in another setup and is not
comparable with this one:

.. list-table::
   :header-rows: 1

   * - 5 cells
     - transient
     - ``PseudoTransient``
     - ``Newton``
   * - T = 0.3, coupled, ``Analytic``
     - 6,025 (+1,200)
     - 625 (+400)
     - 325 (+200)
   * - T = 0.3, reduced, ``Analytic``
     - 5,825 (+1,150)
     - 625 (+400)
     - 325 (+200)
   * - T = 0.3, coupled, ``Differenced``
     - 9,025
     - 1,625
     - 825
   * - T = 0.3, reduced, ``Differenced``
     - 8,700
     - 1,625
     - 825
   * - T = 0.3, consistent
     - 8,925
     - 700
     - 400
   * - T = 0.9, coupled, ``Analytic``
     - 7,600 (+1,200)
     - 625 (+400)
     - 325 (+200)
   * - T = 0.9, reduced, ``Analytic``
     - 6,350 (+1,150)
     - 625 (+400)
     - 325 (+200)
   * - T = 0.9, coupled, ``Differenced``
     - 10,600
     - 1,625
     - 1,025
   * - T = 0.9, reduced, ``Differenced``
     - 9,225
     - 1,625
     - 825
   * - T = 0.9, consistent
     - 18,800
     - 700
     - 475

The model's own response and the differenced one give the same counts to the
call: the difference between them is equilibrium solves, which the case never
sees. The transients take 146 and 155 steps against tier 2's 143 and 147 and
tier 0's 170 and 169, so tier 1 costs 0.85–0.97 times tier 2, and the black
box 1.0–2.5 times tier 1: its Newton converges linearly, which costs residual
evaluations rather than steps, because a run with a field keeps the algebraic
unknowns out of IDA's error test by default (:ref:`suppress-algebraic-error`). With them in it, tier 0's linearly
converged algebraic unknowns read as local error and it took 341 and 937 steps.
Tier 1 agrees with tier 2 to the integration error: on six cells at
:math:`k = 3` to :math:`t = 0.5`, within :math:`2.7\times10^{-7}` in :math:`N`
and :math:`2.6\times10^{-8}` in :math:`B`, and steady states to
:math:`4\times10^{-13}` in the same Newton iterations. ``IDACalcIC`` keeps its
line search here, unlike tier 0: the Jacobian carries the coupling, and the
runs are step for step the same with it on or off, at :math:`T = 0.9` too.

.. _field-coupling-mode:

Choosing the coupling
---------------------

``FieldCoupling = "auto"``, the default, picks the mode from what the case and
the model supply and what the run needs:

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - the case's ``geometryDerivatives``
     - the run
     - mode
   * - ``None``
     - anything not needing the coupled field
     - ``consistent``
   * - ``Analytic``
     - anything
     - ``coupled``
   * - ``Differenced``
     - a transient, or ``TimeMarch``
     - ``coupled``
   * - ``Differenced``
     - a ``PseudoTransient`` or ``Newton`` steady solve
     - ``consistent``

**A field with dynamics, or an adjoint, needs the coupled field**: the black box
can neither carry a differential :math:`\psi` nor differentiate through the
field. So ``auto`` couples whenever the model has a differential DOF, or the run
solves an adjoint (``solveAdjoint``, or ``SteadyStateTest = "Objective"``),
whatever the derivatives cost — and refuses, naming why, when the case declares
none. ``"coupled"`` and ``"consistent"`` take the mode as given; the first is
refused for a case declaring no derivatives, the second for a field with
dynamics or a run with an adjoint. ``"reduced"`` (:ref:`field-tier-1`) carries
an adjoint and is refused only for a field with dynamics, but it is never
``auto``'s choice for a field model: with the model's own unknowns to hand,
``"coupled"`` is the same Jacobian without the equilibrium solves, at the same
cost to the case, and its adjoint transposes the model's own blocks rather than
a response.

**The rest of the table is a measurement.** ``ThetaPinchDiffusion`` on five
cells at :math:`k = 4`, ``rtol`` :math:`10^{-6}`, ``atol`` :math:`10^{-8}`, to
:math:`t = 1`, counting what matters for an expensive flux: calls into the
transport system per physics point, one for values and two for derivatives.
``Analytic``'s calls to the geometry hooks are shown separately as ``+g``,
priced like a derivative evaluation, since that is what an autodiff case pays;
``Differenced``'s :math:`n_{geom} + 1 = 5` value evaluations per Jacobian build
are in its total.

.. list-table::
   :header-rows: 1

   * - 5 cells
     - transient
     - ``PseudoTransient``
     - ``Newton``
   * - T = 0.3, coupled, ``Analytic``
     - 6,025 (+1,200)
     - 550 (+350)
     - 325 (+200)
   * - T = 0.3, coupled, ``Differenced``
     - 9,025
     - 1,425
     - 825
   * - T = 0.3, consistent
     - 8,925
     - 550
     - 400
   * - T = 0.9, coupled, ``Analytic``
     - 7,600 (+1,200)
     - 550 (+350)
     - 325 (+200)
   * - T = 0.9, coupled, ``Differenced``
     - 10,600
     - 1,425
     - 1,025
   * - T = 0.9, consistent
     - 18,800
     - 625
     - 475

**A transient wants the coupled Jacobian whatever the derivatives cost.** IDA
reuses each Jacobian over many steps, so its cost is almost all residual
values, and the black box's linearly converging Newton costs it as much as the
differenced coupling at low pressure and 1.8 times the calls at high, and 1.2–2.1
times the analytic coupling with its hooks priced in. Coupling a transient is
never the more expensive choice. **A steady solve is decided by
what the derivatives cost**, because it builds a Jacobian at nearly every
iteration. Differenced, they make the coupled solve 2.1–2.6 times the black
box's under either steady solver, so ``auto`` runs those as a black box.
Analytic, it depends on what the hooks cost, which nothing can tell the
solver: free — analytic prefactors — coupling ties ``PseudoTransient`` and wins
``Newton``; priced like a derivative evaluation, as autodiff hooks are, it loses
both, by up to 1.6 and 1.3 times. ``auto`` couples an ``Analytic`` case
regardless, for Newton's quadratic convergence and a loss that is bounded; a
case whose hooks are expensive can set ``FieldCoupling = "consistent"`` for its
steady solves. The field-side work, the equilibrium solves included, is free by
this measure.

.. _native-equilibrium:

A native equilibrium model
--------------------------

A field need not be a ``FieldModel`` to reach a config file. An
``EquilibriumModel`` has a registry of its own, the field models' pattern
exactly: ``REGISTER_EQUILIBRIUM_MODEL_HEADER(T)`` in the header,
``REGISTER_EQUILIBRIUM_MODEL_IMPL(T)`` in the ``.cpp``, a ``(toml::value const
&, Grid const &)`` constructor that reads the model's own table, a duplicate or
unknown name thrown with what *is* registered, and ``PhysicsPlugins`` for a
model built outside the tree. A run selects one in place of a field model:

.. code-block:: toml

   [configuration]
   TransportSystem = "ThetaPinchDiffusion"
   EquilibriumModel = "ThetaPinchPressureBalance"

The two keys are exclusive, refused together at load: a run has one source of
geometry. Both are accepted by ``Runner.configure`` as well, with the model's
table nested under its name; see :ref:`field-from-python`.

A native model has no unknowns, so ``FieldCoupling`` chooses between
``consistent`` (:ref:`field-tier-0`) and ``reduced`` (:ref:`field-tier-1`), and
``coupled`` is refused by name. ``reduced`` carries an adjoint, through the
model's transposed response; ``consistent`` does not. A model that supplies only
``Solve`` and ``Geometry`` runs at either tier: its response, and the transpose
the adjoint needs, are then the base class's forward difference of ``Solve`` and
the matrix assembled from it.

The native theta pinch
~~~~~~~~~~~~~~~~~~~~~~

``ThetaPinchPressureBalance`` is the theta pinch's equilibrium as a native
model, and the same physics as :ref:`ThetaPinchEquilibrium <field-coupling>`:
:math:`p + B^2 = C` on every surface (:math:`2\mu_0 = 1`) and the wall's
:math:`\int_0^1 V'\,dx = \pi a^2`. With :math:`B = \Phi_a / V'` the balance is
:math:`V' = \Phi_a / \sqrt{C - p}` pointwise, so the equilibrium of a pressure
profile is a root-find for the one constant :math:`C`, and the four geometry
slots follow in closed form,

.. math::

   V' = \frac{\Phi_a}{\sqrt{C - p}}, \qquad
   \frac{dV'}{dx} = \frac{\Phi_a}{2}\,\frac{p_x}{(C - p)^{3/2}}, \qquad
   \langle|\nabla x|^2\rangle = \frac{4\pi A}{V'^2}, \qquad
   B = \frac{\Phi_a}{V'},

with :math:`A(x) = \int_0^x V'`. It reads its pressure from the case's auxiliary
variable ``PressureAux``, as the field model does, and takes the same slots in
the same order, so ``ThetaPinchDiffusion`` reads either. Its table is
``[ThetaPinchPressureBalance]``: ``PressureAux``, ``WallRadius``,
``VacuumField`` and ``DifferencedResponse``.

**What differs is the discretisation.** The field model holds :math:`V'` as a
C² cubic spline on the transport faces and imposes the balance by Galerkin
projection, so its :math:`V'` is smooth across faces. The native model evaluates
the balance on the transport's own pressure polynomial :math:`p_h`, each cell's
interpolant of its nodal values, so :math:`V'` is discontinuous where
:math:`p_h` is; the integrals (the wall constraint and :math:`A`) are a
16-point Gauss rule per cell on a smooth integrand.

**The pressure gradient is recovered, not read off** :math:`p_h`, and that is
the one subtlety. :math:`dV'/dx` goes into the flux, and
:math:`p_h`'s derivative is good only to :math:`O(h^k)`: read off it, the
manufactured theta pinch (``ThetaPinchMMSTests.cpp``) converges at :math:`k` in
:math:`N` — 0.99, 2.24, 3.11 for :math:`k = 1, 2, 3` — the cost the field model's
header records for a :math:`V'` that breaks where :math:`p_h` does. A
least-squares fit to :math:`p_h`'s nodal values over the cell and its
neighbours is not enough either: the interpolation error repeats its shape in
every cell, the fit differentiates that shape, and at :math:`k = 1` the order
decays back to 1 by 256 cells. So :math:`p_x` comes from a degree-:math:`k+1`
polynomial fitted to the *cell means* of :math:`p_h` over a centred patch of
:math:`2\lceil (k+1)/2 \rceil + 1` cells, moved inward at the ends. A cell mean
carries the interpolation error only as a smooth :math:`O(h^{k+1})` function, so
:math:`dV'/dx` converges at :math:`k + 1` or better: on an exact pressure, 2.00,
2.99 and 5.00. :math:`V'` itself stays the exact balance on :math:`p_h`.

The manufactured study then gives :math:`k + 1` throughout, transient on 16,
32 and 64 cells and steady on 32, 64 and 128:

.. list-table::
   :header-rows: 1

   * -
     - :math:`N`, transient
     - :math:`V'`, transient
     - :math:`N`, steady
     - :math:`V'`, steady
   * - :math:`k = 1`
     - 2.00, 2.00
     - 2.00, 2.00
     - 1.99, 2.00
     - 2.00, 2.00
   * - :math:`k = 2`
     - 3.08, 3.02
     - 3.09, 3.02
     - 3.05, 3.01
     - 3.05, 3.01
   * - :math:`k = 3`
     - 3.89, 3.97
     - 3.80, 3.94
     - 3.97, 3.99
     - 3.95, 3.99

**It lands within the field model's discretisation error of the field model's
answer.** The distance between the two runs' :math:`N` converges at
:math:`k + 1` with them, at 0.37 of the field model's own error at :math:`k =
1`, 0.10–0.34 at :math:`k = 2`, and 1.9–2.1 at :math:`k = 3` (transient), where
the native error is the larger, its fit spanning five cells. :math:`V'` is
where they part: the spline superconverges at :math:`k \le 2` (2.5 and 3.5) and
the native :math:`V'` goes at :math:`k + 1`. On the regression case,
``Tests/RegressionTests/ThetaPinchNative.conf``, the native :math:`N` is
:math:`5\times10^{-4}` from ``ThetaPinch.conf``'s at 8 cells and
:math:`2.6\times10^{-4}` at 16 (RMS, relative), against the field model's own
change on refinement to 32 cells of :math:`8\times10^{-3}` and
:math:`2.6\times10^{-3}`.

Its response is exact (``EquilibriumResponse::Model``): the same evaluation run
in forward-mode duals, with :math:`\delta C = \int (C-p)^{-3/2}\,\delta p /
\int (C-p)^{-3/2}` from the wall. The transpose is exact too, the pressure
columns of :math:`G'` by one dual sweep each, assembled once per equilibrium and
transposed, with no ``Solve``: an adjoint on demand makes no physics call and
one equilibrium solve, for the objective's geometry. ``DifferencedResponse =
true`` falls back to the base class's difference and its assembly, as
``FieldModelEquilibrium``'s flag does; that adjoint makes one solve per input
component per point, once per equilibrium.

.. _native-coupling-mode:

Choosing its coupling
~~~~~~~~~~~~~~~~~~~~~

``FieldCoupling = "auto"`` takes the field model's rule with ``reduced`` for
``coupled``, and one more input, the model's response:

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - the case's ``geometryDerivatives``
     - the model's response
     - the run
     - mode
   * - ``None``
     - either
     - no adjoint
     - ``consistent``
   * - ``Analytic`` or ``Differenced``
     - either
     - a transient, or ``TimeMarch``
     - ``reduced``
   * - ``Analytic``
     - ``Model``
     - a ``PseudoTransient`` or ``Newton`` steady solve
     - ``reduced``
   * - ``Analytic``
     - ``Differenced``
     - a steady solve
     - ``consistent``
   * - ``Differenced``
     - either
     - a steady solve
     - ``consistent``
   * - ``Analytic`` or ``Differenced``
     - either
     - an adjoint
     - ``reduced``

An adjoint is ``reduced`` whatever it costs, since that is the only tier with
one, and is refused for a case declaring no derivatives.

**The table is a measurement**, in the setup of the field-model table above
(five cells, :math:`k = 4`, to :math:`t = 1`, steady solves to
:math:`10^{-8}`), on ``ThetaPinchPressureBalance``, with the equilibrium solves
beside the transport cost (for ``reduced``, with the model's own response and
with the differenced one):

.. list-table::
   :header-rows: 1

   * -
     - transient
     - ``PseudoTransient``
     - ``Newton``
   * - ``Analytic``, reduced
     - 6,700–6,900 (+475–550)
     - 625 (+200)
     - 325 (+100)
   * - ``Differenced``, reduced
     - 9,275–9,450
     - 1,625
     - 825
   * - consistent
     - 14,075–25,300
     - 700
     - 400
   * - equilibrium solves, reduced, ``Model``
     - 221–235
     - 9
     - 5
   * - equilibrium solves, reduced, ``Differenced``
     - 1,006–1,215
     - 44–53
     - 19–28
   * - equilibrium solves, consistent
     - 528–947
     - 10
     - 6

The ranges are T = 0.3 to 0.9. The transport costs follow the field model's --
its steady solves to the call, its transients within a quarter -- so the first
rules are the field model's: ``reduced`` for transients, 1.5–3.8 times cheaper
than the black box, and ``consistent`` for a steady solve of a ``Differenced``
case, which costs ``reduced`` 2.1–2.3 times as much under either solver. **The response moves no
transport cost — it moves only equilibrium solves.** The model's own costs
fewer than the black box everywhere. A differenced one costs 3.2–5.3 times the
black box's on a steady solve, where ``reduced`` saves at most a fifth of the
transport cost, so ``auto`` runs those as a black box. On a transient it costs
1.3–1.9 times the black box's solves to save 2.0–3.8 times the transport calls,
and stays ``reduced``: which side of that trade wins depends on what one
equilibrium solve costs against one sweep of the physics, which ``auto`` cannot
know, and a case that knows can say ``FieldCoupling = "consistent"``.

.. _field-from-python:

From Python
-----------

Every tier is reachable from a ``manta.Runner``, and through the same function
the config file goes through, so ``FieldCoupling`` resolves identically on both
surfaces: a dict and a ``.conf`` describing the same run give netCDF output
equal bit for bit, at every tier (``python/Tests/test_field_coupling.py``).

**A registered model, by name.** ``FieldModel``, ``EquilibriumModel`` and
``FieldCoupling`` go in the dict beside the solver's keys, and the model's own
table is a nested dict under the name it reads, as a C++ case's is:

.. code-block:: python

   runner = manta.Runner("ThetaPinchDiffusion")
   runner.configure({
       "PolynomialDegree": 3, "GridSize": 6,
       "LowerBoundary": 0.0, "UpperBoundary": 1.0,
       "t_final": 1.0, "delta_t": 0.1,
       "FieldModel": "ThetaPinchEquilibrium",
       "FieldCoupling": "reduced",
       "ThetaPinchEquilibrium": {"PressureAux": 0},
       "ThetaPinchDiffusion": {"Temperature": 0.3},
   })

A restart file written by a coupled run carries the field's unknowns, and a
``Runner`` reads them back as the binary does, after checking their count
against the configured model's.

**An equilibrium model written in Python** is a ``manta.EquilibriumModel``: tiers
0 and 1, the black box and the black box coupled through its response. It
declares its geometry slots as data and implements ``Solve`` and ``Geometry``:

.. code-block:: python

   class Balance(manta.EquilibriumModel):
       geometry = [manta.GeometrySlot("Vprime"), manta.GeometrySlot("dVprime_dx"),
                   manta.GeometrySlot("gradx2"), manta.GeometrySlot("B")]
       label = "Phi/Phi_wall"   # the coordinate the slots are a function of
       name = "ThetaPinch"      # the netCDF group they are written into

       def Solve(self, states, points, weights, t, warm):
           p = states["Aux"][:, 0]          # (nPoints, nAux), at the basis nodes
           ...
           return equilibrium               # any object; the solver hands it back

       def Geometry(self, equilibrium, points):
           ...
           return g                         # (len(geometry), len(points))

   runner.set_equilibrium_model(Balance())

The contract is the C++ one (:ref:`field-tier-0`): ``Solve`` is called whenever
the transport state changes and only then, ``warm`` is the model's previous
answer in the run or ``None``, and an exception is a recoverable step failure.
The equilibrium object is opaque and must be a self-contained snapshot, since
the solver may ask for ``Geometry`` of one after a later ``Solve``. At tier 1
the response is the base class's forward difference of ``Solve`` unless the
model declares ``response = manta.EquilibriumResponse.Model`` and implements
``applyResponse(equilibrium, states, dp, inputPoints, weights, t,
geometryPoints)``, returning ``dg``, and ``applyResponseTranspose``, returning
a dict of ``dp`` rows; ``responseStep``, ``setTolerances`` and ``resetForRun``
are optional as in C++. ``manta.registerEquilibriumModel(name, factory)`` names
one, so ``EquilibriumModel = "name"`` reaches it from a dict or, through the
``manta`` command, from a config file; ``factory(config, grid)`` builds it.
``python/Tests/theta_pinch_balance.py`` is ``ThetaPinchPressureBalance``
written that way, and runs the C++ model's numbers to 7e-11.

**A transport case reading the geometry** declares it as class attributes,
``readsGeometry = True`` and ``geometryDerivatives`` (``manta.GeometryDerivatives``
``Analytic``, ``Differenced`` or ``None_``), and reads ``state.geom`` by the
model's slot names — ``state.geom["B"]`` — or by position. ``Analytic`` means
its ``dSigmaFn_dGeometry``, ``dSources_dGeometry`` and ``dAuxG_dGeometry``,
which return one entry per slot. A ``manta.jax`` case reads ``state.Geometry``
and has all three derived by ``jax.grad``, as its other derivatives are; and an
objective reading ``state.Geometry`` sends its ``dg/d(geometry)`` rows to the
solver, which contracts them into the field's share of the gradient.

Current restrictions
--------------------

* **``nScalars > 0`` with a field model is refused**, at ``setFieldModel``. The
  reason is a disagreement between two branches rather than a missing feature:
  the non-superconvergent ``dSources_dScalars`` coupling builds its ``State``
  from ``DGSoln::evalOnNode``, which has no geometry rows, while its
  superconvergent twin reads states that already carry them — so a case reading
  geometry there would work with ``Superconvergent = true`` and read out of
  bounds with it off.
* **A tier-2 field model cannot be written in Python.** ``FieldModel`` has no
  pybind11 class, so a Python run couples a field's own unknowns only through a
  C++ model named in the dict. A Python ``EquilibriumModel`` covers tiers 0
  and 1.
* **Degree and mesh adaptation are refused beside a field**, on both surfaces:
  the adaptive drivers build a solver per level and cannot carry the model.

Two dimensions, and beyond
--------------------------

What is implemented is the *coupling*, not an equilibrium solver: MaNTA supplies
the block structure, the coupled Jacobian and adjoint solves, the DOF
bookkeeping and the serialisation, and a model supplies the physics. A 2-D
Grad–Shafranov solve, or a DESC stellarator equilibrium, plugs in by overriding
the four ``B`` operations above so that MaNTA never forms the model's Jacobian
at all. Neither exists yet.
