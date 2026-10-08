The discretisation
==================

This page describes what the solver actually does: the system it forms, how the
unknowns are laid out, and how the resulting nonlinear system is solved. The one
part of it you cannot skip if you are writing a physics case is
:ref:`the sign convention <sign-convention>`.

The system
----------

A transport system defines, for each variable :math:`i`:

.. math::

   a_i \, \partial_t u_i + \partial_x \sigma_i &= S_i(u, q, \sigma, \phi, x, t) \\
   \sigma_i &= \hat\sigma_i(u, q, x, t) \\
   q_i &= \partial_x u_i

with :math:`q` introduced as an independent unknown so that the flux may depend
on the gradient. Two further families of constraint may be present:

.. math::

   G_j(\phi, u, q, \sigma, x) &= 0, \qquad j = 1 \ldots \texttt{nAux} \\
   G_s(\mu, y, \dot y, t) &= 0, \qquad s = 1 \ldots \texttt{nScalars}

The auxiliary constraints :math:`G_j` are algebraic and pointwise: :math:`\phi_j`
is whatever value satisfies :math:`G_j = 0` at that point, given the other
fields. The scalar constraints :math:`G_s` are global — each is a single equation
in the whole solution, so it may contain integrals over the domain — and each may
be algebraic or differential (carrying :math:`\dot\mu`), which is declared per
scalar through ``isScalarDifferential``.

A fourth family is present only when a :doc:`field model <field_coupling>` is
attached:

.. math::

   R_m(\psi, \dot\psi, u, q, \sigma, \phi, t) = 0,
   \qquad m = 1 \ldots \texttt{nFieldDOF}

with the ``nFieldDOF`` unknowns :math:`\psi` reaching the transport equations
only through the *geometry slots* :math:`g_s(\psi, x, t)` the model derives from
them, which a physics case reads as ``State::geom(s)``. Like the scalars, each
:math:`\psi_m` may be algebraic or differential. Everything below holds with
``nFieldDOF = 0``, which is every run that does not set ``FieldModel``.

.. _sign-convention:

The flux sign convention
------------------------

.. warning::

   **The stored** ``sigma`` **is** :math:`-\hat\sigma`, **not** :math:`\hat\sigma`.

The second line above is a *sign convention*, not an identity. ``residual``
forms the flux row as

.. code-block:: text

   res.sigma = A sigma_h + (I_h sigma_hat, phi)

with ``A`` the mass matrix, so what it enforces is
:math:`\sigma_h = -\Pi(\hat\sigma)`. ``setInitialConditions`` does the same
thing explicitly, with a "remember minus sign" comment. The equation actually
integrated is therefore

.. math::

   a_i \, \partial_t u_i - \partial_x \left[ \hat\sigma_i(u, q, x, t) \right] = S_i

There are two consequences, and both bite.

**A manufactured source must be differentiated with that minus sign.** With
``SigmaFn`` returning :math:`\kappa q`, the source for
:math:`u = \sin(\pi x)(1 + t)` is
:math:`S = \sin(\pi x)\,(1 + \kappa \pi^2 (1 + t))`, which is
:math:`u_t - \kappa u_{xx}` — a diffusion equation. Getting the sign backwards
gives you anti-diffusion, and the case still converges, at the correct rate, to
the wrong function. An order-of-accuracy study cannot detect this; only a
comparison against a closed form can.

**The** ``State::Flux`` **array that physics hooks read carries the negated**
:math:`\sigma_h`, not :math:`\hat\sigma`. A source term that reads the flux back
out of the state is reading :math:`-\hat\sigma`.

Degrees of freedom
------------------

Space is divided into cells. On each cell every field is a polynomial of degree
``PolynomialDegree`` = :math:`k`, expanded in a nodal (Chebyshev-node) basis of
:math:`k+1` functions. The HDG method adds a *trace* unknown :math:`\lambda`
living on the cell faces, one value per face per variable, which is what couples
the cells to one another.

The layout of the global solution vector is

.. code-block:: text

   [ sigma | q | u | aux ]   per cell, for each cell in turn
   [ lambda ]                all face traces
   [ mu ]                    all global scalars
   [ psi ]                   the field model's unknowns, if there is one

This ordering is shared by the solution vector (``DGSoln::Map``) and by the
per-cell Jacobian block ``MX``. Getting a column index wrong in that layout is
the most common way to break the solver silently, because — see below — a wrong
Jacobian does not produce a wrong answer.

The field block is **last** so that attaching a model moves nothing before it,
which is what makes the whole coupling inert when it is absent. ``DGSoln::getDoF()``
is the one authority on the total length: the formula was open-coded in three
places, and a copy that did not know about the field block wrote a *short*
restart file whose recorded ``nDOF`` matched the uncoupled formula — so the
truncated file read back as consistent.

``DGSoln`` and ``DGApprox`` are **views**, ``Eigen::Map`` objects over memory
SUNDIALS owns, not containers. That matters if you hold one across a solve; see
:doc:`running`.

Solving it
----------

The spatial discretisation leaves an index-1 DAE in the vector
:math:`y = (\sigma, q, u, \phi, \lambda, \mu)`, which is handed to SUNDIALS IDA.
Newton's method inside IDA needs the Jacobian
:math:`\partial F/\partial y + \alpha \, \partial F/\partial \dot y`, and MaNTA
supplies it in an unusual way:

* IDA is given a **custom** ``SUNLinearSolver`` (``SunLinSolWrapper``) together
  with a **deliberately empty** ``SUNMatrix`` (``SunMatrixWrapper``), whose only
  purpose is to convince IDA that it has a matrix-based direct solver.
* **The Jacobian is never assembled.** Instead ``updateMatricesForJacSolve``
  builds and factorises the small per-cell blocks, ``solveHDGJac`` statically
  condenses the cell-local unknowns onto :math:`\lambda` and back-substitutes,
  and ``solveJacEq`` wraps that in a Woodbury/bordered elimination to account for
  the global scalars :math:`\mu`.
* With a field model attached, ``solveJacEq`` wraps *that* in a second block
  elimination onto :math:`\psi` — exactly, or by an accelerated block
  Gauss–Seidel sweep that escalates to the exact form rather than returning an
  under-converged answer. See :ref:`field-solve`.

Static condensation is what makes HDG attractive here: the only globally coupled
system is the one for :math:`\lambda`, whose size is (number of faces) ×
(number of variables), independent of :math:`k`.

.. important::

   Because the Jacobian is never formed, **an error in it does not produce a
   wrong answer — only slow Newton convergence.** Several defects in this area
   survived a passing regression suite for months. The tests that can catch such
   an error are the ones that finite-difference the residual and require
   :math:`J \, \delta y = g`, and the ones that measure observed order of
   accuracy. See :doc:`testing`.

.. _tau-scaling:

The stabilisation parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The numerical flux on a face is :math:`\hat\sigma = \sigma_h + \tau (u_h - \lambda)`,
and local conservation makes :math:`\hat\sigma` the accurate flux whatever
:math:`\tau` is. What :math:`\tau` decides is how a mismatch between :math:`u_h`
and the trace is shared out. At a Dirichlet end, where :math:`\lambda` is the
datum :math:`u_b`,

.. math::

   \sigma_h - \sigma = -\tau \, (u_h - u_b) + (\hat\sigma - \sigma),

and the last term is small. So where a boundary layer is unresolved, a
:math:`\tau` much larger than :math:`\kappa/h` pins :math:`u_h` to the wall value
and pushes :math:`\tau` times the leftover jump into :math:`\sigma_h`, where it
shows as an oscillation across the last cell; a :math:`\tau` much smaller leaves
:math:`\sigma_h` accurate and the jump in :math:`u_h`. Neither makes
:math:`u_h` right, which only resolution does. And too small a :math:`\tau`
where :math:`\kappa/h` is large degrades the interior instead.

``tauScaling`` chooses between two forms.

``"Constant"`` (the default)
   :math:`\tau` = ``tau`` on every face, in the units of :math:`\kappa/h`.

``"Diffusive"``
   One-sided, per variable, on each face :math:`f` of each cell :math:`I`,

   .. math::

      \tau_{I,f} = \texttt{tau} \left( \frac{\kappa(f)}{h_I}
          + \texttt{tauFloor} \max_{f'} \frac{\kappa(f')}{h_{I'}} \right),
      \qquad \kappa = \left| \frac{\partial \hat\sigma}{\partial q} \right|,

   ``tau`` is then a dimensionless multiplier and the floor a fraction of the
   largest :math:`\kappa/h` on the grid, so neither depends on units. The floor is
   not optional: where :math:`\kappa` vanishes on a face — a degenerate axis, say
   — and nothing else in its trace row involves :math:`\lambda`, :math:`\tau` is
   what determines :math:`\lambda` there.

   ``TauKappa`` says where :math:`\kappa(f)` comes from.

   ``"Nodal"`` (the default)
      The cell's own :math:`\partial\hat\sigma/\partial q` at the nodes the
      residual and the Jacobian already evaluate the physics at, extrapolated to
      its two faces through the cell's interpolant of those values. Where that
      extrapolation comes out non-positive and every nodal value is positive, the
      interpolant of :math:`\log\kappa` is extrapolated instead, which cannot
      reach zero; where some nodal value is not positive — a Newton iterate, say —
      the face is left to the floor. The case is evaluated nowhere it is not
      evaluated already, so the evaluation plan gains no batch shape, and
      wherever a Jacobian build has just taken the derivatives, :math:`\kappa`
      costs nothing at all.

      Read from the cell's interior, it cannot see the trace. At a Dirichlet wall
      whose layer is narrower than the cell, where :math:`\kappa` at the face is
      set by the datum, the nodes see the cell instead. That is the one place the
      two sources differ by more than a few per cent; see below.

   ``"Face"``
      Evaluated at the face, from the trace (the datum at a Dirichlet end) and the
      cell's own one-sided :math:`q`: one more batched
      ``ComputePhysicsDerivatives`` on the :math:`2N` face points each time
      :math:`\tau` is evaluated. A flux that is singular at the Dirichlet datum —
      Shestakov's :math:`D_0 q^3/u^2` at :math:`u_b = 0` — is evaluated there and
      fails, where ``"Nodal"`` never reads it.

   Both assume a layer is not much narrower than a cell, so that one value of
   :math:`\kappa` per face is representative of it.

   ``tauUpdate`` says when :math:`\tau` is re-evaluated. All three converge to the
   same discrete solution; they differ in cost and robustness on the way.

   ``"Residual"`` (the default)
      On every residual, at that residual's state: one batched
      ``ComputePhysicsDerivatives`` each time, on the physics nodes or the face
      points. The Jacobian carries :math:`\partial\tau/\partial y` -- which
      multiplies the jumps :math:`u_h - \lambda` -- by a finite difference over
      each component of the state :math:`\kappa` is read from, one more call per
      component per Jacobian build. Under ``"Nodal"`` that is the nodal state, and
      the derivative is chained onto the cell's coefficients — through
      :math:`u^*`'s dependence on :math:`q` as well as :math:`u` with
      ``Superconvergent``. Left out are its dependence on global scalars and field
      unknowns, and the floor's through the grid maximum.

   ``"ContinuationStep"``
      Once per pseudo-transient continuation step, and frozen through that step's
      Newton solve, whose Jacobian is then exact with no extra terms. Under
      ``"Face"`` it is evaluated at the state the step starts at; under
      ``"Nodal"`` it is read off the derivatives the most recent Jacobian build
      took — from an iterate at most a few Newton iterations old, and at no cost.
      Convergence is judged with :math:`\tau` refreshed in the same way. Steady
      solves only: a time march would freeze :math:`\tau` at the initial
      condition, and is refused.

      This is a fixed-point iteration on :math:`\tau`, and pseudo-time does not
      damp it: :math:`\tau` depends on :math:`q`, and under ``"Face"`` on
      :math:`\lambda`, which are algebraic, so even a vanishing step lets the Newton solve move them onto the
      frozen-:math:`\tau` equations. From a state where :math:`\tau` is sensitive
      to them, the re-evaluated residual can exceed the one the step started from
      on every step, and the continuation stalls. That was seen once, with
      ``NewtonJacobianReuse`` at its default; with a fresh Jacobian every Newton
      iteration the same run never reached such a state.

   ``"JacobianBuild"``
      As ``"ContinuationStep"``, and also at every Jacobian build, so :math:`\tau`
      is fixed across the Newton iterations that share a Jacobian. The residual
      evaluated just before a rebuild used the previous :math:`\tau`, so that step
      is lagged, and it shows: it takes more Newton iterations than either of the
      other two. Steady solves only.

   Measured on a steady wall-layer problem (5 and 20 cells, uniform and graded,
   :math:`k = 4`, flux exponents 1, 1.75 and 2.5), as flux plus derivative point
   evaluations relative to a constant :math:`\tau`:

   .. list-table::
      :header-rows: 1

      * - ``TauKappa``
        - ``"Residual"``
        - ``"ContinuationStep"``
        - ``"JacobianBuild"``
      * - ``"Nodal"``
        - 2.56-2.77x
        - 1.01-1.13x
        - 1.04-1.31x
      * - ``"Face"``
        - 1.61-1.73x
        - 1.06-1.15x
        - 1.27-1.49x

   The two sources give the same answers to within 2% in :math:`u`, and the wall
   flux to within the same order: the nodal one is 40% worse on the coarsest
   uniform mesh at the steepest wall and up to 36% better on the finer ones.
   ``"Nodal"`` is the cheaper whenever :math:`\tau` is held, because it reads
   :math:`\kappa` off derivatives a Jacobian build has already taken. Under
   ``"Residual"`` it is the dearer by its point count — :math:`N(k+1)` per
   residual against :math:`2N` — though for a case whose cost is per *call*
   rather than per point, a JAX one say, the two make the same number of calls
   and ``"Nodal"`` adds no batch shape to compile. Park's and Jardin's benchmarks
   order the options the same way.

   On Shestakov's degenerate flux ``"Nodal"`` is also the more robust. Under
   ``"Residual"``, ``"Face"`` converged to a visibly wrong steady state in three
   of twelve runs — L1 errors of 0.97, 0.14 and 0.036 against a constant
   :math:`\tau`'s 0.016, 0.011 and 0.0055 — and ``"Nodal"`` in none; under
   ``"JacobianBuild"`` the face source failed in five of twelve, the nodal one
   in one. With a zero Dirichlet value the face source cannot run at all.

   Newton iterations are within 1% of a constant :math:`\tau`'s for
   ``"Residual"`` and ``"ContinuationStep"``; the overhead of ``"Residual"`` is
   almost all the extra call in every residual. ``NewtonJacobianReuse = 1``, the
   default, is cheaper outright on the wall layer than reusing a Jacobian ten
   times -- about 25% fewer point evaluations for every option -- and is the only
   setting at which every run converged, constant :math:`\tau` included.

   The choice of :math:`\tau` does not change which cells the estimator
   :math:`\|u^* - u_h\|_K` ranks worst, but it does change how well that
   estimate is calibrated. On the same problem its ratio to the true cell error
   was 0.27-0.54 at a constant :math:`\tau = 1` and 0.95-1.07 under
   ``"Diffusive"``, on uniform and graded grids alike.

   The adjoint does not carry :math:`\tau`'s dependence on the state and would
   give a silently wrong gradient, so ``"Diffusive"`` with ``solveAdjoint`` is
   refused.

Interpolatory HDG
-----------------

``residual`` evaluates ``SigmaFn``, ``Sources`` and ``AuxG`` *at the nodes* of
the basis and then interpolates the result, rather than integrating them by
quadrature. In the notation of the literature it forms :math:`I_h F(u_h)` with
:math:`I_h` mapping into :math:`W_h = P_k`. This makes MaNTA an *interpolatory*
HDG method — the scheme of `arXiv:1811.09667
<https://arxiv.org/abs/1811.09667>`_, which ``Matrices.cpp`` cites for the
Jacobian form.

The practical consequence is that a physics hook is only ever asked for values
at specific points; it never needs to know anything about quadrature. The
theoretical consequence concerns the postprocessed solution and is the subject of
:doc:`superconvergence`.

The residual and the boundaries
-------------------------------

``residual`` does **not** write the Dirichlet boundary rows. Those constraints
are imposed inside the linear solve instead. The visible effect is that a
finite-differenced Jacobian of ``residual`` is rank-deficient by exactly the
number of Dirichlet boundaries, which is expected rather than a bug, and which
the Jacobian tests account for explicitly.

A second effect is less obvious. The trace unknown :math:`\lambda` at a
Dirichlet end appears in no equation at all: its row and its column in the
condensed trace matrix are identically zero, and the Newton correction there is
pinned to zero. So the solver never computes a value for it, and it holds
whatever was last written into it. The solver writes :math:`g_D(t)` there
directly -- once when the initial condition is built, and again each time the
state is reported -- which is what makes the boundary entry of the trace in the
output and the restart file agree with the boundary condition at that time.
Nothing downstream reads it: the interior sees the Dirichlet datum through the
cell rows, which are assembled from the boundary functions at the residual's own
time.
