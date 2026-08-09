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
``Polynomial_degree`` = :math:`k`, expanded in a nodal (Chebyshev-node) basis of
:math:`k+1` functions. The HDG method adds a *trace* unknown :math:`\lambda`
living on the cell faces, one value per face per variable, which is what couples
the cells to one another.

The layout of the global solution vector is

.. code-block:: text

   [ sigma | q | u | aux ]   per cell, for each cell in turn
   [ lambda ]                all face traces
   [ mu ]                    all global scalars

This ordering is shared by the solution vector (``DGSoln::Map``) and by the
per-cell Jacobian block ``MX``. Getting a column index wrong in that layout is
the most common way to break the solver silently, because — see below — a wrong
Jacobian does not produce a wrong answer.

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

The stabilisation parameter :math:`\tau` (config key ``tau``) is a constant.
Larger values weight the jump penalty between cells more strongly.

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
