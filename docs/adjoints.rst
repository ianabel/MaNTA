Adjoints and gradients
======================

MaNTA can differentiate a scalar objective with respect to problem parameters by
the adjoint state method, at a cost that is **independent of the number of
parameters**. That is what makes gradient-based optimisation over a transport
model practical: one forward solve plus one adjoint solve gives the whole
gradient, however many parameters it has.

The objective
-------------

An ``AdjointProblem`` defines ``ng`` objectives, each of the form

.. math::

   G = \int g(u, q, \sigma, \phi, x) \, \mathrm{d}x

and declares ``np`` parameters with respect to which the gradient is wanted. A
transport system supplies its adjoint problem through
``createAdjointProblem()``; set ``solveAdjoint`` in the configuration to have it
built and solved.

What you implement
------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Hook
     - Returns
   * - ``gFn(gIndex, State, x)``
     - The integrand :math:`g`.
   * - ``dgFn_du``, ``dgFn_dq``, ``dgFn_dsigma``, ``dgFn_dphi``
     - Its derivatives with respect to each field.
   * - ``GFn(gIndex, DGSoln)``
     - The objective value. Usually the integral of ``gFn``.
   * - ``dGFndp(gIndex, pIndex, DGSoln)``
     - The *explicit* dependence of :math:`G` on a parameter — the part that does
       not come through the solution.
   * - ``dSigmaFn_dp``, ``dSources_dp``, ``dAux_dp``
     - How the flux, source and auxiliary constraints depend on each parameter.
       These are what make the gradient non-trivial.
   * - ``getName(pIndex)``
     - The parameter's name, used to label it in the output file.

Batched versions of most of these exist alongside the pointwise ones, with
default implementations that loop; the same pointwise/batched split as in
:doc:`physics_interface`.

Boundary parameters
-------------------

``np`` splits into ``np_boundary`` *trailing* boundary parameters and the rest,
which are internal. Several defaults loop only up to ``getNpInternal()``, so a
parameter that acts through a boundary condition rather than through the
equations must be placed in the trailing block and declared with
``computeLowerBoundarySensitivity`` / ``computeUpperBoundarySensitivity``.

How the gradient is computed
----------------------------

Three stages, after the forward integration:

.. code-block:: text

   initializeMatricesForAdjointSolve   assemble the adjoint operator
   solveAdjointState                   solve for the adjoint variables
   computeAdjointGradients             contract with dSigma/dp, dS/dp -> G_p

.. important::

   **The adjoint operator is the transpose of the forward Jacobian.**
   ``initializeMatricesForAdjointSolve`` assembles the same local blocks as
   ``updateMatricesForJacSolve`` and stores ``M.transpose()``, so the two
   functions must be kept in step block for block. If you add a coupling to one,
   add it to the other.

   This is where the two differ in *consequence*. A block missing from the
   forward Jacobian only slows Newton down — the answer is still right. A block
   missing from the adjoint matrix produces a **silently wrong gradient with a
   perfectly good objective value**. The ``dSigma/dPhi`` block was absent from
   the adjoint assembly for exactly that reason, and cost nothing visible until a
   test with ``nAux > 0`` was written.

Reading the gradients out
-------------------------

.. code-block:: python

   runner.configure(problem, {..., "solveAdjoint": True})
   runner.run(1.0)
   grads = runner.getAdjointGradients()      # dG/dp

The adjoint solve happens inside ``run`` whenever ``solveAdjoint`` is set, so by
the time it returns the gradients exist and ``getAdjointGradients`` merely reads
them. See :doc:`python` for the interaction with ``G()``.

.. note::

   ``WriteAdjoints()`` is commented out in ``Solver.cpp``, so **no run serialises
   the adjoint output**. The objective values and gradients do appear in the
   netCDF file as the ``G<i>``, ``G<i>_p`` and ``G<i>_boundary`` entries, and the
   gradients are verified through the Python interface rather than from file.

Verifying a gradient
--------------------

A gradient that is wrong in a way the solver cannot detect is the failure mode
here, so check it against something independent. The two checks used in the test
suite are worth copying:

* **Finite differences.** Perturb one parameter, re-run, and compare
  :math:`(G(p + h) - G(p - h)) / 2h` against the adjoint gradient. This is the
  general check and the one that catches a missing block.
* **A closed form.** Pick a problem whose objective is analytic. At a polynomial
  degree high enough that the discrete solution is exact, the computed
  :math:`G` should match the closed form to round-off, which pins the objective
  itself rather than just its derivative.

Limitations
-----------

* Spatial adjoint parameters — those indexed by node — are **rejected** when
  ``Superconvergent`` is set, because the postprocessed node set would silently
  redefine how many parameters there are. See :doc:`superconvergence`.
* Anything indexed per auxiliary variable is sized ``nAux``, not ``nVars``.

The dG/dt early-exit gate
-------------------------

An optimisation sweep spends most of its time on steps that turn out to be bad.
``ObjectiveDecreaseTolerance`` lets the solver notice some of them before paying
for the transport solve: after the initial condition is built, it evaluates

.. math::

   \frac{\mathrm{d}G}{\mathrm{d}t} = \int \left(
       \frac{\partial g}{\partial u} \dot u
     + \frac{\partial g}{\partial q} \dot q
     + \frac{\partial g}{\partial \sigma} \dot \sigma
     + \frac{\partial g}{\partial \phi} \dot \phi \right) \mathrm{d}x

and, if any objective is falling faster than the tolerance, abandons the run
without integrating. The convention is that :math:`G` is **maximised**, so a
decrease is the bad direction, and the tolerance is one-sided slack on that.

This works between ``initialize()`` and ``integrate()``, which is why those are
separate phases. From Python, ``Runner.wasRejected()`` reports the verdict and
``Runner.lastDGdt()`` the values behind it; a rejected run leaves the solver at
the initial condition, so ``G()`` still reads, and reports :math:`G(t_0)` rather
than a synthesised value. What a rejected step means for the search is the
driver's decision.

The derivatives come from the same ``dg`` hook and the same projection the adjoint
solve uses to build :math:`G_y`, so the gate and the gradients beside it answer
consistent questions, and no case has to implement anything new.

Where the algebraic derivatives come from
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three of the four terms above are not IDA's to give. :math:`q`, :math:`\sigma`
and :math:`\phi` are algebraic, and ``IDA_YA_YDP_INIT`` computes algebraic
*values* and differential *derivatives* — so at :math:`t_0` those blocks of IDA's
``dydt`` are identically zero, and a gate reading it would differentiate every
objective through its :math:`u` dependence alone. An objective depending on
:math:`q` alone would score exactly zero and could never be rejected.

They are **solved for rather than read**.
``SystemSolver::computeAlgebraicTimeDerivatives()`` differentiates the algebraic
residual rows in :math:`t`, which gives

.. math::

   \frac{\partial F}{\partial y} \, \dot y = -\frac{\partial F}{\partial t}

— a linear system in :math:`(\dot\sigma, \dot q, \dot\phi, \dot\lambda)` once
:math:`\dot u`, which IDA *does* have, is treated as data. The matrix is the
residual Jacobian with no mass term; the rows for the differential unknowns are
replaced by the identity, with the known derivative on the right, and one dense
factorisation finishes it. The answer goes to a separate ``dydtComplete`` vector,
never into IDA's own ``dYdt``, which is the state the integration starts from.

The gate therefore still evaluates at the initial condition, between
``initialize()`` and ``integrate()`` — no step is taken and the run's trajectory
is untouched — but it now evaluates the whole chain rule there. It costs one
assembly and factorisation of the whole system, once, and only on a run that arms
the gate.

.. note::

   :math:`-\partial F/\partial t` is the *explicit* time derivative of the
   residual, and nothing in the tree exposes it analytically:
   ``TransportSystem::LowerBoundary`` has no derivative counterpart and there is
   no ``dSigmaFn_dt`` or ``dAuxG_dt``. It is obtained by central-differencing
   ``residual()`` in :math:`t` with the state held fixed, so the algebraic
   derivatives are second order in the differencing step for a case with explicit
   time dependence and exact for an autonomous one. Measured against a
   manufactured solution the error is around :math:`10^{-11}`, against
   :math:`10^{-14}` for :math:`\dot u`, which is data rather than a difference.

   Note also that an objective linear in the state is the only kind whose
   :math:`\mathrm{d}G/\mathrm{d}t` is nonzero at a uniform initial condition —
   :math:`\int \tfrac{1}{2} u^2` has :math:`\partial g/\partial u = u`, which
   vanishes where :math:`u = 0`.

The gate is unavailable with ``Superconvergent`` set — it throws rather than
differentiate through the wrong projection — and it has no term for the global
scalars, because ``AdjointProblem`` has no ``dgFn_dscalars`` to go with the other
four.
