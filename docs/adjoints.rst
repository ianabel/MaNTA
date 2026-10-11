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

The corrected objective and its error bar
-----------------------------------------

A steady solve stops when :math:`\|F\|` is small, not when :math:`G` is. An
optimisation sweep comparing :math:`G` at two parameter points therefore needs to
know how much of the difference is the answer moving and how much is each solve
stopping short, and ``estimateObjective()`` answers both. Both quantities it needs
are already assembled — :math:`\partial G/\partial y` is ``G_y``, which
``initializeMatricesForAdjointSolve`` builds per objective, and the Newton step to
the solution is :math:`J^{-1}F` from the matrix the solve has already factorised:

.. math::

   G_{\text{corrected}} = G - \frac{\partial G}{\partial y}\cdot J^{-1}F,
   \qquad
   G_{\text{err}} = \left\|\frac{\partial G}{\partial y}\right\| \left\|J^{-1}F\right\|

the first a first-order extrapolation to the fixed point, the second a
Cauchy–Schwarz bound on what is left. The Jacobian is taken at :math:`\alpha = 0`,
the steady operator, whatever the solve was damped with: the fixed point being
extrapolated to is a steady state, not the end of a pseudo-time step.

It runs at every exit from ``solveSteadyState``, including the two that then
throw, so a caught failure still carries a partial answer and a bound saying what
that answer is worth. ``lastSteadyOutcome()`` says which exit it was.
``PyRunner::objectiveEstimate()`` returns ``value``, ``corrected`` and
``uncertainty``, one entry per objective; it is empty when the run had no
``AdjointProblem``, and it costs nothing in that case.

**Solve looser and correct, rather than converging harder.** Measured on
``AdjointTestProblem``, against a reference two orders tighter:

.. list-table::
   :header-rows: 1

   * - ``SteadyStateTolerance``
     - residuals
     - raw error
     - corrected error
   * - 1e-2
     - 31
     - 2.1e-3
     - **1.7e-6**
   * - 1e-3
     - 41
     - 6.0e-6
     - < 1e-10
   * - 1e-4
     - 47
     - 5.6e-6
     - < 1e-10
   * - 1e-6
     - 65
     - < 1e-10
     - < 1e-10

Correcting a solve at 1e-2 is **34% cheaper and three times more accurate** than
running one at 1e-4 and taking :math:`G` raw. The bound holds throughout —
replaying every continuation step of a tight solve, it is 1.04x the true error at
its tightest and 2.3x at its loosest.

.. warning::

   :math:`G_{\text{err}}` bounds *solver* error only. It says how far this solve
   stopped short of its own fixed point, not how far that fixed point is from the
   continuum. It compares two runs at one discretisation and nothing else — which
   is the sweep's question, and not a discretisation error estimate.
