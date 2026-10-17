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

After the forward run, at its final state:

.. code-block:: text

   adjointDerivativesAt        the physics derivatives there: the last Jacobian
                               build's, or one evaluation if it was elsewhere
   assembleAdjointOperator     J^T at alpha = 0 from them, once for all objectives
   assembleObjectiveGradient   dG/dy for one objective, from its dg
   solveAdjointState           J^T z = dG/dy
   accumulateAdjointGradients  contract z with dSigma/dp, dS/dp -> G_p

.. important::

   **The adjoint operator is the transpose of the forward Jacobian.**
   ``assembleAdjointOperator`` assembles the same local blocks as
   ``assembleCellMatrix`` does for ``updateMatricesForJacSolve``, from the same
   physics derivatives, and stores ``M.transpose()``, so the two functions must
   be kept in step block for block. If you add a coupling to one, add it to the
   other.

   This is where the two differ in *consequence*. A block missing from the
   forward Jacobian only slows Newton down — the answer is still right. A block
   missing from the adjoint matrix produces a **silently wrong gradient with a
   perfectly good objective value**. The ``dSigma/dPhi`` block was absent from
   the adjoint assembly for exactly that reason, and cost nothing visible until a
   test with ``nAux > 0`` was written.

The derivatives are the last Jacobian build's whenever that build was taken at
the final state, bit for bit. A steady solve that estimates its objective on the
way out (see below) builds there, so its adjoint costs **no evaluation of the
physics at all**. A time march's last build is at one of IDA's iterates, and a
steady solve that did not estimate stopped one Newton correction past its last
build; for those the derivatives are evaluated once at the final state — once,
not once per objective — which is the evaluation plan's ``Adjoint`` site.

Reading the gradients out
-------------------------

.. code-block:: python

   runner.configure(problem, {..., "solveAdjoint": True})
   runner.run(1.0)
   grads = runner.getAdjointGradients()      # dG/dp

The adjoint solve happens inside ``run`` whenever ``solveAdjoint`` is set, so by
the time it returns the gradients exist and ``getAdjointGradients`` merely reads
them. See :doc:`python` for the interaction with ``G()``.

.. _adjoint-on-demand:

The adjoint on demand
---------------------

The adjoint state itself, and what it says about the physics, can be had from
Python wherever a state exists — after a run, or between the slices of a steady
solve — and **without calling the physics**:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Call
     - Returns
   * - ``solve_adjoint(g=None)``
     - Solves :math:`J^T z = \partial G/\partial y` at the current state for
       objective ``g``, or for every objective, and keeps the answers.
   * - ``adjoint_state(g)``
     - :math:`z` as nodal arrays: ``sigma``, ``q`` and ``u`` of shape
       ``(nPoints, nVars)``, ``aux`` of shape ``(nPoints, nAux)``, ``lambda``
       of shape ``(nCells + 1, nVars)`` on the faces, ``field`` of shape
       ``(nField,)``, the node positions ``x`` and face positions ``x_faces``;
       and ``operator_at_state``.
   * - ``objective_gradient(g)``
     - :math:`\partial G/\partial y`, the right-hand side, in the same layout.
       Its ``lambda`` is zero, since no objective reads the trace. Needs no
       solve.
   * - ``objective_sensitivity(g)``
     - :math:`w = \partial G/\partial \Gamma` for the flux and the sources:
       ``flux`` and ``sources`` of shape ``(nPoints, nVars)``, one row per
       physics node ``x``, and ``operator_at_state``.

The layouts are the batched interface's — a ``GlobalState`` crosses as
``(nPoints, nVars)`` arrays — with ``nPoints = nCells * (k + 1)`` for the nodal
arrays. The physics nodes of ``objective_sensitivity`` are the ones the case is
evaluated at: the :math:`k+2` star nodes under ``Superconvergent``, the
:math:`k+1` basis nodes otherwise. :math:`z_u` is the discrete adjoint function
in the nodal basis: moving the source by :math:`\delta S` moves :math:`G` by
:math:`\int z_h \, \delta S \, \mathrm{d}x`, and :math:`z_\sigma` is its
derivative's counterpart for the flux. The objective comes from the case's
``createAdjointProblem`` whether or not ``solveAdjoint`` is set, as ``G()``'s
does.

**The current state** is the one ``getSolution()`` reads: the final state after
a run, and the last slice's between the slices of a steady solve. An answer is
kept for the state it was solved at and withdrawn once the state moves —
``adjoint_state`` then raises until ``solve_adjoint`` is called again. A run
with ``solveAdjoint`` set leaves its end-of-run states behind, so they can be
read without another solve.

No physics call
'''''''''''''''

The operator is :math:`\partial F/\partial y` at :math:`\alpha = 0`, assembled
from the physics derivatives the last Jacobian build took and transposed, so the
case is asked for nothing: not ``ComputePhysics``, not
``ComputePhysicsDerivatives``, not a pointwise hook. What is called is the
objective's own ``dg``, once per objective, for the right-hand side — and with a
field model, the model's ``Geometry`` on the nodes and its own factorisation of
:math:`B`. For an expensive flux model that is the difference between an adjoint
solve per Newton iteration being affordable and not. A unit test counts every
entry point of a case and requires zero across a solve, its sensitivity and its
right-hand side.

The price is which Jacobian it is. ``operator_at_state`` is true when the last
build was at the current state: a run with ``solveAdjoint``, and a steady solve
or slice that estimated its objective on the way out, both leave one there.
Otherwise — between Newton iterations, or after a run that did neither — it is
the Jacobian at the last iterate built, against :math:`\partial G/\partial y` at
this one, and :math:`z` carries an error first order in the distance between
them. For a linear problem the two are the same matrix.

The adjoint keeps its own factorisations, so an adjoint solve between two
forward ones leaves the forward solve bit for bit as it was: a sliced steady
solve with an adjoint solve between every slice ends where one without them
does. It is refused with global scalars, whose bordering its operator does not
carry, and under ``tauScaling = "Diffusive"``, whose
:math:`\mathrm{d}\tau/\mathrm{d}y` it does not.

``manta.jax.FFIRunner`` inherits all four calls: they run on the host and touch
no device memory.

The sensitivity to the physics
''''''''''''''''''''''''''''''

Let :math:`\Gamma_m` be a physics output at physics node :math:`m` — the flux
:math:`\hat\sigma` the case returns, or its source. The residual reads both
linearly, through the projection :math:`P` onto the test space (the cell mass
matrix on the basis nodes, ``A9`` on the star nodes): the sigma rows carry
:math:`+P\,\Gamma_{\text{flux}}` and the u rows :math:`-P\,\Gamma_{\text{source}}`.
With :math:`J^T z = \partial G/\partial y`,

.. math::

   w_m = \frac{\partial G}{\partial \Gamma_m}
       = -z^T \frac{\partial F}{\partial \Gamma_m},
   \qquad
   w_{\text{flux}} = -P^T z_\sigma,
   \qquad
   w_{\text{sources}} = P^T z_u .

That says where the objective is sensitive to the physics, which is a statement
about model uncertainty in its own right: a flux known to 10% at node :math:`m`
puts up to :math:`0.1\,|w_m \hat\sigma_m|` into :math:`G`.

**A sign or operator error in** :math:`w` **is invisible to every aggregate**, so
it is checked node by node: the unit tests rerun the steady solve with one node's
flux or source moved by :math:`\pm h` and difference :math:`G`, with and without
``Superconvergent``, and :math:`w` matches to :math:`10^{-9}` of its size. Beside
it the same check of the other candidate — :math:`z(x_m)` times the node's
quadrature weight, which is what ``diag(w)`` in place of :math:`P` would give —
fails by orders of magnitude more, which is what says the data can tell the two
apart (the trap recorded beside ``DerivativeSubVector`` in
``AdjointVectors.cpp``). Against the continuum the two sensitivities are
:math:`\int z\,\ell_m\,\mathrm{d}x` and :math:`-\int z'\,\ell_m\,\mathrm{d}x`,
:math:`\ell_m` the basis function of node :math:`m`, and :math:`w` agrees with
both to discretisation accuracy.

Writing the adjoint out
-----------------------

``WriteAdjoints = true``, with ``solveAdjoint`` and ``WriteOutput``, writes at
the end of the run:

* the scalars ``ng``, ``np`` and ``np_boundary``, and ``G<i>`` per objective —
  the :math:`u^\star`-based objective under ``Superconvergent``, the one
  ``G_p`` differentiates;
* the groups ``G<i>_p`` and ``G<i>_boundary``, holding the gradient with respect
  to each named parameter: a scalar each, or with spatial parameters an array on
  the group's dimension ``node`` beside the node positions ``x``;
* the adjoint state beside the solution: ``z<i>_u``, ``z<i>_q`` and
  ``z<i>_sigma`` in each variable's group, sampled on the output grid like the
  solution and with no time dimension; ``<aux>_z<i>`` beside each auxiliary
  variable; and ``<dof>_z<i>`` in a field model's group.

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

A steady solve stops when its Newton correction is small, not when the error in
:math:`G` is. An
optimisation sweep comparing :math:`G` at two parameter points therefore needs to
know how much of the difference is the answer moving and how much is each solve
stopping short, and ``estimateObjective()`` answers both. It needs
:math:`\partial G/\partial y`, which costs only the objective's own ``dg``, and
the Newton step to the solution, :math:`J^{-1}F`, from one residual and one
Jacobian build at the current state:

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

**Under the default test the correction has little left to do.** Measured on
``AdjointTestProblem`` by ``PseudoTransient``, against a reference solved to a
``SteadyStateStepTolerance`` of 1e-8, in physics sweeps (residuals plus
Jacobians):

.. list-table::
   :header-rows: 1

   * - test, threshold
     - sweeps
     - raw error
     - corrected error
   * - ``NewtonStep``, 3.3
     - 14
     - < 1e-15
     - < 1e-15
   * - ``NewtonStep``, 0.33 (default)
     - 14
     - < 1e-15
     - < 1e-15
   * - ``NewtonStep``, 1e-4
     - 14
     - < 1e-15
     - < 1e-15
   * - ``Residual``, 1e-2
     - 11
     - 2.8e-4
     - **8.0e-8**
   * - ``Residual``, 1e-3
     - 15
     - 2.5e-8
     - < 1e-13

``NewtonStep`` bounds the same :math:`J^{-1}F` the correction applies, and a
correction that passes its test leaves the next one quadratically smaller, so
every threshold tried stops with :math:`G` at round-off, for about what
``Residual`` spends at 1e-3. Where the correction earns its place is a
``Residual`` solve stopped loosely: at 1e-2 it recovers three and a half orders
of :math:`G` from one Jacobian solve. By ``Newton`` both tests reach round-off
in six sweeps. The bound holds throughout -- replaying every continuation step
of a tight ``Residual`` solve, it is 1.04x the true error at its tightest and
2.3x at its loosest.

.. warning::

   :math:`G_{\text{err}}` bounds *solver* error only. It says how far this solve
   stopped short of its own fixed point, not how far that fixed point is from the
   continuum. It compares two runs at one discretisation and nothing else — which
   is the sweep's question, and not a discretisation error estimate.
