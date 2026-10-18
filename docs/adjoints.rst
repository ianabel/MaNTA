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

To stop a solve on this quantity rather than estimate it afterwards, set
``SteadyStateTest = "Objective"``: the undamped Newton solve then ends when
:math:`|(\partial G/\partial y)\cdot J^{-1}F|` — :math:`G - G_{\text{corrected}}`
at the iterate, which is :math:`|z^T F|` for the adjoint state :math:`z` — is
inside the algebraic share of ``ObjectiveTolerance`` for every objective. It
reads the number off the correction the iteration solves for anyway, so it costs
no adjoint solve and no physics call. See :ref:`objective-steady-test`.

.. warning::

   :math:`G_{\text{err}}` bounds *solver* error only. It says how far this solve
   stopped short of its own fixed point, not how far that fixed point is from the
   continuum. It compares two runs at one discretisation and nothing else — which
   is the sweep's question, and not a discretisation error estimate.

.. _discretisation-estimate:

The discretisation error
------------------------

``EstimateDiscretisationError = true`` adds the other half: an estimate of how
far the fixed point is from the continuum, per objective and per cell, made
alongside the estimate above at the end of each steady solve. With it,

.. math::

   G \approx G_{\text{corrected}} - \eta_{\text{disc}}

and ``objectiveEstimate()`` reports :math:`\eta_{\text{disc}}` as
``discretisation``, its per-cell contributions :math:`\eta_K` as
``discretisation_per_cell`` (one row per objective, one column per cell, beside
``x_faces``), and :math:`G_{\text{corrected}} - \eta_{\text{disc}}` as
``continuum``. In C++ it is ``lastObjectiveEstimate().discretisation``, and the
solve prints one line per objective with all three numbers. It needs
``solveAdjoint`` and ``EstimateObjectiveOnFinish``, and is refused under a time
march, with global scalars or a field model, under a ``Diffusive`` tau, and at
:math:`k = 0`.

How it is made
''''''''''''''

Write :math:`R(U)` for the residual of the continuum problem in MaNTA's HDG form:
the rows ``residual()`` forms, with every interpolant replaced by the function it
interpolates and every integral exact. Per cell, for test functions :math:`v` of
any degree and the stored :math:`\sigma = -\hat\sigma` (:ref:`sign-convention`),

.. math::

   R_\sigma(v) &= (\sigma + \hat\sigma(u, q), v) \\
   R_q(v)      &= -(q, v) - (u, v') + \langle \lambda, v\, n \rangle \\
   R_u(v)      &= (\sigma', v) + \langle \tau (u - \lambda), v \rangle - (S, v)

and on each face the sum over its two cells of
:math:`\sigma n + \tau(u - \lambda)`, with the boundary data in all of them as
``residual()`` puts it. The exact solution — :math:`q = u'`,
:math:`\sigma = -\hat\sigma`, :math:`\lambda` the trace of :math:`u` — makes every
row vanish. For any state :math:`\tilde U` near it, linearising,

.. math::

   z^T R(\tilde U) \approx R'(U)\,(\tilde U - U) \cdot z
                   = G'(U)\,(\tilde U - U) \approx G(\tilde U) - G(U)

where :math:`z` solves :math:`R'(U)^T z = G'(U)`: the continuum adjoint, with
:math:`G` the exact integral of :math:`g`. Nothing at the scheme's own level can
evaluate this: at a converged state its residual :math:`F_h` is zero, which is
the whole content of having converged. So both :math:`R` and :math:`z` are taken
from a richer discretisation of the same mesh — the plain HDG scheme of degree
:math:`k_e`:

* :math:`\tilde U` is the discrete state written as a polynomial of degree
  :math:`k_e` per cell — :math:`u_h`, or :math:`u^*` under ``Superconvergent``,
  whose residual and objective are functions of :math:`u^*` — with its trace
  unchanged;
* :math:`F_e(\tilde U)` is that scheme's residual there, from the physics at its
  :math:`k_e + 1` nodes per cell;
* :math:`z_e` is that scheme's adjoint, :math:`J_e^T z_e = \partial G_e/\partial y`,
  from the physics derivatives at the same nodes.

Then

.. math::

   \eta_{\text{disc}} =
   \underbrace{z_e \cdot F_e(\tilde U)}_{\text{residual}}
   + \underbrace{G_h(u_h) - G_e(\tilde U)}_{\text{quadrature}}
   - \underbrace{\frac{\partial G_h}{\partial y} \cdot J^{-1} F_h}_{\text{algebraic}}

The residual part is :math:`G(\tilde U) - G(u)` to first order, the identity
above with the enriched pair standing in for the continuum one — equivalently,
the linearised difference between this level's objective and the one a degree
:math:`k_e` solve would return, which is the error exactly when that solve is far
more accurate. The quadrature part is what the objective's own rule leaves out:
:math:`G_h` is :math:`\sum_m w_m g(u_m)` on the scheme's physics nodes — on
:math:`u^*` at the star nodes under ``Superconvergent`` — and the same sum on the
enriched nodes stands in for the exact integral. The algebraic part is
:math:`G_h - G_{\text{corrected}}`: the residual part carries it as well, since a
state short of its fixed point has :math:`F_h \ne 0`, and it is removed so that
the two estimates add. The parts are also in ``DiscretisationEstimate`` in C++.

:math:`\eta_K` is the same sum restricted to cell :math:`K`, each face's trace
rows shared equally between the cells on either side of it, so the cells sum to
the total to round-off. **It is a split of the total, not a localisation of the
error**: the residual of each cell is tested against the whole enriched adjoint
rather than against the part of it the discretisation cannot represent, so
neighbouring cells carry large contributions that cancel —
:math:`\sum_K |\eta_K|` is 3700 times :math:`|\eta|` on :math:`u = x - x^{4/3}`
at 10 cells and :math:`10^6` times on a manufactured :math:`\sin \pi x` problem.
:ref:`adaptivity-objective-where` has what that means for adapting the mesh.

**The enriched degree is two beyond the degree the scheme interpolates its
physics at**: :math:`k_e = k + 2`, and :math:`k + 3` under ``Superconvergent``,
which already interpolates at :math:`k + 1`. One degree is not enough, and the
reason is measured rather than argued. The plain scheme's :math:`G` converges at
:math:`h^{k+2}` for even :math:`k` and :math:`h^{k+1}` for odd — the
interpolatory rule on :math:`k+1` Chebyshev points is exact one degree further
when :math:`k+1` is odd — so for even :math:`k` the next degree up converges at
the same rate as the scheme, and an estimate built on it is off by the constant
ratio of the two errors, which refinement does not remove: on the linear
manufactured problem it tends to 1.055 at :math:`k = 2` and 1.06 at
:math:`k = 4`. Under ``Superconvergent`` the scheme at degree :math:`k` returns the plain scheme's :math:`G` at :math:`k+1` on that
problem to every printed digit, so the same argument moves up one.

Measured effectivity
''''''''''''''''''''

:math:`\eta_{\text{disc}} / (G_{\text{corrected}} - G)` against closed forms, at
tightly converged solutions (``Tests/UnitTests/DiscretisationEstimateTests.cpp``,
``measure_effectivity``). Four steady manufactured problems on :math:`[0, 1]`
with :math:`u = \sin \pi x` — linear diffusion, the nonlinear flux
:math:`(1 + u^2)\,q`, a source :math:`-u^3`, and the diffusivity :math:`1 + x` —
each with four objectives, :math:`\int u`, :math:`\int x u`, :math:`\int u^2`
and :math:`\int x q`; :math:`k = 1` to 5, 4, 8 and 16 cells, leaving out the
rows whose error is below 1e-11, where round-off takes over:

.. list-table::
   :header-rows: 1

   * - scheme
     - improved adjoint
     - :math:`k_e`
     - median
     - range
     - within 10%
   * - plain
     - **enriched**
     - :math:`k+2`
     - 1.000
     - 0.84 – 1.44
     - 95%
   * - plain
     - enriched
     - :math:`k+1`
     - 1.000
     - 0.40 – 1.58
     - 81%
   * - plain
     - postprocessed
     - :math:`k+2`
     - 0.999
     - −1.27 – 5.28
     - 67%
   * - superconvergent
     - **enriched**
     - :math:`k+3`
     - 1.000
     - 0.88 – 1.00
     - 99%
   * - superconvergent
     - enriched
     - :math:`k+2`
     - 0.999
     - −1.42 – 2.48
     - 80%
   * - superconvergent
     - postprocessed
     - :math:`k+3`
     - 1.000
     - 0.73 – 5.08
     - 78%

The plain scheme's outliers are all on coarse meshes and go with refinement.
The worst two are errors passing through zero: 1.44 for the nonlinear flux with
:math:`\int u^2` at :math:`k = 3` on 4 cells, whose error changes sign between 4
and 8 cells (3.2e-5, then −1.6e-6; the effectivity is 0.97 on 8 and 0.998 on 16),
and 0.84 for the source :math:`-u^3` with :math:`\int x q` at :math:`k = 1`. The
rest are :math:`\int u^2` at :math:`k \le 2`, 1.10 – 1.24 on 4 and 8 cells: the
estimate is first order, and the term the linearisation drops is
:math:`\int (u_h - u)^2`. On two cells the plain scheme is not yet asymptotic
(−3.8 at worst), while the superconvergent one already holds 0.94 – 1.03.

A fifth problem routes the nonlinear flux through an auxiliary variable,
:math:`\hat\sigma = (1 + \phi)\,q` with :math:`\phi = u^2` and a source reading
:math:`\phi`. It reaches 1 more slowly: within 0.94 – 1.08 on 16 cells in both
schemes, but on 4 and 8 cells its error changes sign from one mesh to the next at
low :math:`k`, and where it passes near zero the ratio means nothing — 2.6 on the
plain scheme at :math:`k = 3` on 4 cells, an error of 4.6e-6 between errors of
9.7e-3 and −1.1e-6, and 5.1 superconvergent at :math:`k = 1`.

**No calibration is applied**; the median is 1.000 in both schemes.

**The postprocessed adjoint was the cheaper candidate and is not used.** It
replaces :math:`z_e` by :math:`z_h` with its :math:`u` block reconstructed as
``Postprocessor`` reconstructs a solution's — with :math:`z_\sigma` in the role
:math:`q` plays there, since the adjoint's :math:`\sigma` column,
:math:`(z_\sigma, w) = -(z_u, w') - \langle z_\lambda, w\,n\rangle`, makes
:math:`z_\sigma` the weak derivative of :math:`z_u` — which saves the derivative
evaluation and the enriched solve. It is exact enough for a linear problem with a
linear objective and erratic otherwise: at :math:`k = 2`, 0.26 – 0.29 for
linear diffusion with :math:`\int u^2`, and 0.23 – 0.34 for the nonlinear flux
with :math:`\int u`, on 4 to 16 cells. The first is the quadrature objective:
:math:`G_h = \sum_m w_m g(u_m)` has gradient :math:`\mathrm{diag}(w)\,\partial g`
rather than the mass-matrix projection of :math:`\partial g`, the two differ by a
mean-free pattern within each cell wherever :math:`\partial g` varies, and
:math:`z_h` inherits it — an error no local reconstruction removes, and for
:math:`g = u^2` the same order as the one being estimated. That does not explain
the second, which is presumably the :math:`\sigma` rows: the reconstruction
improves only :math:`z_u`, so :math:`z_\sigma` is lifted with nothing above
degree :math:`k`, while once :math:`\hat\sigma` is nonlinear the enriched
residual's :math:`\sigma` rows are nonzero precisely there. It
stays in the C++ API as ``DiscretisationAdjoint::Postprocessed``, the comparison
the test measures, and nothing else selects it.

On the benchmarks, by ``Runner`` (the effectivity against each case's closed
form, objective as in ``python-examples/objective-tolerance``):

* **Park** (:math:`\int x u`, the cylinder's stored energy): 0.98 – 1.04 across
  :math:`k = 1` to 5 on 2 to 16 cells, both schemes; within 0.5% from 4 cells on.
* **Shestakov** (:math:`\int u`, :math:`n_b = 0.05`, 10, 20 and 40 cells, so the
  source edge is a cell face): on 40 cells 0.75, 0.64, 0.55, 0.49, 0.43 for
  :math:`k = 1` to 5 on the plain scheme and 0.91, 0.87, 0.83, 0.80, 0.77
  superconvergent, within 0.08 of that on 10 and 20. Here it **underestimates,
  by a factor that depends on** :math:`k`, and that is a limitation of the
  method rather than noise. The
  error is the boundary-cell flux deficit described in
  ``python-examples/shestakov-nonlinear/ANALYSIS.md``, proportional to
  :math:`h/(k+1)^2`, and the enriched scheme commits the same one at its own
  degree, so the estimate is
  :math:`e_k - e_{k_e} = e_k\,(1 - ((k+1)/(k+3))^2)`: 0.75, 0.64, 0.56, 0.49,
  0.44 for the plain scheme, which is what is measured. With ``zeroFlux`` the
  deficit goes but the axis's :math:`x^{4/3}` remains, the error converges at
  about :math:`h^{2.3}` at every :math:`k`, and the effectivity is again steady
  in :math:`h` and erratic in :math:`k` (0.30 – 1.10 plain, 1.17 – 1.91
  superconvergent). With the source edge inside a cell (8 and 16 cells) it is
  erratic outright, −3.1 to 3.9.

So the estimate is trustworthy where the solution is smooth enough that two more
degrees buy a much smaller error — which on the smooth cases measured holds from
4 cells, and from 16 on the aux problem — and it **reports the part of the error that refinement in**
:math:`k` **removes**. An error that decays only algebraically in :math:`k`, from
a singularity or a data discontinuity the richer scheme resolves no better, is
underestimated, and nothing in the estimate itself says so.

What it costs
'''''''''''''

Per estimate: one ``ComputePhysics`` and one ``ComputePhysicsDerivatives`` at
:math:`N(k+3)` points — :math:`N(k+4)` under ``Superconvergent`` — which is the
evaluation plan's ``DiscretisationEstimate`` site, once per steady solve or slice
(cadence ``PerObjectiveEstimate``); in physics, :math:`(k+3)/(k+1)` of a Newton
iteration, or :math:`(k+4)/(k+2)` under ``Superconvergent``. ``MaxPhysicsBatch``
applies to that batch too, and a level whose enriched batch is past it is
refused before its first evaluation. Then the
enriched residual and adjoint operator are assembled and factorised at degree
:math:`k_e`, and one transposed solve is made per objective; the enriched solver
is built on the first estimate and kept, and asks the case for nothing.

On a case whose physics costs nothing (the nonlinear manufactured problem in
C++, ``measure_cost``) that assembly is what is timed: an estimate costs 3.1 to
4.9 times a residual plus a Jacobian build, over 64 cells at :math:`k = 2` to
8 cells at :math:`k = 8`, in either scheme. On the Python benchmarks it adds 7%
to 35% to a whole steady solve's wall time (59% once, on Shestakov's plain
scheme), and the physics point-evaluations rise by exactly :math:`N(k+3)` of each
kind, :math:`N(k+4)` superconvergent. The physics is what the estimate is for, and where the physics is
expensive the two evaluations are the cost.
