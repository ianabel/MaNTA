Superconvergence and ``u_star``
===============================

HDG methods admit an element-local postprocessing that produces a solution one
polynomial degree richer than the one the solver carries, and which converges one
order faster. MaNTA computes it on every run, and can optionally use it inside the
discretisation.

``u_star``
----------

For any run with ``PolynomialDegree`` :math:`\ge 1`, each variable's output group
gains a field ``u_star``: the postprocessed solution
:math:`u^\star \in P_{k+1}`, reconstructed cell by cell from :math:`(u_h, q_h)`.

This costs two small matrix–vector products per variable per cell and is done
whether or not the ``Superconvergent`` flag is set — the flag controls whether the
*method* uses :math:`u^\star`, not whether it is computed. So you can always plot
``u_star`` against ``u`` and see what the extra order buys.

The reconstruction solves a local Neumann problem on each cell,

.. math::

   (\partial_x u^\star, \partial_x z)_K &= (q_h, \partial_x z)_K
       \quad \forall z \in P_{k+1}(K) \\
   (u^\star, 1)_K &= (u_h, 1)_K

— the gradient is matched to :math:`q_h` in a least-squares sense, and the cell
mean is pinned to that of :math:`u_h` to fix the constant. Eliminating the
Lagrange multiplier gives per-cell operators that are built once and reused.

The ``Superconvergent`` flag
----------------------------

.. code-block:: toml

   [configuration]
   Superconvergent = true

With this set, the physics is evaluated at the :math:`k+2` nodes of the degree-
:math:`k+1` basis, with :math:`u^\star` in place of :math:`u_h`, and the result is
interpolated into :math:`P_{k+1}` rather than :math:`P_k`. This is the
interpolatory HDG\ :sub:`k` scheme of Chen, Cockburn, Singler and Zhang
(*J. Sci. Comput.* **81**, 2188).

Both halves matter. Interpolating a non-polynomial source into :math:`P_k`
introduces an :math:`O(h^{k+1})` consistency error with no orthogonality against
the test space, which alone caps the rate; interpolating into :math:`P_{k+1}`
makes it :math:`O(h^{k+2})` and harmless.

The Jacobian gains a chain factor for each block, and the only genuinely new
coupling is that :math:`u^\star` on a cell depends on that cell's :math:`q` as
well as its :math:`u`. Everything else is unchanged: each block stays
:math:`(k+1) \times (k+1)`, :math:`u^\star` is cell-local, and so the degree-of-
freedom layout, the static condensation, the restart format and the Python type
casters are all untouched. **No physics case needs changing** to run under the
flag, in C++, Python or JAX — the batched hooks loop over however many points they
are given.

The flag defaults to ``false``, and with it off the discretisation is exactly
what it was before the option existed.

Measured orders
---------------

Observed order of accuracy for :math:`u = \sin(\pi x)(1+t)`, flag off and flag on:

.. list-table::
   :header-rows: 1
   :widths: 34 8 20 20

   * - Case
     - :math:`k`
     - off: :math:`u`, :math:`u^\star`
     - on: :math:`u`, :math:`u^\star`
   * - linear, constant :math:`\kappa`
     - 1
     - 1.96, 2.19
     - 1.96, **3.05**
   * - linear, constant :math:`\kappa`
     - 2
     - 2.97, 4.08
     - 2.97, 4.03
   * - nonlinear reaction :math:`u^3 - u`
     - 1
     - 1.96, 2.23
     - 1.97, **3.07**
   * - nonlinear reaction :math:`u^3 - u`
     - 2
     - 2.95, 4.10
     - 2.97, 4.03

With the flag on, :math:`u^\star` reaches :math:`k+2` in every case and
:math:`u_h` keeps its optimal :math:`k+1`.

.. note::

   That table is not what a first reading of the literature predicts, and it is
   worth being precise about what it does and does not say.

   With the flag **off**, :math:`u^\star` superconverges at :math:`k = 2` but not
   at :math:`k = 1` — and this is true whether or not the source is nonlinear. The
   papers attribute the loss of superconvergence to :math:`I_h F(u_h)` evaluating
   :math:`F` at the :math:`O(h^{k+1})`-accurate :math:`u_h`, which would predict
   that the nonlinear rows differ from the linear ones. They do not.

   So whatever caps the rate at :math:`k = 1` here is not the mechanism the
   papers describe, and the :math:`k = 2` rows say the interpolatory method is not
   universally losing superconvergence in MaNTA. A partial explanation: nodal
   interpolation of a *known* smooth source at the Chebyshev nodes leaves an error
   very nearly orthogonal to :math:`P_k`, so it does not pollute the duality
   argument the way evaluating :math:`F` at an approximate :math:`u_h` does.

   The tests therefore assert only what is measured — that :math:`u^\star` reaches
   :math:`k+2` with the flag on and that :math:`u_h` does not regress. They do not
   assert that the flag improves on the flag-off rate, because for these problems
   there is not always anything to improve.

What is not covered
-------------------

* **A general nonlinear flux** :math:`\hat\sigma(u, q)` is outside the papers'
  theory — their conclusion names :math:`F(\nabla u, u)` as open. The Jacobian is
  verified for such a flux, but no order study asserts :math:`k+2` for one.
* :math:`k = 0` is **rejected** with ``std::invalid_argument``. The degree-0 basis
  cannot be evaluated away from its node, and the theory requires
  :math:`k \ge 1` regardless.
* **Spatial adjoint parameters** are rejected with the flag on; see
  :doc:`adjoints`.
* ``nAux > 0`` and ``nScalars > 0`` are handled by the flag-on Jacobian, but no
  order study measures them.
