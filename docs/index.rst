MaNTA
=====

**MaNTA** — the Maryland Nonlinear Transport Analyzer — solves one-dimensional
nonlinear reaction–diffusion and transport systems. It discretises space with a
hybridizable discontinuous Galerkin (HDG) method and integrates the result in
time as an index-1 differential-algebraic system using SUNDIALS IDA.

For each of ``nVars`` variables :math:`u_i(x, t)` it advances

.. math::

   a_i \, \partial_t u_i - \partial_x \left[ \hat\sigma_i(u, q, x, t) \right]
       = S_i(u, q, \sigma, \phi, x, t),
   \qquad q_i = \partial_x u_i,

subject to Dirichlet, Neumann or mixed conditions at each end of the domain. Alongside
those it can carry ``nAux`` algebraic auxiliary fields :math:`\phi_j`, defined
pointwise by constraints :math:`G_j = 0`, and ``nScalars`` global unknowns
:math:`\mu` that are not functions of :math:`x` at all — a total plasma current,
say, or a control-loop state.

The physics — the flux :math:`\hat\sigma_i`, the source :math:`S_i`, the
constraints — is supplied by a *transport system*, written either in C++ against
:doc:`the TransportSystem interface <physics_interface>` or
:doc:`in Python <python>`. Everything outside ``PhysicsCases/`` is generic.

What MaNTA can additionally do
------------------------------

* **Adjoints.** Define an objective :math:`G = \int g \, \mathrm{d}x` and MaNTA
  will return :math:`\mathrm{d}G/\mathrm{d}p` for the parameters you nominate,
  by the adjoint state method — at a cost independent of the number of
  parameters. See :doc:`adjoints`.
* **Superconvergence.** Every run reconstructs a postprocessed solution
  :math:`u^\star` that is one polynomial degree richer than :math:`u_h`, and an
  opt-in flag makes the discretisation itself exploit it. See
  :doc:`superconvergence`.
* **Steady states.** ``run_ss`` integrates until :math:`\mathrm{d}y/\mathrm{d}t`
  falls below a tolerance rather than to a fixed final time.
* **Restarts.** Every run writes a restart file that a later run can resume from.

.. note::

   These pages document the solver, its configuration and its interfaces. They
   deliberately do **not** document any individual physics case in
   ``PhysicsCases/``; :doc:`physics_interface` describes the interface such a
   case implements, which is the part that is stable.

Where to start
--------------

Building it and running something is :doc:`install` then :doc:`configuration`.
If you want to understand what the solver is actually doing — and in particular
the flux sign convention, which is the most common way to get a new physics case
subtly wrong — read :doc:`formulation` first.

.. toctree::
   :maxdepth: 2
   :caption: Using MaNTA

   install
   configuration
   running

.. toctree::
   :maxdepth: 2
   :caption: How it works

   formulation
   superconvergence

.. toctree::
   :maxdepth: 2
   :caption: Extending MaNTA

   physics_interface
   python
   out_of_tree
   adjoints

.. toctree::
   :maxdepth: 2
   :caption: Development

   testing
