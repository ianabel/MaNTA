Writing a transport system
==========================

A *transport system* supplies the physics: the flux, the source, any auxiliary
constraints, the initial condition and the boundary conditions. Everything
outside ``PhysicsCases/`` is generic, and the solver knows nothing about what
your variables mean.

This page documents the interface. It deliberately does not document any of the
cases that ship in ``PhysicsCases/`` — read those as examples, but treat the
interface described here as the contract.

Declaring what a case is
------------------------

A case describes itself as data, once, and cannot be constructed without doing
so. That description is a ``SystemSpec``:

.. code-block:: cpp

   MyProblem::MyProblem(toml::value const &config, Grid const &grid)
       : TransportSystem({.variables = {{"n", "density",  "m^-3",
                                         BoundaryKind::Neumann,
                                         BoundaryKind::Dirichlet},
                                        {"T", "temperature", "eV"}},
                          .aux     = {{"phi", "electrostatic potential", "V"}},
                          .scalars = {{"I", "plasma current", "A", /*differential=*/true}}})
   {
       // config parsing only
   }

``nVars``, ``nScalars`` and ``nAux`` are derived from it and are ``const``; the
names reach the netCDF output; the boundary *kind* is per variable and per end;
and ``differential`` marks a scalar whose constraint involves :math:`\dot y`.

The spec is validated in the constructor — it must declare at least one
variable, and names are unique across all three groups — so a case that fails
validation never becomes an object.

If the shape depends on the configuration, build the spec in a static helper and
pass it up, so it is still complete before the base class exists:

.. code-block:: cpp

   class MyProblem : public TransportSystem
   {
       static SystemSpec buildSpec(toml::value const &config);
       MyProblem(toml::value const &config, Grid const &grid)
           : TransportSystem(buildSpec(config)) { ... }
   };

.. note::

   ``numberedFields(n)`` / ``numberedScalars(n)`` / ``numberedAux(n)`` in
   ``SystemSpec.hpp`` produce the placeholder names ``Var0``, ``Scalar0``,
   ``AuxVariable0``. They are for a case whose *width comes from its
   configuration* and which therefore has no names to give —
   ``MatrixDiffusion``, ``MatrixDiffusionTest`` and ``LinearDiffSourceTest`` are
   the three in the tree. Every other case names its variables, and the netCDF
   groups take those names.

Two ways in
-----------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Base class
     - When to use it
   * - ``TransportSystem``
     - You write the flux, the source, and their derivatives. Full control;
       more to get wrong.
   * - ``AutodiffTransportSystem``
     - You write only ``Flux`` and ``Source``, in ``autodiff`` types, and every
       Jacobian entry is derived for you. Prefer this unless you have a reason
       not to.

The autodiff route
------------------

Derive from ``AutodiffTransportSystem`` and implement two functions:

.. code-block:: cpp

   Real Flux(Index i, RealVector u, RealVector q, Real x, Time t) override;

   Real Source(Index i, RealVector u, RealVector q, RealVector sigma,
               RealVector phi, Real x, Time t) override;

Its constructor takes the spec alongside the config and grid:

.. code-block:: cpp

   MyCase::MyCase(toml::value const &config, Grid const &grid)
       : AutodiffTransportSystem(config, grid, buildSpec(config)) {}

``Flux`` returns :math:`\hat\sigma_i` — the physical flux, exactly what
``SigmaFn`` would return. The sign convention discussed in
:ref:`the formulation <sign-convention>` is applied by the solver, not by you.

An ``[AutodiffTransportSystem]`` config section may set ``isUpperDirichlet`` /
``isLowerDirichlet``, which are applied to every variable; a case that declares
per-variable boundary kinds in its own spec keeps them unless those keys are
present.

.. tip::

   ``autodiff`` expression templates hold **references** to their operands. A
   lambda whose result is stored — anything passed to an integrator, for instance
   — must declare ``-> Real`` explicitly. With a deduced return type it hands back
   an expression referring to dead temporaries, and the symptom is a silently
   wrong (often zero) answer rather than a crash.

The explicit route
------------------

Derive from ``TransportSystem``. The pure virtuals you must implement:

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Hook
     - Returns
   * - ``SigmaFn(i, State, x, t)``
     - :math:`\hat\sigma_i`, the flux.
   * - ``Sources(i, State, x, t)``
     - :math:`S_i`, the source.
   * - ``dSigmaFn_du``, ``dSigmaFn_dq``
     - Derivatives of the flux with respect to :math:`u` and :math:`q`.
   * - ``dSources_du``, ``dSources_dq``, ``dSources_dsigma``
     - Derivatives of the source.
   * - ``InitialValue(i, x)``, ``InitialDerivative(i, x)``
     - The initial condition and its :math:`x` derivative.

Reading the ``State``
~~~~~~~~~~~~~~~~~~~~~

Every hook receives a ``State``. Use the named accessors:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Accessor
     - Meaning
   * - ``s.u(i)``
     - the variable :math:`u_i`
   * - ``s.q(i)``
     - its derivative :math:`q_i = \partial_x u_i`
   * - ``s.sigma(i)``
     - the **stored** flux, :math:`-\hat\sigma_i`
   * - ``s.sigmaHat(i)``
     - the **physical** flux, the quantity ``SigmaFn`` returns
   * - ``s.phi(i)``
     - the auxiliary variable :math:`\phi_i`
   * - ``s.scalar(i)``
     - the global scalar :math:`\mu_i`

They are bounds-checked in a ``DEBUG`` build, which is worth having: anything
indexed per auxiliary variable is sized ``nAux``, not ``nVars``, and those
coincide in almost every case.

.. note::

   The derivative hooks receive their output vector **already zeroed**. Assign
   only the entries that are nonzero; there is no need to call ``setZero()``
   first, and an omitted entry means zero rather than uninitialised memory.

Boundary conditions
~~~~~~~~~~~~~~~~~~~

The *kind* is spec data. Only the *value* is a function you write, because it
can depend on :math:`t`:

.. code-block:: cpp

   Value LowerBoundary(Index i, Time t) const override;
   Value UpperBoundary(Index i, Time t) const override;

A Dirichlet boundary fixes :math:`u`; a Neumann boundary fixes the flux.
``aFn(i, x)`` supplies the coefficient :math:`a_i` multiplying
:math:`\partial_t u_i`, and defaults to 1.

Auxiliary variables and global scalars
--------------------------------------

Declare ``aux`` in the spec and implement ``AuxG`` (the constraint
:math:`G_j = 0`) and ``AuxGPrime``, plus ``dSources_dPhi`` and ``dSigma_dPhi``
so the solver knows how the flux and source depend on :math:`\phi`.
``InitialAuxValue`` seeds them.

Declare ``scalars`` in the spec and implement two hooks, plus
``InitialScalarValue``, ``InitialScalarDerivative`` (consulted only for
differential scalars) and ``dSources_dScalars`` — which is indexed by
**scalar**, not by variable:

.. code-block:: cpp

   Value ScalarG(Index s, GlobalState const &y, GlobalState const &ydot,
                 std::vector<Position> const &abscissae, Values const &weights,
                 Matrix const &phiBoundary, Time t) override;

   void ScalarGPrime(GlobalStateMatrix &dG, GlobalStateMatrix &dGdot,
                     GlobalState const &y, GlobalState const &ydot,
                     std::vector<Position> const &abscissae, Values const &weights,
                     Matrix const &phiBoundary, Time t) override;

A scalar constraint is a functional of the whole solution, so it is handed the
solution sampled on the element nodes rather than pointwise data:

* ``weights`` gives one quadrature weight per node, so an integral over the
  domain is ``ScalarHooks::integrate(y.Variable().row(i), weights)``. **Use
  these rather than a quadrature rule of your own** — see the warning below.
* ``phiBoundary`` is :math:`(k+1)\times 2`, the basis functions of the first and
  last cells evaluated at the ends of the domain. The nodes are Chebyshev points
  of the first kind and so are strictly interior, which makes this the only way
  to reach a boundary point value: ``ScalarHooks::boundaryValue(...)``.
* ``abscissae`` gives each node's position, for an integrand that depends on
  :math:`x` itself.

``ScalarGPrime`` reports **every** scalar at once, as the derivative with
respect to the degrees of freedom. For :math:`G = \mu - \int u\,dx` the
:math:`u` entry at node :math:`j` is :math:`-w_j`. ``dG[s]`` and ``dGdot[s]``
arrive zeroed. ``ScalarHooks::addBoundaryDerivative`` is the counterpart of
``boundaryValue`` for a constraint on a boundary point.

.. warning::

   Integrating with your own rule is how ``ScalarTestLD3`` came to disagree with
   itself. It computed its mass with a global adaptive Kronrod rule over a
   piecewise polynomial, which is not a smooth function of the coefficients, and
   its finite-difference reference disagreed with the exact :math:`\int\phi` by
   8% at :math:`k = 4` on 16 cells. The weights you are handed are the ones the
   solver integrates with.

Pointwise and batched
---------------------

**Every physics hook exists in two forms**: pointwise, taking a ``State`` and one
position, and batched, taking a ``GlobalState`` and a vector of positions. The
batched versions have default implementations in ``TransportSystem.hpp`` that are
serial loops over the pointwise version, several of them under
``#pragma omp parallel for`` when built with ``OMP=on``.

A case may override either level. Overriding the batched level is how a
vectorised implementation — a JAX case, say — avoids being called once per point.
``ComputePhysics`` and ``ComputePhysicsDerivatives`` are the whole-state entry
points, and because they loop over ``states.size()`` rather than a fixed count,
they do not care how many points they are handed. That is what allows
``Superconvergent`` to evaluate the physics at :math:`k+2` points per cell
instead of :math:`k+1` without any physics case changing.

Diagnostics
-----------

``initialiseDiagnostics`` and ``writeDiagnostics`` let a case add derived
quantities to the netCDF file alongside the solution. Names, descriptions and
units come from the spec and need no hooks.

Registration
------------

Two macros connect a class name to the string used in a config file:

.. code-block:: cpp

   // in the header, inside the class
   REGISTER_PHYSICS_HEADER(MyProblem)

   // in the .cpp, at file scope
   REGISTER_PHYSICS_IMPL(MyProblem)

These declare and define a ``PhysicsCaseRegister<MyProblem>`` whose constructor
inserts a factory into a process-global map during static initialisation. Two
consequences worth knowing:

* **A case only appears if its object file is linked in.** Nothing references it
  directly, so a missing entry in the build is a link-line problem that produces
  **no compile error**. An unrecognised ``TransportSystem`` name now throws with
  the list of what *is* registered, which is usually enough to spot it. Add new
  files to ``PHYSICS_SOURCES`` in the Makefile, or build the case
  :doc:`out of tree <out_of_tree>`.
* **A duplicate name is an error.** Registration used to be a silent no-op that
  kept the first registration, leaving the second case unreachable with nothing
  said; it throws now.

The constructor takes the parsed config and the grid:

.. code-block:: cpp

   explicit MyProblem(toml::value const &config, Grid const &grid);

so a case can read its own table out of the config file. Remember that unknown
keys are silently ignored; a typo in your own parameter name will leave you with
the default.

Debugging a new case
--------------------

The failure modes here are unusual enough to be worth stating plainly.

A wrong **derivative** hook does not give a wrong answer. Because the Jacobian is
never assembled, an error there costs Newton iterations, not accuracy — so the
run may simply be slow, or fail to converge, with no incorrect output to inspect.
The test that catches it is the one that finite-differences the residual and
requires :math:`J\,\delta y = g`. For a scalar constraint, ``checkScalarDerivative``
in ``Tests/UnitTests/ScalarJacobianTests.cpp`` does the same for any case and is
the first thing to run when a scalar system converges slowly.

A wrong **sign** on the flux does give a wrong answer, but a plausible-looking
one that converges at the correct rate to the wrong function. Only a comparison
against a closed-form solution finds it. See :doc:`testing` for both, and
:ref:`the sign convention <sign-convention>` for why.
