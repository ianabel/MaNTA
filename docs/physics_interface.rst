Writing a transport system
==========================

A *transport system* supplies the physics: the flux, the source, any auxiliary
constraints, the initial condition and the boundary conditions. Everything
outside ``PhysicsCases/`` is generic, and the solver knows nothing about what
your variables mean.

This page documents the interface. It deliberately does not document any of the
cases that ship in ``PhysicsCases/`` — read those as examples, but treat the
interface described here as the contract.

Two ways in
-----------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Base class
     - When to use it
   * - ``TransportSystem``
     - You write the flux, the source, **and every derivative of them**. Full
       control; more to get wrong.
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

``Real`` is ``autodiff::dual`` and ``RealVector`` is ``autodiff::VectorXdual``,
so these are ordinary arithmetic expressions that happen to carry derivative
information. An overload of ``Source`` additionally taking ``RealVector Scalars``
is available when ``nScalars > 0``, and ``GFunc`` plays the same role for
auxiliary constraints when ``nAux > 0``.

``Flux`` returns :math:`\hat\sigma_i` — the physical flux, exactly what
``SigmaFn`` would return. The sign convention discussed in
:ref:`the formulation <sign-convention>` is applied by the solver, not by you.

Initial conditions come from ``InitialFunction``, in ``Real2nd`` so that
``InitialDerivative`` can be differentiated out of it rather than written twice,
and ``AutodiffTransportSystem`` offers a small menu of standard profiles through
its ``ProfileType`` enum.

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

The ``State`` passed to each carries ``Variable`` (:math:`u`), ``Derivative``
(:math:`q`), ``Flux`` (:math:`\sigma`, negated — see the warning in
:doc:`formulation`), ``Aux`` (:math:`\phi`) and ``Scalars`` (:math:`\mu`), each an
Eigen vector indexed by variable.

Boundary conditions:

.. code-block:: cpp

   Value LowerBoundary(Index i, Time t) const override;
   Value UpperBoundary(Index i, Time t) const override;
   bool isLowerBoundaryDirichlet(Index i) const override;
   bool isUpperBoundaryDirichlet(Index i) const override;

A Dirichlet boundary fixes :math:`u`; a Neumann boundary fixes the flux. The
choice is per variable and per end.

``aFn(i, x)`` supplies the coefficient :math:`a_i` multiplying
:math:`\partial_t u_i`, and defaults to 1.

Auxiliary variables and global scalars
--------------------------------------

Set ``nAux`` in your constructor and implement ``AuxG`` (the constraint
:math:`G_j = 0`) and ``AuxGPrime``, plus ``dSources_dPhi`` and ``dSigma_dPhi``
so the solver knows how the flux and source depend on :math:`\phi`.
``InitialAuxValue`` seeds them.

Set ``nScalars`` and implement ``ScalarG`` (or ``ScalarGExtended`` when the
constraint involves :math:`\dot y`), ``ScalarGPrime``, ``isScalarDifferential``
per scalar, ``InitialScalarValue``, and ``dSources_dScalars`` — which is indexed
by **scalar**, not by variable.

.. note::

   Anything sized per auxiliary variable is sized ``nAux``, not ``nVars``. Those
   two coincide in almost every existing case, which makes a confusion between
   them invisible until it is not.

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

Naming, units and diagnostics
-----------------------------

Optional but worth implementing, since they name things in the output file:
``getVariableName``, ``getVariableDescription``, ``getVariableUnits``, and the
``Scalar``/``Aux`` equivalents. ``initialiseDiagnostics`` and
``writeDiagnostics`` let a case add derived quantities to the netCDF file
alongside the solution.

Registration
------------

Two macros connect a class name to the string used in a config file:

.. code-block:: cpp

   // in the header, inside the class
   REGISTER_PHYSICS_HEADER(MyProblem)

   // in the .cpp, at file scope
   REGISTER_PHYSICS_IMPL(MyProblem)

These declare and define a ``PhysicsCaseRegister<MyProblem>`` whose constructor
inserts a factory into a process-global map during static initialisation. That
mechanism has three consequences worth knowing:

* **A case only appears if its object file is linked in.** Nothing references it
  directly, so a missing entry in the build is a link-line problem that produces
  **no compile error** — just an unrecognised ``TransportSystem`` name at run
  time. Add new files to ``PHYSICS_SOURCES`` in the Makefile.
* **A duplicate name is silently ignored.** Registration uses ``map::insert``, so
  the first registration of a given name wins and later ones are dropped without
  a warning.
* **The map is never reset**, so tests that register throwaway cases must use
  unique names.

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
requires :math:`J\,\delta y = g`.

A wrong **sign** on the flux does give a wrong answer, but a plausible-looking
one that converges at the correct rate to the wrong function. Only a comparison
against a closed-form solution finds it. See :doc:`testing` for both, and
:ref:`the sign convention <sign-convention>` for why.
