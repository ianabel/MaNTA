The Python interface
====================

``make python`` builds a pybind11 extension, ``python/MaNTA<suffix>.so``, which
does two separate jobs: it drives the solver from Python, and it lets a transport
system be *written* in Python.

.. code-block:: sh

   make python
   export PATH="$PWD/.venv/bin:$PATH"
   cd python && python -c "import MaNTA; print(MaNTA.__doc__)"

Module contents
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - What it is
   * - ``MaNTA.run(path)``
     - Run a TOML config file, exactly as the ``MaNTA`` binary would.
   * - ``MaNTA.Runner``
     - The configure/run/inspect interface. See below.
   * - ``MaNTA.TransportSystem``
     - Base class to subclass when writing physics in Python.
   * - ``MaNTA.AdjointProblem``
     - Base class for an objective and its parameter derivatives.
   * - ``MaNTA.registerPhysicsCase(name, factory)``
     - Register a Python case under a name the TOML interface can use.
   * - ``MaNTA.getNodes(...)``
     - The positions at which fluxes and sources are evaluated, for a given grid
       and polynomial degree.

``Runner``
----------

.. code-block:: python

   import MaNTA

   problem = MyTransportSystem()           # subclass of MaNTA.TransportSystem
   runner = MaNTA.Runner()
   runner.configure(problem, {
       "Polynomial_degree": 3,
       "Grid_size": 30,
       "Lower_boundary": 0.0,
       "Upper_boundary": 1.0,
       "delta_t": 0.1,
       "OutputFilename": "run1",
   })
   runner.run(1.0)                          # integrate to t = 1
   sol = runner.getSolution()

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Method
     - Does
   * - ``configure(problem, dict)``
     - Builds a **fresh** solver and applies the settings. The parameter table is
       in :ref:`config-divergences` — it is not identical to the TOML keys.
   * - ``run(tFinal)``
     - Integrates to ``tFinal``.
   * - ``run_ss()``
     - Integrates until :math:`\mathrm{d}y/\mathrm{d}t` falls below
       ``SteadyStateTolerance``.
   * - ``getSolution()``
     - The solution fields.
   * - ``getPostprocessedSolution()``
     - The same, with :math:`u^\star` in place of :math:`u`.
   * - ``G()``
     - The objective value, without gradients.
   * - ``getAdjointGradients()``
     - :math:`\mathrm{d}G/\mathrm{d}p`.

``configure`` building a fresh solver each time is load-bearing rather than
incidental: it is what makes ``Runner`` the only route that supports repeated
configure/run cycles in one process, and it sidesteps the second-integration
failure described in :doc:`running`. An optimisation driver that loops over
``configure`` → ``run`` → ``getAdjointGradients`` is on the supported path.

.. note::

   ``G()`` returns the objective without the gradient, but the saving is in the
   *run*, not in ``G`` itself. With ``solveAdjoint = True`` the gradients are
   already computed by the time ``run`` returns, and ``getAdjointGradients`` only
   reads them back. To actually skip the adjoint solve, configure with
   ``solveAdjoint = False``; ``G`` then builds an adjoint problem on demand purely
   to evaluate the objective.

Writing physics in Python
-------------------------

Subclass ``MaNTA.TransportSystem`` and override the hooks you need — the same
names as the C++ interface described in :doc:`physics_interface`:

.. code-block:: python

   import MaNTA
   import numpy as np

   class Diffusion(MaNTA.TransportSystem):
       def __init__(self, kappa):
           super().__init__()
           self.nVars = 1
           self.kappa = kappa
           self.isLowerDirichlet = True
           self.isUpperDirichlet = True

       def LowerBoundary(self, i, t): return 0.0
       def UpperBoundary(self, i, t): return 0.0

       def SigmaFn(self, i, state, x, t):
           return self.kappa * state["Derivative"][i]

       def Sources(self, i, state, x, t):
           return 0.0

       def dSigmaFn_dq(self, i, out, state, x, t):
           out[i] = self.kappa

       # ... the remaining derivative hooks ...

       def InitialValue(self, i, x):     return np.sin(np.pi * x)
       def InitialDerivative(self, i, x): return np.pi * np.cos(np.pi * x)

The subclass is inspected **once**, when it is first used, and classified as
either *pointwise* or *vectorised*. The vectorised path requires **both**
``ComputePhysics`` and ``ComputePhysicsDerivatives`` to be overridden; supplying
one without the other is rejected rather than silently half-used. The same probe
enforces the extra hooks that become mandatory when ``nScalars > 0`` or
``nAux > 0``.

State and GlobalState
---------------------

The two C++ state types cross the boundary as dictionaries:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - C++ type
     - Python form
   * - ``State``
     - A dict of 1-D arrays: ``"Variable"``, ``"Derivative"``, ``"Flux"``,
       ``"Aux"``, ``"Scalars"``. Each is indexed by variable.
   * - ``GlobalState``
     - A dict of 2-D arrays, each ``(nPoints, nVars)``.

.. warning::

   The ``GlobalState`` caster **transposes in both directions** — C++ stores
   ``(nVars, nPoints)`` and Python sees ``(nPoints, nVars)``. A round-trip test
   therefore cannot detect a missing transpose; to check orientation, look at the
   array shape from inside a batched call.

Scalar hooks: the signatures differ from C++
--------------------------------------------

The C++ scalar hooks take ``DGSoln`` and ``Interval``, which have no Python
representation. The trampoline evaluates on the nodes and passes a
``GlobalState`` plus the quadrature data instead, so a Python subclass with
``nScalars > 0`` must implement these — **not** the C++ signatures:

.. code-block:: python

   InitialScalarValue(s)                                    -> float
   InitialScalarDerivative(s, states, states_dot, weights)  -> float
   isScalarDifferential(i)                                  -> bool
   ScalarG(s, states, states_dot, weights, t)               -> float
   ScalarGPrime(states, states_dot, weights, phi_boundary, t)
       -> (list of nScalars GlobalState dicts,   # d G_s / d state
           list of nScalars GlobalState dicts)   # d G_s / d state_dot
   dSources_dScalars(s, state, x, t)  -> vector of length nScalars

``weights`` is one quadrature weight per node — length ``nCells * (k+1)`` — so an
integral over the domain is simply ``weights @ u``. ``phi_boundary`` is
``(k+1, 2)``, the basis functions evaluated at the two ends of the domain.

Note that ``dSources_dScalars`` is indexed by **scalar**, not by variable, and
that ``InitialScalarDerivative`` is only consulted for scalars where
``isScalarDifferential`` returns true.

JAX
---

``python/JAXTransportSystem.py`` and ``python/State.py`` wrap the dict interface
in `equinox <https://github.com/patrick-kidger/equinox>`_ modules, so a physics
case can be written as JAX functions and have its derivatives supplied by
``jax.grad`` rather than by hand. The adapters ``MaNTA_Decorator`` and
``Physics_Decorator`` handle the conversion. Such a case is a *vectorised*
subclass in the sense above: it overrides ``ComputePhysics`` and
``ComputePhysicsDerivatives`` and is called once per batch.

An ``XLA_FFI`` build additionally exposes the solver itself as a JAX foreign
function, so a whole MaNTA run can sit inside a JAX computation. That path needs
jaxlib headers at build time.
