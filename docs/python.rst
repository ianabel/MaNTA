The Python interface
====================

``make python`` builds the ``manta`` package -- a pybind11 extension,
``python/manta/_manta<suffix>.so``, wrapped by a thin Python layer. It does two
separate jobs: it drives the solver from Python, and it lets a transport system
be *written* in Python.

.. code-block:: sh

   make python
   pip install .                     # then `import manta` works anywhere
   python -c "import manta; print(manta.__doc__)"

A case and its driver can live in your own repository rather than in this tree;
see :doc:`out_of_tree`.

Module contents
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - What it is
   * - ``manta.run(path)``
     - Run a TOML config file, exactly as the ``MaNTA`` binary would.
   * - ``manta.Runner``
     - The configure/run/inspect interface. See below.
   * - ``manta.TransportSystem``
     - Base class to subclass when writing physics in Python.
   * - ``manta.AdjointProblem``
     - Base class for an objective and its parameter derivatives.
   * - ``manta.registerPhysicsCase(name, factory)``
     - Register a Python case under a name the TOML interface can use.
   * - ``manta.getNodes(...)``
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
     - Builds a **fresh** solver and applies the settings. The keys are the ones
       a config file uses — both read one schema; see :ref:`config-divergences`
       for the two deliberate exceptions.
   * - ``run(tFinal)``
     - Integrates to ``tFinal``.
   * - ``run()``
     - Integrates to the configuration's ``t_final``; an error if it had none.
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
   * - ``wasRejected()``
     - Whether the last run was abandoned by the ``ObjectiveDecreaseTolerance``
       gate instead of integrated. Always ``False`` when that is not configured.
   * - ``lastDGdt()``
     - The :math:`\mathrm{d}G/\mathrm{d}t` values behind that decision, one per
       objective.

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

Subclass ``manta.TransportSystem`` and override the hooks you need — the same
names as the C++ interface described in :doc:`physics_interface`:

.. code-block:: python

   import manta
   import numpy as np

   class Diffusion(manta.TransportSystem):
       variables = [manta.Field("n", "density", "m^-3")]

       def __init__(self, kappa):
           super().__init__()          # reads the class attributes above
           self.kappa = kappa

       def LowerBoundary(self, i, t): return 0.0
       def UpperBoundary(self, i, t): return 0.0

       def SigmaFn(self, i, state, x, t):
           return self.kappa * state["Derivative"][i]

       def Sources(self, i, state, x, t):
           return 0.0

       def dSigmaFn_dq(self, i, state, x, t):
           return np.full(self.nVars, self.kappa)

       # dSigmaFn_du and the three dSources_* hooks are identically zero here,
       # so they are simply left out.

       def InitialValue(self, i, x):     return np.sin(np.pi * x)
       def InitialDerivative(self, i, x): return np.pi * np.cos(np.pi * x)

Declaring the case as data replaces assigning ``self.nVars`` and
``self.isUpperDirichlet`` after construction. Those could not be validated --
the object already existed by the time they were set -- and the boundary flags
were read uninitialised if a case forgot them. ``nVars``, ``nScalars`` and
``nAux`` are read-only properties derived from the declaration.

``manta.Field(name, description, units, lower=..., upper=...)`` takes
``manta.Dirichlet`` (the default) or ``manta.Neumann`` per end;
``manta.Scalar(name, ..., differential=True)`` marks a scalar whose constraint
involves the time derivative; ``manta.Aux(name, ...)`` declares an auxiliary
variable. A case whose shape depends on its configuration cannot write that at
class scope and passes a spec instead:

.. code-block:: python

   manta.TransportSystem.__init__(self, manta.numbered_spec(nVars, nAux=1))

``SigmaFn`` and ``Sources`` are required, and so are ``LowerBoundary`` and
``UpperBoundary`` — this said "only ``SigmaFn`` and ``Sources``" until a case
that took it literally was tried. A case that omits the boundary hooks has
nowhere to get its boundary values from and is rejected when the solver first
asks for one, naming the hook it wants. (The base-class defaults exist for two
narrower situations: a restart, which recovers the values from the stored
profile, and ``AutodiffTransportSystem``, which reads them from its own
``uL``/``uR`` config keys.)

**The derivative hooks are optional**: an absent one means that block is
identically zero, which is what the framework's zeroed output buffer already
provides. Before this, the simplest possible case had to write four functions
returning ``np.zeros(nVars)`` before it would run.

The subclass is inspected **once**, when it is first used, and classified as
either *pointwise* or *vectorised*. The vectorised path requires **both**
``ComputePhysics`` and ``ComputePhysicsDerivatives`` to be overridden; supplying
one without the other is rejected rather than silently half-used. The same probe
enforces the extra hooks that become mandatory when ``nScalars > 0`` or
``nAux > 0``.

State and GlobalState
---------------------

A pointwise ``State`` is an object with named fields:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Meaning
   * - ``s.u``
     - the variables
   * - ``s.q``
     - :math:`\partial_x u`
   * - ``s.sigma``
     - the **stored** flux, :math:`-\hat\sigma`
   * - ``s.sigmaHat``
     - the **physical** flux, what ``SigmaFn`` returned (read-only)
   * - ``s.phi``
     - the auxiliary variables
   * - ``s.scalars``
     - the global scalars

Each is indexable by position or by the name the case declared, and converts to
a numpy array:

.. code-block:: python

   def SigmaFn(self, i, s, x, t):
       return self.kappa * s.q["density"]     # or s.q[0]

This replaced a dict of five 1-D arrays keyed ``"Variable"``, ``"Derivative"``,
``"Flux"``, ``"Aux"``, ``"Scalars"``, so the idiom was
``state["Derivative"][0]``. That named storage rather than meaning, gave no hint
that ``"Flux"`` held the *negated* flux, turned a mistyped key into a KeyError
deep inside the first residual evaluation, and copied all five vectors on every
call — once per point, per hook.

.. warning::

   A ``State`` is a **view of solver memory**, valid only inside the call it was
   passed to. To keep values past the hook's return, copy them:
   ``np.array(s.u, copy=True)``. Keeping the view itself, or a ``np.asarray``
   view of one of its fields, reads freed memory.

   There is deliberately no way to construct one from Python.

``GlobalState`` — the batched form — is still a dict of ``(nPoints, nVars)``
arrays keyed the old way. That is what the vectorised and JAX paths actually
want, and the keys stay because a 2-D array of every point is a different
object from a view of one point.

.. warning::

   The ``GlobalState`` caster **transposes in both directions** — C++ stores
   ``(nVars, nPoints)`` and Python sees ``(nPoints, nVars)``. A round-trip
   therefore cannot detect a missing transpose; to check orientation, look at the
   array shape from inside a batched call.

Type stubs
----------

The package ships ``py.typed`` and two stub files, so an editor completes
``manta.`` and a type checker can see the hook signatures:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Origin
   * - ``manta/_manta.pyi``
     - **Generated** from the built extension by ``make stubs``. Do not edit --
       ``make stubs-check`` fails if it no longer matches what the module
       exposes, and CI runs that.
   * - ``manta/__init__.pyi``
     - Hand-written, for the Python layer stubgen cannot see: the class-level
       spec attributes and the hook signatures a case implements.

The point of the second file is that a physics case with a wrong hook signature
becomes a type error rather than a ``RuntimeError`` on the first residual
evaluation:

.. code-block:: sh

   mypy --check-untyped-defs mycase.py

``--check-untyped-defs`` is not optional here. A physics case is ordinary
unannotated Python, and mypy skips unannotated definitions without it -- which
would make the hook declarations decorative. ``mypy.ini`` in the repository has
it on and is a reasonable starting point for your own.

.. note::

   A wrong hook signature is reported **twice**: once against
   ``manta.TransportSystem``, which is the message to read, and once against
   ``manta._manta.TransportSystem``, whose signature is the C++ one and carries
   out-parameters you never write. The duplicate is unavoidable while both
   classes are in the MRO.

Scalar hooks
------------

A subclass that declares scalars implements:

.. code-block:: python

   InitialScalarValue(s)                                    -> float
   InitialScalarDerivative(s, states, states_dot, weights)  -> float
   ScalarG(s, states, states_dot, abscissae, weights, phi_boundary, t) -> float
   ScalarGPrime(states, states_dot, abscissae, weights, phi_boundary, t)
       -> (list of nScalars GlobalState dicts,   # d G_s / d state
           list of nScalars GlobalState dicts)   # d G_s / d state_dot
   dSources_dScalars(s, state, x, t)  -> vector of length nScalars

These are now the **same** signatures as the C++ hooks, argument for argument.
They used to differ, because the C++ side took ``DGSoln``, a ``std::function``
test function and an ``Interval``, none of which have a Python representation --
so the trampoline was a translation layer between two interfaces that had to be
kept in step by hand. The C++ side has adopted this one.

``weights`` is one quadrature weight per node -- length ``nCells * (k+1)`` -- so
an integral over the domain is simply ``weights @ u``, and the derivative of
that integral with respect to node ``j`` is ``weights[j]``. ``phi_boundary`` is
``(k+1, 2)``, the basis functions of the first and last cells evaluated at the
two ends of the domain; because the nodes are strictly interior, this is the
only way to express a constraint on a boundary point value. ``abscissae`` gives
each node's position.

Whether a scalar is differential is part of the declaration --
``manta.Scalar("mu", differential=True)`` -- rather than an
``isScalarDifferential`` hook. ``InitialScalarDerivative`` is consulted only for
scalars declared that way.

Note that ``dSources_dScalars`` is indexed by **scalar**, not by variable.

JAX
---

``manta.jax`` wraps the dict interface in
`equinox <https://github.com/patrick-kidger/equinox>`_ modules, so a physics
case can be written as JAX functions and have its derivatives supplied by
``jax.grad`` rather than by hand. The adapters ``MaNTA_Decorator`` and
``Physics_Decorator`` handle the conversion. Such a case is a *vectorised*
subclass in the sense above: it overrides ``ComputePhysics`` and
``ComputePhysicsDerivatives`` and is called once per batch.

It is the only part of the package that needs JAX, so it is an optional extra
rather than a dependency — ``import manta`` stays numpy-only::

   pip install manta[jax]

``manta.jax.JAXTransportSystem`` is the pointwise base class and
``manta.jax.VectorizedTransportSystem`` the batched one;
``manta.jax.JAXAdjointProblem`` supplies the objective and its parameter
derivatives. Worked examples are in ``python-examples/jax-diffusion``,
``python-examples/jax-linear-diffusion`` and
``python-examples/jax-nonlinear-adjoint``.

An ``XLA_FFI`` build additionally exposes the solver itself as a JAX foreign
function, so a whole MaNTA run can sit inside a JAX computation. That path needs
jaxlib headers at build time, and is reached through ``manta.jax.FFIRunner``.
That one name is imported on demand rather than with the rest of the layer,
because the bindings it registers exist only in such a build; on any other it
raises an ``ImportError`` naming the flag, leaving the rest of ``manta.jax``
usable. ``python-examples/adjoints/jvp.py`` is the worked example.
