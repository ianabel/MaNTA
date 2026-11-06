"""MaNTA -- Maryland Nonlinear Transport Analyzer.

A physics case is a subclass of :class:`manta.TransportSystem` that declares
what it is as data and implements the flux and the source:

    import manta

    class Diffusion(manta.TransportSystem):
        variables = [manta.Field("n", "density", "m^-3",
                                 lower=manta.Neumann, upper=manta.Dirichlet)]

        def __init__(self, config, grid):
            super().__init__()
            self.kappa = config["kappa"]

        def SigmaFn(self, i, state, x, t):
            return self.kappa * state.q[0]

        def Sources(self, i, state, x, t):
            return 0.0

    manta.registerPhysicsCase("Diffusion", Diffusion)

The case and the driver can live in your own repository; nothing needs to be
inside the MaNTA source tree.

A physics case written in C++ is driven the same way, named rather than
constructed -- :func:`physics_cases` lists what this build carries, and the
case's own configuration table goes in the same dict as the solver's keys:

    runner = manta.Runner("LinearDiffusion")
    runner.configure({
        "Polynomial_degree": 3, "Grid_size": 20,
        "Lower_boundary": 0.0, "Upper_boundary": 1.0,
        "delta_t": 0.05, "OutputFilename": "run",
        "DiffusionProblem": {"Kappa": 2.0},          # the case's own table
    })
    runner.run(1.0)

Use :func:`load_physics_plugin` first for a C++ case built out of tree.
"""

from . import _manta as _core

# Everything the compiled core exposes. Named explicitly rather than by a
# star-import so that `manta.<tab>` is a real list and a typo is an error here
# rather than an AttributeError at run time.
from ._manta import (  # noqa: F401
    AdjointProblem,
    Aux,
    BoundaryCondition,
    BoundaryKind,
    Dirichlet,
    EvaluationCadence,
    EvaluationEntry,
    EvaluationKind,
    EvaluationPlan,
    EvaluationSite,
    EquilibriumResponse,
    EquilibriumSpec,
    Field,
    GeometryDerivatives,
    GeometrySlot,
    Grid,
    Mixed,
    Neumann,
    Regrid,
    Runner,
    Scalar,
    SteadyOutcome,
    SystemSpec,
    TomlValue,
    getNodes,
    getPhysicsNodes,
    getPhysicsWeights,
    load_physics_plugin,
    numbered_spec,
    physics_cases,
    registerEquilibriumModel,
    registerPhysicsCase,
    run,
)

__all__ = [
    "AdjointProblem",
    "Aux",
    "BoundaryCondition",
    "BoundaryKind",
    "Dirichlet",
    "EvaluationCadence",
    "EvaluationEntry",
    "EvaluationKind",
    "EvaluationPlan",
    "EvaluationSite",
    "EquilibriumModel",
    "EquilibriumResponse",
    "EquilibriumSpec",
    "Field",
    "GeometryDerivatives",
    "GeometrySlot",
    "Grid",
    "Mixed",
    "Neumann",
    "Regrid",
    "Runner",
    "Scalar",
    "SteadyOutcome",
    "SteadySolve",
    "SystemSpec",
    "TomlValue",
    "TransportSystem",
    "getNodes",
    "getPhysicsNodes",
    "getPhysicsWeights",
    "load_physics_plugin",
    "numbered_spec",
    "physics_cases",
    "registerEquilibriumModel",
    "registerPhysicsCase",
    "run",
]


class SteadySolve:
    """A steady solve driven in slices, so a driver can look between them.

    Each slice runs at most ``MaxContinuationSteps`` continuation steps and then
    hands back. Resuming carries the state *and* the pseudo-time step the last
    slice reached, so slicing costs no extra continuation steps -- a solve that
    takes sixteen uninterrupted takes sixteen in slices of three.

    Iterating yields ``(outcome, stats)`` per slice and stops of its own accord
    once a slice returns anything but :attr:`SteadyOutcome.OutOfSteps`::

        with manta.SteadySolve(runner) as solve:
            for outcome, stats in solve:
                if stats["residual_norm"] < 1e-6:
                    solve.stop()

    Leaving the block writes the output, the restart file and the adjoint solve,
    then tears the solve down. Leaving it by an exception -- or by calling
    :meth:`abandon` -- tears it down and writes nothing. Either way the SUNDIALS
    objects are freed, which is the reason to use the context manager rather
    than the Runner methods directly: nothing else frees them.

    Parameters
    ----------
    runner:
        A configured :class:`Runner`. ``DegreeAdaptation`` is refused -- see the
        note in :meth:`Runner.start_steady`.
    estimate:
        Whether each slice estimates the objective on its way out. That costs a
        residual, a Jacobian build and a solve *per slice* and needs
        ``solveAdjoint``, so a driver reading the estimate only at the end
        should pass ``False`` and let :meth:`finish` produce the one that counts.
    """

    def __init__(self, runner, estimate=True):
        self._runner = runner
        self._estimate = estimate
        self._started = False
        self._closed = False
        self._stop = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._close(write=exc_type is None)
        return False

    def __iter__(self):
        while not self._stop:
            if self._started:
                outcome = self._runner.continue_steady(self._estimate)
            else:
                outcome = self._runner.start_steady(self._estimate)
                self._started = True
            yield outcome, self._runner.steadyStats()
            if outcome != SteadyOutcome.OutOfSteps:
                return

    def stop(self):
        """Stop after the current slice, and still write the result."""
        self._stop = True

    def abandon(self):
        """Stop after the current slice and write nothing."""
        self._stop = True
        self._close(write=False)

    def _close(self, write):
        if self._closed or not self._started:
            self._closed = True
            return
        self._closed = True
        if write:
            self._runner.finish_steady()
        else:
            self._runner.abandon_steady()


class TransportSystem(_core.TransportSystem):
    """Base class for a physics case.

    The compiled core requires a :class:`SystemSpec` at construction -- the
    counts and boundary kinds have to be known before the object exists, which
    is what stops a case being half-described. This layer lets that spec be
    written as class attributes, which is where a reader looks for it:

        class MyCase(manta.TransportSystem):
            variables = [manta.Field("n", lower=manta.Neumann)]
            aux = [manta.Aux("phi")]

            def __init__(self, config, grid):
                super().__init__()

    ``super().__init__()`` with no arguments reads those attributes. Passing a
    spec explicitly still works, for a case whose shape depends on its
    configuration and so cannot be written down at class scope:

        super().__init__(manta.numbered_spec(nVars, nAux=1))
        super().__init__(variables=[...], scalars=[...], aux=[...])

    ``prepareEvaluation(plan)``, optional, is told where the solver will
    evaluate the case -- an :class:`EvaluationPlan` of every point set, batch
    size and cadence -- before the first physics call of any run whose plan is
    new to this instance. A *changed* plan, from a new mesh or degree, reaches a
    case only if it declares ``regrid = manta.Regrid.InPlace``, saying it
    rebuilds in ``prepareEvaluation`` whatever depended on the old plan (or that
    nothing did). Without it the case is evaluated by its first plan only, and
    reusing it elsewhere -- adapting the mesh or degree, or reconfiguring a
    Runner onto another grid -- is refused. ``regrid`` is honoured on both paths
    above; an explicit spec need not repeat it.

    A case that reads a field's geometry through ``state.geom`` declares
    ``readsGeometry = True``, and says where the coupled Jacobian gets its
    derivatives in the geometry with ``geometryDerivatives``:
    ``GeometryDerivatives.Analytic`` (its ``dSigmaFn_dGeometry`` and friends),
    ``Differenced`` (the solver differences the values) or ``None_``, the
    default, which runs a field as a black box. Both are honoured like
    ``regrid``.
    """

    # Overridden by subclasses. Empty rather than absent so that a case which
    # declares only variables does not have to spell out the other two.
    variables = ()
    scalars = ()
    aux = ()
    regrid = Regrid.Fixed
    readsGeometry = False
    geometryDerivatives = GeometryDerivatives.None_

    def __init__(self, spec=None, *, variables=None, scalars=None, aux=None):
        cls = type(self)
        if spec is not None:
            if variables is not None or scalars is not None or aux is not None:
                raise TypeError(
                    "give TransportSystem.__init__ either a spec or the "
                    "variables/scalars/aux lists, not both"
                )
            # A copy rather than an edit, so the caller's spec is left as given.
            # What the class declares fills in only what the spec left at its
            # default.
            regrid = spec.regrid if spec.regrid != Regrid.Fixed else cls.regrid
            reads = spec.readsGeometry or cls.readsGeometry
            derivs = (
                spec.geometryDerivatives
                if spec.geometryDerivatives != GeometryDerivatives.None_
                else cls.geometryDerivatives
            )
            if (
                regrid != spec.regrid
                or reads != spec.readsGeometry
                or derivs != spec.geometryDerivatives
            ):
                spec = SystemSpec(
                    variables=list(spec.variables),
                    scalars=list(spec.scalars),
                    aux=list(spec.aux),
                    regrid=regrid,
                    readsGeometry=reads,
                    geometryDerivatives=derivs,
                )
            super().__init__(spec)
            return

        spec = SystemSpec(
            variables=list(self.variables if variables is None else variables),
            scalars=list(self.scalars if scalars is None else scalars),
            aux=list(self.aux if aux is None else aux),
            regrid=cls.regrid,
            readsGeometry=cls.readsGeometry,
            geometryDerivatives=cls.geometryDerivatives,
        )

        if not spec.variables:
            raise TypeError(
                f"{cls.__name__} declares no variables. Either set a class-level "
                f"`variables = [manta.Field(...)]`, or pass a spec to "
                f"super().__init__() if the shape depends on the configuration."
            )

        super().__init__(spec)


class EquilibriumModel(_core.EquilibriumModel):
    """Base class for a magnetic equilibrium written in Python.

    A black box from the transport's pressure to the geometry -- tier 0 of the
    field coupling (``FieldCoupling = "consistent"``), and tier 1
    (``"reduced"``) with its response, differenced by default. Declare the
    geometry slots as data and implement two methods::

        class Balance(manta.EquilibriumModel):
            geometry = [manta.GeometrySlot("Vprime"), manta.GeometrySlot("B")]
            label = "x"           # the coordinate the slots are a function of
            name = "Balance"      # the netCDF group they are written into

            def Solve(self, states, points, weights, t, warm):
                ...               # any object: the solver only hands it back
            def Geometry(self, equilibrium, points):
                ...               # (len(geometry), len(points)) array

        runner.set_equilibrium_model(Balance())

    ``states`` is the transport state at the basis nodes ``points``, as the
    dict a vectorised case is handed -- ``states["Aux"]`` is ``(nPoints,
    nAux)`` -- and ``weights`` integrates a nodal function over the domain.
    ``warm`` is this model's last answer in the run, or None. Raise if the
    solve fails: the integrator cuts its step, as for a physics hook.

    Optional: ``applyResponse(equilibrium, states, dp, inputPoints, weights,
    t, geometryPoints)`` returning ``dg`` and ``applyResponseTranspose(...,
    dg, ...)`` returning a dict of ``dp`` rows, with ``response =
    manta.EquilibriumResponse.Model``; ``responseStep()``, the relative step of
    the differenced response; ``setTolerances(relative, absolute)``;
    ``resetForRun()``. ``docs/field_coupling.rst`` has the contract.
    """

    geometry = ()
    label = ""
    name = "Field"
    response = EquilibriumResponse.Differenced

    def __init__(self, spec=None):
        if spec is None:
            cls = type(self)
            if not cls.geometry:
                raise TypeError(
                    f"{cls.__name__} declares no geometry slots. Set a class-level "
                    f"`geometry = [manta.GeometrySlot(...)]`, or pass an "
                    f"EquilibriumSpec to super().__init__()."
                )
            spec = EquilibriumSpec(
                geometry=list(cls.geometry),
                label=cls.label,
                name=cls.name,
                response=cls.response,
            )
        super().__init__(spec)
