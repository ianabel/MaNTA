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
"""

from . import _manta as _core

# Everything the compiled core exposes. Named explicitly rather than by a
# star-import so that `manta.<tab>` is a real list and a typo is an error here
# rather than an AttributeError at run time.
from ._manta import (  # noqa: F401
    AdjointProblem,
    Aux,
    BoundaryKind,
    Dirichlet,
    Field,
    Grid,
    Neumann,
    Runner,
    Scalar,
    SystemSpec,
    TomlValue,
    getNodes,
    numbered_spec,
    registerPhysicsCase,
    run,
)

__all__ = [
    "AdjointProblem",
    "Aux",
    "BoundaryKind",
    "Dirichlet",
    "Field",
    "Grid",
    "Neumann",
    "Runner",
    "Scalar",
    "SystemSpec",
    "TomlValue",
    "TransportSystem",
    "getNodes",
    "numbered_spec",
    "registerPhysicsCase",
    "run",
]


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
    """

    # Overridden by subclasses. Empty rather than absent so that a case which
    # declares only variables does not have to spell out the other two.
    variables = ()
    scalars = ()
    aux = ()

    def __init__(self, spec=None, *, variables=None, scalars=None, aux=None):
        if spec is not None:
            if variables is not None or scalars is not None or aux is not None:
                raise TypeError(
                    "give TransportSystem.__init__ either a spec or the "
                    "variables/scalars/aux lists, not both"
                )
            super().__init__(spec)
            return

        cls = type(self)
        spec = SystemSpec(
            variables=list(self.variables if variables is None else variables),
            scalars=list(self.scalars if scalars is None else scalars),
            aux=list(self.aux if aux is None else aux),
        )

        if not spec.variables:
            raise TypeError(
                f"{cls.__name__} declares no variables. Either set a class-level "
                f"`variables = [manta.Field(...)]`, or pass a spec to "
                f"super().__init__() if the shape depends on the configuration."
            )

        super().__init__(spec)
