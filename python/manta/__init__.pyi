"""Type stubs for the `manta` package.

The compiled core's stub, `_manta.pyi`, is generated from the extension by
`make stubs` and must not be edited. This file is the hand-written half: it
covers the Python layer in __init__.py, which stubgen cannot see.

What it buys, beyond editor completion on an otherwise opaque extension: a
subclass that gets a hook's signature wrong is a type error rather than a
RuntimeError on the first Jacobian evaluation, and `State` shows up as
something other than an array -- which matters, because it is a view of solver
memory that dies when the hook returns.

Run it over your own physics case with mypy.ini's settings; `check_untyped_defs`
is the one that matters, because a physics case is ordinary unannotated Python
and mypy skips unannotated defs without it.

One wart: a wrong hook signature is reported twice, once against
`manta.TransportSystem` (this file, and the one to read) and once against
`manta._manta.TransportSystem` (the C++ binding, whose signature carries
out-parameters you never write). The second is noise; the first says what to fix.
"""

from collections.abc import Sequence
from typing import Any

from ._manta import (
    AdjointProblem as AdjointProblem,
    Aux as Aux,
    BoundaryCondition as BoundaryCondition,
    BoundaryKind as BoundaryKind,
    Field as Field,
    Grid as Grid,
    Mixed as Mixed,
    Runner as Runner,
    Scalar as Scalar,
    State as State,
    StateField as StateField,
    SystemSpec as SystemSpec,
    TomlValue as TomlValue,
    getNodes as getNodes,
    numbered_spec as numbered_spec,
    registerPhysicsCase as registerPhysicsCase,
    run as run,
)
from ._manta import TransportSystem as _TransportSystem

Dirichlet: BoundaryKind
Neumann: BoundaryKind

__all__: list[str]

class TransportSystem(_TransportSystem):
    """Base class for a physics case.

    A subclass declares what it is with the three class attributes and
    implements the hooks it needs. Only SigmaFn and Sources are required; an
    absent derivative hook means that block is identically zero.
    """

    variables: Sequence[Field]
    scalars: Sequence[Scalar]
    aux: Sequence[Aux]

    def __init__(
        self,
        spec: SystemSpec | None = ...,
        *,
        variables: Sequence[Field] | None = ...,
        scalars: Sequence[Scalar] | None = ...,
        aux: Sequence[Aux] | None = ...,
    ) -> None: ...

    # The hooks a subclass overrides, declared so that a wrong signature in a
    # physics case is caught statically rather than on the first residual
    # evaluation.
    #
    # `state` is a view of solver memory, valid only for the duration of the
    # call -- copy anything you need to keep (np.array(s.u, copy=True)).
    #
    # Every one carries `type: ignore[override]`, and that is not a workaround:
    # what a case implements is genuinely not what the bound base method's
    # signature says, in two ways.
    #
    #   * The derivative hooks take a C++ out-parameter. `dSigmaFn_du` is
    #     `(Index, VectorRef, const State &, Position, Time)` in C++, and the
    #     binding shows that; a Python case writes `(i, state, x, t)` and
    #     *returns* the vector, which the trampoline copies into the out
    #     parameter.
    #
    #   * pybind11 widens numerics, so the base takes
    #     `SupportsInt | SupportsIndex` where a case writes `int`. Narrowing a
    #     parameter in an override is a Liskov violation, and mypy is right to
    #     say so -- but `int` is what an author should read here.
    #
    # These declarations are what a *user's* subclass is checked against, which
    # is the point. warn_unused_ignores is on, so if the asymmetry ever goes
    # away mypy will say the ignore is no longer needed.
    def SigmaFn(self, i: int, state: State, x: float, t: float) -> float: ...  # type: ignore[override]
    def Sources(self, i: int, state: State, x: float, t: float) -> float: ...  # type: ignore[override]
    def dSigmaFn_du(self, i: int, state: State, x: float, t: float) -> Any: ...  # type: ignore[override]
    def dSigmaFn_dq(self, i: int, state: State, x: float, t: float) -> Any: ...  # type: ignore[override]
    def dSources_du(self, i: int, state: State, x: float, t: float) -> Any: ...  # type: ignore[override]
    def dSources_dq(self, i: int, state: State, x: float, t: float) -> Any: ...  # type: ignore[override]
    def dSources_dsigma(self, i: int, state: State, x: float, t: float) -> Any: ...  # type: ignore[override]

    # Derivatives with respect to a field model's geometry slots (state.geom).
    # Optional, like the five above: an absent hook is an identically zero
    # coupling block, which is the correct answer for a case that does not
    # read geometry at all.
    def dSigmaFn_dGeometry(self, i: int, state: State, x: float, t: float) -> Any: ...  # type: ignore[override]
    def dSources_dGeometry(self, i: int, state: State, x: float, t: float) -> Any: ...  # type: ignore[override]
    def aFn(self, i: int, x: float) -> float: ...  # type: ignore[override]
    def LowerBoundary(self, i: int, t: float) -> float: ...  # type: ignore[override]
    def UpperBoundary(self, i: int, t: float) -> float: ...  # type: ignore[override]
    def InitialValue(self, i: int, x: float) -> float: ...  # type: ignore[override]
    def InitialDerivative(self, i: int, x: float) -> float: ...  # type: ignore[override]

    # Auxiliary variables. AuxGPrime fills its out-parameter rather than
    # returning: `out` is a view that arrives zeroed.
    def InitialAuxValue(self, i: int, x: float) -> float: ...  # type: ignore[override]
    def AuxG(self, i: int, state: State, x: float, t: float) -> float: ...  # type: ignore[override]
    def AuxGPrime(self, i: int, out: State, state: State, x: float, t: float) -> None: ...  # type: ignore[override]
    def dSources_dPhi(self, i: int, state: State, x: float, t: float) -> Any: ...  # type: ignore[override]
    def dSigma_dPhi(self, i: int, state: State, x: float, t: float) -> Any: ...  # type: ignore[override]
    def dAuxG_dGeometry(self, i: int, state: State, x: float, t: float) -> Any: ...  # type: ignore[override]

    # Global scalars. These take the solution sampled on the element nodes, as
    # dicts of (nPoints, nVars) arrays, plus the quadrature data.
    def InitialScalarValue(self, s: int) -> float: ...  # type: ignore[override]
    def InitialScalarDerivative(self, s: int, states: Any, states_dot: Any, weights: Any) -> float: ...
    def ScalarG(self, s: int, states: Any, states_dot: Any, abscissae: Any, weights: Any, phi_boundary: Any, t: float) -> float: ...  # type: ignore[override]
    def ScalarGPrime(self, states: Any, states_dot: Any, abscissae: Any, weights: Any, phi_boundary: Any, t: float) -> tuple[list[Any], list[Any]]: ...
    def dSources_dScalars(self, s: int, state: State, x: float, t: float) -> Any: ...  # type: ignore[override]
