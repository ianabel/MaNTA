"""The JAX layer's auxiliary-variable path, one hook at a time.

`test_reference_solutions.py::test_jax_aux_test` runs a whole JAXAuxTest solve
and compares it against a checked-in reference, and that is the real coverage.
But it was `strict=True` xfail for months, and what it reported when it failed
was `RuntimeError: Error occurred when trying to calculate AuxGPrime` from
somewhere inside IDA's first Newton solve -- true, and no help at all in
finding which argument was wrong. Worse, the symptom it was xfailed *on* was a
different one again ("IDA corrector fails to converge at t = 0"), because the
first fault to be hit changes as the surrounding code moves.

Four faults sat behind it, none of which any other JAX fixture could reach,
because `nAux` is zero in all of them:

  * `AuxGPrime(i, out, state, x, t)` and `dAux_dp(i, pIndex, state, x)` carry an
    extra argument ahead of the state, so `MaNTA_Decorator` -- whose wrapper is
    `(self, index, states, positions, *args)` -- converted that argument as
    though it were the state and passed the *state* to `jnp.array()`.
    `ShiftedState_Decorator` is the one they take now.
  * `dgFn_dphi` was not decorated at all, and indexed its result `["Aux"]`, left
    from when a State crossed as a dict.
  * `JAXAdjointProblem.__init__` bound `sigma` and `source` but not `aux`, so
    its `dAux()` raised AttributeError.

These check the two that can be driven directly. `dgFn_dphi` needed a State,
which cannot be constructed from Python, so the solve was its only cover -- and
it has since gone from `manta.jax` entirely, along with `dAux_dp`:
`PyAdjointProblem` raises from both rather than dispatching, because dg/dphi now
arrives with the rest of the batched `dg` and dAux/dp with the batched `dAux`.
The decorator test at the bottom still earns its place, because `AuxGPrime`
remains on `ShiftedState_Decorator`.
"""

import numpy as np
import pytest

pytest.importorskip("equinox")

from typing import NamedTuple  # noqa: E402

import manta as MaNTA  # noqa: E402
from manta.jax import (  # noqa: E402
    JAXAdjointProblem,
    JAXTransportSystem,
    ShiftedState_Decorator,
)
from manta.jax import State as JaxState  # noqa: E402


D = 2.0
NPOINTS = 3
U = np.array([0.5, 1.5, 2.5])
POSITIONS = [0.1, 0.2, 0.3]


class Params(NamedTuple):
    D: float


class JAXAux(JAXTransportSystem):
    """G = a - D u^2, so dG/du = -2 D u, dG/da = 1, dG/dq = 0, dG/dD = -u^2.

    Every one of those is a closed form, which is the point: the hooks below
    are checked against hand-differentiation, not against another autodiff run.
    """

    def __init__(self):
        super().__init__(MaNTA.numbered_spec(1, nAux=1))
        self.params = Params(D=D)

    def sigma(self, index, state, x, t, params):
        return params.D * state.Derivative[0]

    def source(self, index, state, x, t, params):
        return state.Aux[0]

    def aux(self, index, state, x, t, params):
        return state.Aux[0] - params.D * state.Variable[0] ** 2

    def LowerBoundary(self, index, t):
        return 0.0

    def UpperBoundary(self, index, t):
        return 0.0

    def InitialValue(self, index, x):
        return 0.0

    def InitialAuxValue(self, index, x):
        return 0.0

    def g(self, state, x, params):
        return 0.5 * state.Variable[0] ** 2


def global_state():
    return {
        "Variable": U[:, None].copy(),
        "Derivative": np.zeros((NPOINTS, 1)),
        "Flux": np.zeros((NPOINTS, 1)),
        "Aux": np.zeros((NPOINTS, 1)),
        "Scalars": np.zeros(0),
    }


class RecordingSystem(JAXAux):
    """Reads `out` back after the real hook has filled it.

    The out-parameter is a window onto a C++ State, and `AuxGPrime_v`'s own
    output is a GlobalState the type caster copies -- so the mutation never
    reaches Python through the return. Reading it here, inside the call, is
    what the solver itself sees.
    """

    def __init__(self):
        super().__init__()
        self.written = []

    def AuxGPrime(self, index, out, state, x, t):
        super().AuxGPrime(index, out, state, x, t)
        self.written.append(
            {
                "u": np.array(out.u, copy=True),
                "q": np.array(out.q, copy=True),
                "phi": np.array(out.phi, copy=True),
                "x": x,
            }
        )


def test_the_jax_layer_fills_the_auxgprime_out_parameter():
    """The defect the JAXAuxTest xfail was: AuxGPrime never ran at all.

    Driven through the batched entry point, which with no vectorised override
    falls back to the C++ serial loop -- the same loop the solver uses, slicing
    a GlobalState into pointwise States and calling the Python hook per point.
    """
    system = RecordingSystem()
    states = global_state()
    out = {k: np.zeros_like(v) for k, v in states.items()}

    MaNTA.TransportSystem.AuxGPrime_v(system, 0, out, states, POSITIONS, 0.0)

    assert len(system.written) == NPOINTS, (
        f"AuxGPrime ran {len(system.written)} times, expected {NPOINTS}"
    )
    for point, seen in enumerate(system.written):
        assert seen["u"] == pytest.approx([-2.0 * D * U[point]]), (
            f"dG/du at point {point}"
        )
        assert seen["phi"] == pytest.approx([1.0]), f"dG/da at point {point}"
        assert seen["q"] == pytest.approx([0.0]), f"dG/dq at point {point}"


def test_auxgprime_gets_the_state_and_not_the_out_parameter():
    """The specific misalignment, rather than its consequences.

    `out` arrives zeroed and the state does not, so a decorator that swapped
    them would read zeros. Checking a derivative that depends on u -- and only
    at points where u differs -- is what distinguishes the two.
    """
    system = RecordingSystem()
    MaNTA.TransportSystem.AuxGPrime_v(
        system, 0, {k: np.zeros_like(v) for k, v in global_state().items()},
        global_state(), POSITIONS, 0.0,
    )

    dGdu = np.array([seen["u"][0] for seen in system.written])
    assert dGdu == pytest.approx(-2.0 * D * U)
    # Zeros would be the signature of having differentiated the out-parameter.
    assert not np.allclose(dGdu, 0.0)


def test_the_jax_adjoint_differentiates_aux_by_its_parameters():
    """JAXAdjointProblem.dAux() differentiates self.aux, which was never bound.

    __init__ captured the transport system's `sigma` and `source` and not its
    `aux`, so this third branch raised AttributeError. Only ever reached with
    nAux > 0, and only through here: `ComputePhysicsDerivatives` is what
    SystemSolver.cpp:1661 calls, and the aux block is the one it fills from
    dAux(). (The bound `AdjointProblem.dAux` is a different entry point, which
    the solver never takes and which expects the raw GlobalState.)
    """
    system = JAXAux()
    adjoint = JAXAdjointProblem(system, system.g)

    fluxes, sources, aux = adjoint.ComputePhysicsDerivatives(
        global_state(), POSITIONS
    )

    assert len(aux) == system.nAux
    # One parameter, D, and G = a - D u^2, so dG/dD = -u^2 at each point.
    assert np.asarray(aux[0].D).ravel() == pytest.approx(-(U**2))


def test_the_shifted_decorator_converts_the_state_not_the_extra_argument():
    """The misalignment itself, at the decorator rather than through a hook.

    `AuxGPrime` is the one user of ShiftedState_Decorator now -- `dAux_dp` was
    the other, and is gone -- and it cannot be driven from Python: the
    trampoline hands it a State, which has no constructor on this side, and the
    batched wrapper that would loop over it is overridden. A solve is its only
    end-to-end cover, so pin the argument order here, where the shape of a
    state-one-argument-later hook is the thing under test rather than any one
    hook that has it.
    """
    seen = {}

    class Hook:
        @ShiftedState_Decorator
        def hook(self, index, extra, state, x, t):
            seen.update(index=index, extra=extra, state=state, x=x, t=t)
            return "the return value"

    sentinel = object()
    returned = Hook().hook(3, sentinel, global_state(), POSITIONS, 1.5)

    assert seen["extra"] is sentinel, "the extra argument must pass through untouched"
    assert isinstance(seen["state"], JaxState), (
        f"the state must be converted, got {type(seen['state'])}"
    )
    assert np.asarray(seen["state"].Variable).ravel() == pytest.approx(U)
    assert np.asarray(seen["x"]) == pytest.approx(POSITIONS)
    assert seen["index"] == 3
    assert seen["t"] == 1.5
    assert returned == "the return value", "dAux_dp returns a value; it must survive"
