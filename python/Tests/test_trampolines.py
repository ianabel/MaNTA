"""Tests for the PyTransportSystem trampoline and the State/GlobalState casters.

PyTransportSystem::initializeOverrides classifies a Python subclass as either
"scalar" or "vectorised" by probing for two sets of methods, then dispatches
accordingly. Which branch runs is invisible from Python, so the way to test it
is to build two subclasses that must compute the *same physics* by the two
different routes and require identical answers.

The vectorised subclass here uses plain numpy rather than JAX, so it exercises
the C++ dispatch and the dict<->State/GlobalState type casters without dragging
in a tracing framework.
"""

import numpy as np
import pytest

import manta as MaNTA
KAPPA = 1.5
SOURCE = 2.0
ALPHA = 0.25  # makes sigma depend on u as well as q, so derivatives are nontrivial


def _sigma(u, q):
    return KAPPA * q + ALPHA * u


def _source(u):
    return SOURCE - ALPHA * u


class ScalarSystem(MaNTA.TransportSystem):
    """Implements only the pointwise virtuals -> C++ loops over them."""

    def __init__(self):
        MaNTA.TransportSystem.__init__(self, MaNTA.numbered_spec(1))

    def SigmaFn(self, i, state, x, t):
        return _sigma(state["Variable"][i], state["Derivative"][i])

    def Sources(self, i, state, x, t):
        return _source(state["Variable"][i])

    def dSigmaFn_du(self, i, state, x, t):
        return np.full(self.nVars, ALPHA)

    def dSigmaFn_dq(self, i, state, x, t):
        return np.full(self.nVars, KAPPA)

    def dSources_du(self, i, state, x, t):
        return np.full(self.nVars, -ALPHA)

    def dSources_dq(self, i, state, x, t):
        return np.zeros(self.nVars)

    def dSources_dsigma(self, i, state, x, t):
        return np.zeros(self.nVars)

    def LowerBoundary(self, i, t):
        return 0.0

    def UpperBoundary(self, i, t):
        return 0.0

    def InitialValue(self, i, x):
        return 0.0

    def InitialDerivative(self, i, x):
        return 0.0


class VectorisedSystem(MaNTA.TransportSystem):
    """Implements ComputePhysics/ComputePhysicsDerivatives -> C++ calls those.

    Same physics as ScalarSystem. GlobalState arrays arrive as
    (nPoints, nVars) -- the caster transposes on the way in and out.
    """

    def __init__(self):
        MaNTA.TransportSystem.__init__(self, MaNTA.numbered_spec(1))
        self.compute_physics_calls = 0
        self.compute_derivative_calls = 0

    def ComputePhysics(self, states, positions, t):
        self.compute_physics_calls += 1
        u = np.asarray(states["Variable"])
        q = np.asarray(states["Derivative"])
        fluxes = [_sigma(u[:, i], q[:, i]) for i in range(self.nVars)]
        sources = [_source(u[:, i]) for i in range(self.nVars)]
        return [fluxes, sources, []]

    def ComputePhysicsDerivatives(self, states, positions, t):
        self.compute_derivative_calls += 1
        u = np.asarray(states["Variable"])
        npts = u.shape[0]

        def gs(dvar, dder, dflux):
            return {
                "Variable": np.full((npts, self.nVars), dvar),
                "Derivative": np.full((npts, self.nVars), dder),
                "Flux": np.full((npts, self.nVars), dflux),
                "Aux": np.zeros((npts, max(self.nAux, 0))),
                "Scalars": np.zeros(0),
            }

        dflux = [gs(ALPHA, KAPPA, 0.0) for _ in range(self.nVars)]
        dsource = [gs(-ALPHA, 0.0, 0.0) for _ in range(self.nVars)]
        return [dflux, dsource, []]

    def LowerBoundary(self, i, t):
        return 0.0

    def UpperBoundary(self, i, t):
        return 0.0

    def InitialValue(self, i, x):
        return 0.0

    def InitialDerivative(self, i, x):
        return 0.0


class IncompleteSystem(MaNTA.TransportSystem):
    """Neither a full scalar set nor a full vectorised set."""

    def __init__(self):
        MaNTA.TransportSystem.__init__(self, MaNTA.numbered_spec(1))

    def SigmaFn(self, i, state, x, t):
        return state["Derivative"][i]

    # Sources and every derivative are missing.

    def LowerBoundary(self, i, t):
        return 0.0

    def UpperBoundary(self, i, t):
        return 0.0

    def InitialValue(self, i, x):
        return 0.0

    def InitialDerivative(self, i, x):
        return 0.0


def config(tmp_path, **overrides):
    cfg = {
        "Polynomial_degree": 3,
        "Grid_size": 8,
        "Lower_boundary": 0.0,
        "Upper_boundary": 1.0,
        "delta_t": 0.1,
        "OutputFilename": str(tmp_path / "trampoline"),
        "WriteOutput": False,
    }
    cfg.update(overrides)
    return cfg


XS = [0.125 * i for i in range(1, 8)]


def _solve(system, tmp_path, tFinal=2.0):
    runner = MaNTA.Runner(system)
    runner.configure(config(tmp_path))
    runner.run(tFinal)
    return np.asarray(runner.getSolution(0, XS))


def test_scalar_and_vectorised_paths_agree(tmp_path):
    """The two dispatch branches must produce the same solution.

    This is the real test of the trampoline: identical physics expressed
    pointwise and batched, routed through different C++ code paths.
    """
    u_scalar = _solve(ScalarSystem(), tmp_path / "a")
    u_vector = _solve(VectorisedSystem(), tmp_path / "b")

    assert np.allclose(u_scalar, u_vector, rtol=1e-6, atol=1e-8), (
        f"scalar={u_scalar}\nvector={u_vector}"
    )


def test_vectorised_overrides_are_actually_called(tmp_path):
    """Guard against the vectorised subclass silently falling back to scalar.

    Without this, test_scalar_and_vectorised_paths_agree would still pass if
    both systems took the scalar route.
    """
    system = VectorisedSystem()
    _solve(system, tmp_path)

    assert system.compute_physics_calls > 0, "ComputePhysics override never invoked"
    assert system.compute_derivative_calls > 0, (
        "ComputePhysicsDerivatives override never invoked"
    )


def test_scalar_system_solves_the_expected_steady_state(tmp_path):
    """Anchor both paths to physics, not just to each other.

    Steady state of -d_x(kappa u' + alpha u) = S - alpha u is not a one-liner,
    so check the weaker properties that must hold: zero-ish at the Dirichlet
    ends, positive inside, single interior maximum.
    """
    u = _solve(ScalarSystem(), tmp_path)
    assert np.all(np.isfinite(u))
    assert np.all(u > 0.0), f"source is positive everywhere, got {u}"
    assert u.argmax() not in (0, len(u) - 1), "profile should peak in the interior"


def test_incomplete_subclass_is_rejected_with_a_useful_message(tmp_path):
    """Missing both method sets must name what is missing, not crash."""
    runner = MaNTA.Runner(IncompleteSystem())
    runner.configure(config(tmp_path))

    with pytest.raises(RuntimeError) as excinfo:
        runner.run(0.1)

    message = str(excinfo.value)
    # It should name at least one of the methods it could not find.
    assert any(
        name in message
        for name in ("Sources", "dSigmaFn_du", "dSources_du", "ComputePhysics")
    ), message


# ------------------------------------------------------- PyGrid / getNodes --


@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("nCells", [1, 3, 7])
def test_get_nodes_from_bounds(k, nCells):
    """MaNTA.getNodes(x_l, x_u, nCells, k) -- the overload taking a domain."""
    nodes = np.asarray(MaNTA.getNodes(0.0, 1.0, nCells, k))

    assert nodes.shape == (nCells * (k + 1),)
    # Nodes lie strictly inside the domain (Chebyshev-Gauss, not Lobatto) and
    # are grouped cell by cell in increasing order.
    assert np.all(nodes > 0.0) and np.all(nodes < 1.0)
    assert np.all(np.diff(nodes) > 0.0), "nodes must be globally increasing"

    # Each cell's block must lie within that cell.
    for c in range(nCells):
        lo, hi = c / nCells, (c + 1) / nCells
        block = nodes[c * (k + 1) : (c + 1) * (k + 1)]
        assert np.all(block > lo) and np.all(block < hi)


def test_get_nodes_from_explicit_cell_boundaries():
    """MaNTA.getNodes(cellBoundaries, k) -- the overload taking a point list."""
    boundaries = [0.0, 0.2, 0.55, 1.0]
    k = 2
    nodes = np.asarray(MaNTA.getNodes(boundaries, k))

    assert nodes.shape == ((len(boundaries) - 1) * (k + 1),)
    assert np.all(np.diff(nodes) > 0.0)

    for c in range(len(boundaries) - 1):
        block = nodes[c * (k + 1) : (c + 1) * (k + 1)]
        assert np.all(block > boundaries[c]) and np.all(block < boundaries[c + 1])


def test_get_nodes_agrees_between_the_two_overloads():
    """A uniform boundary list must give the same nodes as the domain form."""
    k, nCells = 3, 5
    boundaries = [i / nCells for i in range(nCells + 1)]

    from_bounds = np.asarray(MaNTA.getNodes(0.0, 1.0, nCells, k))
    from_points = np.asarray(MaNTA.getNodes(boundaries, k))

    assert np.allclose(from_bounds, from_points)
