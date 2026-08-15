"""Tests for the dict <-> State / GlobalState type casters in Python.cpp.

Every value that crosses between the solver and a Python physics case goes
through one of these two casters, and they are the only place in the project
where an array is silently transposed:

    State        Vector (nVars,)         <-> dict of 1-D arrays
    GlobalState  Matrix (nVars, nPoints) <-> dict of (nPoints, nVars) arrays

The GlobalState caster transposes on the way in *and* on the way out. A
round-trip therefore cannot detect a missing transpose on its own -- two
mistakes would cancel. What does detect it is checking the orientation from the
inside: run a batched call through the C++ serial fallback, which slices the
GlobalState column by column into pointwise States, and require each slice to
be the right point rather than the right variable. With nVars != nPoints, a
transposed load cannot even produce the right shape, let alone the right values.

Everything here calls the *bound base-class* method
(`MaNTA.TransportSystem.SigmaFn(obj, ...)`) rather than `obj.SigmaFn(...)`.
That matters: the latter would dispatch straight to the Python override and
never enter C++ at all, so the caster would not run and the test would pass
vacuously.
"""

import numpy as np
import pytest

import manta as MaNTA
NVARS = 2
NAUX = 3
NSCALARS = 2
NPOINTS = 6  # deliberately != NVARS, so a transpose cannot hide


class Recorder(MaNTA.TransportSystem):
    """Records what the C++ side hands it, and hands back what it is told to."""

    def __init__(self, nVars=NVARS, nAux=0, nScalars=0):
        MaNTA.TransportSystem.__init__(self, MaNTA.numbered_spec(nVars, nScalars=nScalars, nAux=nAux))
        self.seen = []
        self.seen_batched = []

    def _record(self, i, state, x, t):
        # Copied out: the view is a window onto solver memory and is only valid
        # for the duration of this call.
        self.seen.append(
            {
                "i": i,
                "x": x,
                "t": t,
                "state": {
                    "Variable": np.array(state.u, copy=True),
                    "Derivative": np.array(state.q, copy=True),
                    "Flux": np.array(state.sigma, copy=True),
                    "Aux": np.array(state.phi, copy=True),
                    "Scalars": np.array(state.scalars, copy=True),
                },
            }
        )

    def SigmaFn(self, i, state, x, t):
        self._record(i, state, x, t)
        return float(state.u[i])

    def Sources(self, i, state, x, t):
        self._record(i, state, x, t)
        return float(state.q[i])

    def dSigmaFn_du(self, i, state, x, t):
        self._record(i, state, x, t)
        return np.asarray(state.sigma, dtype=float)

    def dSigmaFn_dq(self, i, state, x, t):
        return np.zeros(self.nVars)

    def dSources_du(self, i, state, x, t):
        return np.zeros(self.nVars)

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

    # --- required once nAux > 0 ----------------------------------------
    # Declaring nAux without these is rejected at setup (it used to segfault
    # mid-solve); they are stubs because nothing here runs the solver.
    def AuxGPrime(self, i, state, x, t):
        return {
            "Variable": np.zeros(self.nVars),
            "Derivative": np.zeros(self.nVars),
            "Flux": np.zeros(self.nVars),
            "Aux": np.zeros(self.nAux),
            "Scalars": np.zeros(self.nScalars),
        }

    def dSources_dPhi(self, i, state, x, t):
        return np.zeros(self.nAux)

    def dSigma_dPhi(self, i, state, x, t):
        return np.zeros(self.nAux)

    def InitialAuxValue(self, i, x):
        return 0.0

    # --- required once nScalars > 0 ------------------------------------
    # initializeOverrides refuses a subclass that declares scalars without all
    # four of these, so they have to exist even though nothing here runs the
    # solver. Their presence also exercises that branch of the check.
    def ScalarG(self, s, states, states_dt, abscissae, weights, phi_boundary, t):
        return 0.0

    def ScalarGPrime(self, states, states_dt, abscissae, weights, phi_boundary, t):
        return [[], []]

    def InitialScalarDerivative(self, s, states, states_dt, weights):
        return 0.0

    def dSources_dScalars(self, s, state, x, t):
        return np.zeros(self.nScalars)

    def InitialScalarValue(self, s):
        return 0.0

    def isScalarDifferential(self, i):
        return False


def make_state(nVars=NVARS, nAux=NAUX, nScalars=NSCALARS):
    """A pointwise state whose every entry is distinct."""
    return {
        "Variable": np.arange(1.0, nVars + 1.0),
        "Derivative": np.arange(10.0, 10.0 + nVars),
        "Flux": np.arange(100.0, 100.0 + nVars),
        "Aux": np.arange(1000.0, 1000.0 + nAux),
        "Scalars": np.arange(-1.0, -1.0 - nScalars, -1.0),
    }


def make_global_state(nVars=NVARS, nPoints=NPOINTS, nAux=NAUX, nScalars=NSCALARS):
    """A batched state where entry (point j, variable v) is 10*j + v.

    That encoding makes the orientation readable off any single value.
    """
    grid = np.arange(nPoints)[:, None] * 10.0 + np.arange(nVars)[None, :]
    return {
        "Variable": grid,
        "Derivative": grid + 0.5,
        "Flux": grid + 0.25,
        "Aux": np.arange(nPoints)[:, None] * 10.0 + np.arange(nAux)[None, :] + 0.75,
        "Scalars": np.arange(-1.0, -1.0 - nScalars, -1.0),
    }


POSITIONS = [0.1 * (j + 1) for j in range(NPOINTS)]


# ----------------------------------------------------------- the State view --
#
# These used to hand C++ a dict and check it came back unchanged. A State is a
# non-owning view of solver memory now, with no way to construct one from
# Python, so the way in is the batched entry point: with no vectorised
# override it falls back to the C++ serial loop, which slices the GlobalState
# into pointwise States and calls the Python hook once per point.


def test_the_view_shows_each_field_under_its_own_name():
    system = Recorder(nAux=NAUX, nScalars=NSCALARS)
    states = make_global_state()

    MaNTA.TransportSystem.SigmaFn_v(system, 0, states, POSITIONS, 2.5)

    seen = system.seen[0]["state"]
    assert np.allclose(seen["Variable"], states["Variable"][0, :])   # state.u
    assert np.allclose(seen["Derivative"], states["Derivative"][0, :])  # state.q
    assert np.allclose(seen["Flux"], states["Flux"][0, :])           # state.sigma
    assert np.allclose(seen["Aux"], states["Aux"][0, :])             # state.phi
    assert np.allclose(seen["Scalars"], states["Scalars"])           # state.scalars


def test_the_value_read_through_the_view_comes_back_to_cpp():
    """The view is only half the story -- the value must survive the return."""
    system = Recorder(nAux=NAUX, nScalars=NSCALARS)
    states = make_global_state()

    for i in range(NVARS):
        out = np.asarray(
            MaNTA.TransportSystem.SigmaFn_v(system, i, states, POSITIONS, 0.0)
        )
        # SigmaFn returns state.u[i], one value per point.
        assert np.allclose(out, states["Variable"][:, i])


def test_an_empty_field_is_a_zero_length_array_not_a_failure():
    """nAux = nScalars = 0 leaves Eigen with a null data pointer, which is the
    one case a naive view would hand to py::capsule and crash on."""
    system = Recorder(nAux=0, nScalars=0)
    states = make_global_state(nAux=0, nScalars=0)

    MaNTA.TransportSystem.SigmaFn_v(system, 0, states, POSITIONS, 0.0)

    seen = system.seen[0]["state"]
    assert seen["Aux"].shape == (0,)
    assert seen["Scalars"].shape == (0,)


def test_a_field_is_indexable_by_declared_name_as_well_as_position():
    """The point of naming variables: s.u["density"] rather than s.u[0]."""

    class Named(Recorder):
        def __init__(self):
            MaNTA.TransportSystem.__init__(
                self,
                variables=[MaNTA.Field("density"), MaNTA.Field("temperature")],
                aux=[MaNTA.Aux("potential")],
            )
            self.seen = []
            self.seen_batched = []
            self.by_name = []

        def SigmaFn(self, i, state, x, t):
            self.by_name.append(
                (state.u["density"], state.u["temperature"], state.phi["potential"])
            )
            return float(state.u[i])

    system = Named()
    states = make_global_state(nAux=1, nScalars=0)
    MaNTA.TransportSystem.SigmaFn_v(system, 0, states, POSITIONS, 0.0)

    for j, (density, temperature, potential) in enumerate(system.by_name):
        assert density == pytest.approx(states["Variable"][j, 0])
        assert temperature == pytest.approx(states["Variable"][j, 1])
        assert potential == pytest.approx(states["Aux"][j, 0])


def test_an_undeclared_name_says_what_is_declared():
    class Named(Recorder):
        def __init__(self):
            MaNTA.TransportSystem.__init__(
                self, variables=[MaNTA.Field("density"), MaNTA.Field("temperature")]
            )
            self.seen = []
            self.seen_batched = []

        def SigmaFn(self, i, state, x, t):
            return float(state.u["dnesity"])  # typo, on purpose

    system = Named()
    states = make_global_state(nAux=0, nScalars=0)
    with pytest.raises(RuntimeError, match="density"):
        MaNTA.TransportSystem.SigmaFn_v(system, 0, states, POSITIONS, 0.0)


def test_sigma_hat_is_the_negation_of_the_stored_flux():
    """The sign convention, made visible: sigma is what the solver stores,
    sigmaHat is what SigmaFn returned."""

    class Flux(Recorder):
        def __init__(self):
            MaNTA.TransportSystem.__init__(self, MaNTA.numbered_spec(NVARS))
            self.seen = []
            self.seen_batched = []
            self.pairs = []

        def SigmaFn(self, i, state, x, t):
            self.pairs.append((state.sigma[i], state.sigmaHat[i]))
            return 0.0

    system = Flux()
    states = make_global_state(nAux=0, nScalars=0)
    MaNTA.TransportSystem.SigmaFn_v(system, 0, states, POSITIONS, 0.0)

    for stored, physical in system.pairs:
        assert physical == pytest.approx(-stored)


# --------------------------------------------------- the GlobalState caster --


def test_global_state_is_transposed_on_the_way_into_cpp():
    """The orientation test.

    SigmaFn_v with no vectorised override falls back to the C++ serial loop,
    which slices the GlobalState with `states[j]` -- column j, i.e. point j. So
    the value handed to the pointwise SigmaFn at call j must be point j's row of
    the Python array. If the caster did not transpose on load, column j would be
    variable j instead and this would be wrong (and, since nPoints != nVars,
    would run off the end).
    """
    system = Recorder(nAux=NAUX, nScalars=NSCALARS)
    states = make_global_state()

    out = np.asarray(
        MaNTA.TransportSystem.SigmaFn_v(system, 0, states, POSITIONS, 1.0)
    )

    assert out.shape == (NPOINTS,)
    # SigmaFn returns state.u[0] = 10*j + 0.
    assert np.allclose(out, states["Variable"][:, 0])

    assert len(system.seen) == NPOINTS
    for j, call in enumerate(system.seen):
        assert call["x"] == pytest.approx(POSITIONS[j]), f"point {j} got x = {call['x']}"
        assert np.allclose(call["state"]["Variable"], states["Variable"][j, :]), (
            f"point {j}: got {call['state']['Variable']}, "
            f"want {states['Variable'][j, :]}"
        )
        assert np.allclose(call["state"]["Aux"], states["Aux"][j, :])
        # Scalars are global, so every point sees the same vector.
        assert np.allclose(call["state"]["Scalars"], states["Scalars"])


def test_global_state_round_trips_through_a_vectorised_override():
    """dict -> GlobalState -> dict, with the transpose applied in both directions.

    ComputePhysics is the one hook that receives a GlobalState *as a dict*, so
    it is where the cast direction can be observed. A non-square shape means a
    caster that transposed on only one side would produce (nVars, nPoints) here
    and fail the shape check.
    """

    class Vectorised(Recorder):
        def ComputePhysics(self, states, positions, t):
            self.seen_batched.append(
                {k: np.array(v, copy=True) for k, v in states.items()}
            )
            n = len(positions)
            sigma = [np.zeros(n) for _ in range(self.nVars)]
            source = [np.zeros(n) for _ in range(self.nVars)]
            aux = [np.zeros(n) for _ in range(self.nAux)]
            return [sigma, source, aux]

        def ComputePhysicsDerivatives(self, states, positions, t):
            raise AssertionError("not exercised by this test")

    system = Vectorised(nAux=NAUX, nScalars=NSCALARS)
    states = make_global_state()

    MaNTA.TransportSystem.ComputePhysics(system, states, POSITIONS, 0.0)

    assert len(system.seen_batched) == 1
    got = system.seen_batched[0]
    for field in ("Variable", "Derivative", "Flux", "Aux"):
        assert got[field].shape == states[field].shape, (
            f"{field}: {got[field].shape} != {states[field].shape}"
        )
        assert np.allclose(got[field], states[field]), field
    assert np.allclose(got["Scalars"], states["Scalars"])


def test_global_state_scalars_are_a_flat_vector_not_a_broadcast_matrix():
    """Scalars are global, so they stay 1-D across the GlobalState boundary.

    The caster reads them through a raw buffer (py::array_t + Map) rather than
    the Eigen path used for the matrices, so the shape contract is separate.
    """

    class Vectorised(Recorder):
        def ComputePhysics(self, states, positions, t):
            self.seen_batched.append(np.array(states["Scalars"], copy=True))
            n = len(positions)
            return [
                [np.zeros(n) for _ in range(self.nVars)],
                [np.zeros(n) for _ in range(self.nVars)],
                [],
            ]

        def ComputePhysicsDerivatives(self, states, positions, t):
            raise AssertionError("not exercised by this test")

    system = Vectorised(nAux=0, nScalars=NSCALARS)
    states = make_global_state(nAux=0)
    states["Aux"] = np.zeros((NPOINTS, 0))

    MaNTA.TransportSystem.ComputePhysics(system, states, POSITIONS, 0.0)

    scalars = system.seen_batched[0]
    assert scalars.ndim == 1, f"expected 1-D scalars, got shape {scalars.shape}"
    assert np.allclose(scalars, states["Scalars"])


@pytest.mark.parametrize("nPoints", [1, 5, 17])
def test_global_state_survives_a_range_of_point_counts(nPoints):
    """Guard against an off-by-one in the (nVars, nPoints) sizing."""
    system = Recorder(nAux=0, nScalars=0)
    states = make_global_state(nPoints=nPoints, nAux=0, nScalars=0)
    positions = [0.05 * (j + 1) for j in range(nPoints)]

    out = np.asarray(
        MaNTA.TransportSystem.SigmaFn_v(system, 1, states, positions, 0.0)
    )
    assert out.shape == (nPoints,)
    assert np.allclose(out, states["Variable"][:, 1])


def test_geometry_reaches_a_batched_call_with_the_right_orientation():
    """The GlobalState caster transposes in both directions, so a round trip
    cannot detect a missing transpose. Check the orientation from inside a
    batched call, where the (nPoints, nGeom) shape is observable."""
    seen = {}

    class GeometryProbe(MaNTA.TransportSystem):
        variables = [MaNTA.Field("n", "density", "m^-3")]

        def ComputePhysics(self, states, positions, t):
            seen["shape"] = states["Geometry"].shape
            seen["npoints"] = len(positions)
            n = len(positions)
            return [[np.zeros(n)], [np.zeros(n)], []]

        def ComputePhysicsDerivatives(self, states, positions, t):
            return {}

    system = GeometryProbe()
    n_geom = 4  # deliberately != NPOINTS, so a transpose cannot hide
    states = make_global_state(nVars=1, nAux=0, nScalars=0)
    # "Geometry" is optional on the way into C++ (every fixture above omits
    # it), but once loaded it must survive the round trip back out to the
    # override with the (nPoints, nGeom) orientation every other key has.
    states["Geometry"] = (
        np.arange(NPOINTS)[:, None] * 10.0 + np.arange(n_geom)[None, :]
    )

    MaNTA.TransportSystem.ComputePhysics(system, states, POSITIONS, 0.0)

    assert seen["shape"] == (NPOINTS, n_geom)
    assert seen["shape"][0] == seen["npoints"]
