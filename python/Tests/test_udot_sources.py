"""Sources that read ``state.udot``, from Python.

The C++ side is covered in ``Tests/UnitTests/TimeDerivativeSourceTests.cpp``,
including the two mutations that show the tests bite. What is Python's own is
the path from a class attribute to the solver: ``manta.Field(...,
source_reads_time_derivatives=True)`` has to reach the spec, ``state.udot`` has
to be filled by the time ``Sources`` is called, and ``dSources_dudot`` has to be
found by the trampoline's optional-override probe.

The manufactured problem is the same shape as the C++ one. Two variables solve
the same equation with the same exact solution; variable 1's source carries
``C * udot[0]`` with a compensating ``- C * du_exact/dt`` subtracted, so the
exact solution is unchanged and a run that never filled ``udot`` would be
integrating a source short by ``C du/dt``. That is what makes the comparison
against ``C = 0`` an assertion rather than a formality: measured, dropping the
term takes the error from 7e-5 to 2e-2.
"""

import numpy as np
import pytest

import manta


C = 0.4
XS = [0.125, 0.25, 0.5, 0.75, 0.875]


def exact(x, t):
    return np.sin(np.pi * x) * (1.0 + t)


def diffusion_source(x, t):
    """u_t - u_xx for u = sin(pi x)(1 + t)."""
    return np.sin(np.pi * x) * (1.0 + np.pi**2 * (1.0 + t))


class Coupled(manta.TransportSystem):
    """Variable 1's source reads du_0/dt; variable 0's reads nothing.

    The asymmetry is deliberate. Only variable 1 declares the flag, so a run in
    which ``udot`` were filled for everything, or the flag honoured per system
    rather than per variable, would still pass a test that declared both.
    """

    variables = [
        manta.Field("n", "first", ""),
        manta.Field("T", "second", "", source_reads_time_derivatives=True),
    ]

    def __init__(self, coupling=C):
        super().__init__()
        self.coupling = coupling
        self.udot_seen = []

    def SigmaFn(self, i, s, x, t):
        return s.q[i]

    def Sources(self, i, s, x, t):
        if i == 0:
            return diffusion_source(x, t)
        self.udot_seen.append(s.udot[0])
        return (diffusion_source(x, t)
                - self.coupling * np.sin(np.pi * x)
                + self.coupling * s.udot[0])

    def dSigmaFn_dq(self, i, s, x, t):
        v = np.zeros(2)
        v[i] = 1.0
        return v

    def dSources_dudot(self, i, s, x, t):
        v = np.zeros(2)
        if i == 1:
            v[0] = self.coupling
        return v

    def InitialValue(self, i, x):
        return exact(x, 0.0)

    def InitialDerivative(self, i, x):
        return np.pi * np.cos(np.pi * x)

    def LowerBoundary(self, i, t):
        return 0.0

    def UpperBoundary(self, i, t):
        return 0.0


def solved(system, t_final, tmp_path, stem):
    runner = manta.Runner(system)
    runner.configure({
        "Polynomial_degree": 3,
        "Grid_size": 8,
        "Lower_boundary": 0.0,
        "Upper_boundary": 1.0,
        "delta_t": t_final,
        "Relative_tolerance": 1e-10,
        "Absolute_tolerance": [1e-11],
        "MinStepSize": 1e-14,
        "OutputFilename": str(tmp_path / stem),
        "WriteOutput": False,
    })
    runner.run(t_final)
    return np.asarray(runner.getSolution(1, list(XS))).reshape(-1)


def test_the_flag_reaches_the_spec():
    """A class attribute, not a runtime setting: it describes the equations."""
    case = Coupled()
    assert case.spec.variables[1].source_reads_time_derivatives
    assert not case.spec.variables[0].source_reads_time_derivatives


def test_the_default_is_off():
    """Every existing case keeps its behaviour by declaring nothing."""
    assert not manta.Field("u").source_reads_time_derivatives


def test_udot_is_filled_before_sources_is_called(tmp_path):
    """The value has to be there, and it has to be the time derivative.

    At the exact solution ``du/dt = sin(pi x)``, which is O(1) on this domain, so
    a ``udot`` that were never filled would show up here as a column of zeros
    rather than as a small error.
    """
    case = Coupled()
    solved(case, 0.05, tmp_path, "udot_seen")

    seen = np.abs(np.asarray(case.udot_seen))
    assert seen.size > 0
    assert seen.max() > 0.1


def test_the_coupled_run_matches_the_closed_form(tmp_path):
    """The assertion the coupling is built to make fail if udot were dropped.

    Both problems have the same exact solution, so their errors should be
    comparable; without the term the coupled one is short by ``C du/dt`` and the
    error is two orders larger.
    """
    t_final = 0.1

    coupled = solved(Coupled(C), t_final, tmp_path, "udot_coupled")
    uncoupled = solved(Coupled(0.0), t_final, tmp_path, "udot_uncoupled")
    reference = exact(np.asarray(XS), t_final)

    err_coupled = np.max(np.abs(coupled - reference))
    err_uncoupled = np.max(np.abs(uncoupled - reference))

    assert err_coupled < 1e-4
    assert err_coupled < 10.0 * err_uncoupled


def test_a_case_that_declares_nothing_sees_zero(tmp_path):
    """udot is not filled at all unless a variable asks, and reads as zero."""

    class Silent(manta.TransportSystem):
        variables = [manta.Field("u", "the diffused quantity", "")]

        def __init__(self):
            super().__init__()
            self.udot_seen = []

        def SigmaFn(self, i, s, x, t):
            return s.q[i]

        def Sources(self, i, s, x, t):
            self.udot_seen.append(s.udot[0])
            return diffusion_source(x, t)

        def dSigmaFn_dq(self, i, s, x, t):
            return np.array([1.0])

        def InitialValue(self, i, x):
            return exact(x, 0.0)

        def InitialDerivative(self, i, x):
            return np.pi * np.cos(np.pi * x)

        def LowerBoundary(self, i, t):
            return 0.0

        def UpperBoundary(self, i, t):
            return 0.0

    case = Silent()
    runner = manta.Runner(case)
    runner.configure({
        "Polynomial_degree": 3,
        "Grid_size": 8,
        "Lower_boundary": 0.0,
        "Upper_boundary": 1.0,
        "delta_t": 0.05,
        "Relative_tolerance": 1e-8,
        "Absolute_tolerance": [1e-9],
        "MinStepSize": 1e-14,
        "OutputFilename": str(tmp_path / "udot_silent"),
        "WriteOutput": False,
    })
    runner.run(0.05)

    seen = np.asarray(case.udot_seen)
    assert seen.size > 0
    assert np.all(seen == 0.0)


class CoupledVectorized(manta.TransportSystem):
    """The same problem on the batched path, which is a different interface.

    A case is classified as vectorised by supplying ``ComputePhysics`` and
    ``ComputePhysicsDerivatives``, and it then never sees a ``manta.State``: the
    whole grid crosses as a dict of ``(nPoints, nVars)`` arrays. So ``du/dt``
    reaches it as the ``"VariableDot"`` entry of that dict, and the derivative
    goes back through ``ComputeSourceTimeDerivatives`` -- a call of its own,
    because ``ComputePhysicsDerivatives`` returns exactly three groups and
    widening that would change the interface for every vectorised case there is.

    The block goes in the **Variable** slice of its dict. ``dS_i/d(udot_j)`` has
    the shape ``dS_i/du_j`` has and lands in the same block of the cell matrix,
    one factor of alpha apart.
    """

    variables = [
        manta.Field("n", "first", ""),
        manta.Field("T", "second", "", source_reads_time_derivatives=True),
    ]

    def __init__(self, coupling=C):
        super().__init__()
        self.coupling = coupling
        self.udot_seen = []

    @staticmethod
    def _zero_blocks(n):
        return {
            "Variable": np.zeros((n, 2)),
            "Derivative": np.zeros((n, 2)),
            "Flux": np.zeros((n, 2)),
            "Aux": np.zeros((n, 0)),
            "Scalars": np.zeros(0),
        }

    def ComputePhysics(self, states, positions, t):
        x = np.asarray(positions)
        q = np.asarray(states["Derivative"])
        udot = np.asarray(states["VariableDot"])
        self.udot_seen.append(np.array(udot, copy=True))

        # `udot.size` rather than a shape assumption: a run in which no
        # variable declares the flag hands over an empty array, which is what
        # the last test in this file pins.
        dn_dt = udot[:, 0] if udot.size else 0.0
        base = diffusion_source(x, t)
        sources = [
            base,
            base - self.coupling * np.sin(np.pi * x) + self.coupling * dn_dt,
        ]
        return [[q[:, 0], q[:, 1]], sources, []]

    def ComputePhysicsDerivatives(self, states, positions, t):
        n = len(positions)
        dflux = []
        for i in range(2):
            block = self._zero_blocks(n)
            block["Derivative"][:, i] = 1.0
            dflux.append(block)
        return [dflux, [self._zero_blocks(n) for _ in range(2)], []]

    def ComputeSourceTimeDerivatives(self, states, positions, t):
        n = len(positions)
        out = [self._zero_blocks(n) for _ in range(2)]
        out[1]["Variable"][:, 0] = self.coupling
        return out

    def InitialValue(self, i, x):
        return exact(x, 0.0)

    def InitialDerivative(self, i, x):
        return np.pi * np.cos(np.pi * x)

    def LowerBoundary(self, i, t):
        return 0.0

    def UpperBoundary(self, i, t):
        return 0.0


def test_the_batched_state_carries_udot(tmp_path):
    """The dict has the key, it is the right shape, and it is filled.

    ``(nPoints, nVars)`` like every other entry, and O(1) at the exact solution,
    so a key that existed but was never filled would fail the second assertion
    rather than the first.
    """
    case = CoupledVectorized()
    solved(case, 0.05, tmp_path, "udot_vec_seen")

    filled = [a for a in case.udot_seen if a.size]

    assert filled, "no call to ComputePhysics saw a filled VariableDot"
    assert all(a.shape[1] == 2 for a in filled)
    assert np.abs(np.concatenate([a[:, 0] for a in filled])).max() > 0.1

    # Not every call has one. The physics is evaluated while the initial
    # condition is being built, before there is a dYdt to read, and those calls
    # hand over the empty array -- which is the other half of why a vectorised
    # case tests `.size` rather than indexing straight in.
    assert any(a.size == 0 for a in case.udot_seen)


def test_a_vectorised_coupled_run_matches_the_closed_form(tmp_path):
    """The batched path end to end, against the same closed form.

    Both the value and the Jacobian block go through interfaces the pointwise
    case never touches, so this is not a repeat of the test above it.
    """
    t_final = 0.1

    coupled = solved(CoupledVectorized(C), t_final, tmp_path, "udot_vec_coupled")
    uncoupled = solved(CoupledVectorized(0.0), t_final, tmp_path, "udot_vec_none")
    reference = exact(np.asarray(XS), t_final)

    err_coupled = np.max(np.abs(coupled - reference))
    err_uncoupled = np.max(np.abs(uncoupled - reference))

    assert err_coupled < 1e-4
    assert err_coupled < 10.0 * err_uncoupled


def test_the_batched_state_omits_udot_when_nothing_declares_it(tmp_path):
    """Empty, not a grid of zeros: a case that does not ask pays nothing.

    The solver never builds the matrix, so what crosses is an empty array. A
    vectorised case that wants to know whether it is in a run that fills it
    should test the size rather than assume a shape.
    """

    class SilentVectorized(CoupledVectorized):
        variables = [
            manta.Field("n", "first", ""),
            manta.Field("T", "second", ""),
        ]

    case = SilentVectorized(0.0)
    solved(case, 0.05, tmp_path, "udot_vec_silent")

    assert case.udot_seen
    assert all(a.size == 0 for a in case.udot_seen)


def _jax_state(nPoints=None):
    """A state dict of the shape ``manta.jax``'s State.from_manta accepts.

    Pointwise when ``nPoints`` is None -- one value per variable, which is what
    a ``manta.State`` view presents -- and batched otherwise.
    """
    if nPoints is None:
        return {
            "Variable": np.array([0.5, 0.25]),
            "Derivative": np.array([0.1, -0.2]),
            "Flux": np.zeros(2),
            "Aux": np.zeros(0),
            "Scalars": np.zeros(0),
            "VariableDot": np.array([0.7, 0.3]),
        }
    return {
        "Variable": np.full((nPoints, 2), 0.5),
        "Derivative": np.zeros((nPoints, 2)),
        "Flux": np.zeros((nPoints, 2)),
        "Aux": np.zeros((nPoints, 0)),
        "Scalars": np.zeros(0),
        "VariableDot": np.linspace(0.1, 0.9, 2 * nPoints).reshape(nPoints, 2),
    }


def test_a_jax_case_gets_the_block_from_its_own_gradient():
    """``manta.jax`` needs no hook from the case, and that is the whole point.

    ``State.VariableDot`` is a field of the layer's state like ``Variable`` or
    ``Derivative``, so ``grad(source)`` carries :math:`dS/d\\dot u` beside the
    four blocks it already produced and ``JAXTransportSystem`` hands over that
    component as ``dSources_dudot``.

    Driven through the hooks rather than a solve. The pointwise JAX path crosses
    into Python once per point *per hook*, so a run of the manufactured problem
    above takes minutes where the numpy case takes a second, and it is float32
    besides -- neither of which says anything about ``udot``. The hooks take the
    same dict ``State.from_manta`` accepts, which is what makes this direct.
    """
    pytest.importorskip("equinox")

    import jax.numpy as jnp

    from manta.jax import JAXTransportSystem

    class JaxCoupled(JAXTransportSystem):
        def __init__(self, coupling):
            super().__init__(
                manta.SystemSpec(
                    variables=[
                        manta.Field("n", "first", ""),
                        manta.Field(
                            "T", "second", "", source_reads_time_derivatives=True
                        ),
                    ]
                )
            )
            self.params = coupling

        def sigma(self, index, state, x, t, params):
            return state.Derivative[index]

        def source(self, index, state, x, t, params):
            # jnp.where, not an `if`: the derivative hooks are jitted with the
            # variable index as an argument, so it is a tracer here and a Python
            # branch on it raises.
            base = jnp.sin(jnp.pi * x) * (1.0 + jnp.pi**2 * (1.0 + t))
            coupling = params * (state.VariableDot[0] - jnp.sin(jnp.pi * x))
            return base + jnp.where(index == 1, coupling, 0.0)

        def aux(self, index, state, x, t, params):
            return 0.0

        def InitialValue(self, index, x):
            return jnp.sin(jnp.pi * x)

        def LowerBoundary(self, index, t):
            return 0.0

        def UpperBoundary(self, index, t):
            return 0.0

    case = JaxCoupled(C)
    state = _jax_state()
    x, t = 0.25, 0.05

    expected = diffusion_source(x, t) + C * (
        state["VariableDot"][0] - np.sin(np.pi * x)
    )
    assert np.isclose(case.Sources(1, state, x, t), expected, rtol=1e-6)
    assert np.isclose(case.Sources(0, state, x, t), diffusion_source(x, t), rtol=1e-6)

    assert np.allclose(case.dSources_dudot(1, state, x, t), [C, 0.0], atol=1e-6)
    assert np.allclose(case.dSources_dudot(0, state, x, t), [0.0, 0.0], atol=1e-6)


def test_a_vectorised_jax_case_returns_the_block_in_the_variable_slice():
    """The batched JAX route, which is the one the physics systems take.

    ``ComputeSourceTimeDerivatives`` is answered by the layer, out of the same
    ``dSources`` gradient, and has to put the block where the solver reads it:
    the Variable slice, with the other slices shaped as the state is so the
    solver's own shape check passes.
    """
    pytest.importorskip("equinox")

    import jax.numpy as jnp

    from manta.jax import VectorizedTransportSystem

    class JaxVecCoupled(VectorizedTransportSystem):
        def __init__(self, coupling):
            super().__init__(
                manta.SystemSpec(
                    variables=[
                        manta.Field("n", "first", ""),
                        manta.Field(
                            "T", "second", "", source_reads_time_derivatives=True
                        ),
                    ]
                )
            )
            self.params = coupling

        def sigma(self, index, state, x, t, params):
            return state.Derivative[index]

        def source(self, index, state, x, t, params):
            base = jnp.sin(jnp.pi * x) * (1.0 + jnp.pi**2 * (1.0 + t))
            coupling = params * (state.VariableDot[0] - jnp.sin(jnp.pi * x))
            return base + jnp.where(index == 1, coupling, 0.0)

        def aux(self, index, state, x, t, params):
            return 0.0

        def InitialValue(self, index, x):
            return jnp.sin(jnp.pi * x)

        def LowerBoundary(self, index, t):
            return 0.0

        def UpperBoundary(self, index, t):
            return 0.0

    nPoints = 3
    case = JaxVecCoupled(C)
    out = case.ComputeSourceTimeDerivatives(_jax_state(nPoints), [0.1, 0.5, 0.9], 0.0)

    assert len(out) == 2
    assert np.allclose(out[1]["Variable"][:, 0], C, atol=1e-6)
    assert np.allclose(out[1]["Variable"][:, 1], 0.0, atol=1e-6)
    assert np.allclose(out[0]["Variable"], 0.0, atol=1e-6)

    for block in out:
        assert block["Variable"].shape == (nPoints, 2)
        assert block["Derivative"].shape == (nPoints, 2)
        assert np.all(block["Derivative"] == 0.0)


def test_a_vectorised_block_reaches_the_solver(tmp_path):
    """The Jacobian block, not just the value -- and the refusal proves it.

    A missing Jacobian term costs Newton iterations and nothing else, so a solve
    that still converges says nothing about whether the solver ever read the
    block. ``checkEffectiveMassMatrix`` does: it assembles ``X - dS/d(udot)`` out
    of this same path and refuses a run whose source cancels its own mass term.
    With ``aFn = 1`` a source carrying ``+1.0 * udot`` for its own variable is
    exactly that case, so the refusal happens if and only if the batched block
    arrived.
    """

    class Cancelling(CoupledVectorized):
        variables = [
            manta.Field("n", "first", ""),
            manta.Field("T", "second", "", source_reads_time_derivatives=True),
        ]

        def ComputePhysics(self, states, positions, t):
            x = np.asarray(positions)
            q = np.asarray(states["Derivative"])
            udot = np.asarray(states["VariableDot"])
            own = udot[:, 1] if udot.size else 0.0
            base = diffusion_source(x, t)
            return [[q[:, 0], q[:, 1]], [base, base + own], []]

        def ComputeSourceTimeDerivatives(self, states, positions, t):
            n = len(positions)
            out = [self._zero_blocks(n) for _ in range(2)]
            out[1]["Variable"][:, 1] = 1.0
            return out

    with pytest.raises((ValueError, RuntimeError), match="T"):
        solved(Cancelling(), 0.05, tmp_path, "udot_vec_cancelled")
