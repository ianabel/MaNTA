"""One hook at a time over the whole batch.

A Python case used to be all-pointwise or all-vectorised: SigmaFn, Sources and
the rest one point at a time, or ComputePhysics and ComputePhysicsDerivatives
for everything at once. The per-hook batched names -- SigmaFn_v, Sources_v,
AuxG_v, dSigma, dSources, AuxGPrime_v -- were looked up by the trampoline but
never registered, so they could only throw. Each now stands in for the pointwise
hooks it covers, and a case may batch any of them and leave the rest pointwise.

The fixture is one problem written both ways, so every combination must run
the same integration: a nonlinear flux, so the derivative hooks matter, and an
auxiliary variable, so the aux hooks are reached.

    d_t u - d_x[(1 + u^2/2) q] = a + 1,   a = u^2
"""

import numpy as np
import pytest

import manta


class Base(manta.TransportSystem):
    def __init__(self):
        manta.TransportSystem.__init__(self, manta.numbered_spec(1, nAux=1))
        self.calls = {}

    def count(self, name):
        self.calls[name] = self.calls.get(name, 0) + 1

    def LowerBoundary(self, i, t):
        return 0.0

    def UpperBoundary(self, i, t):
        return 0.0

    def InitialValue(self, i, x):
        return 0.0

    def InitialDerivative(self, i, x):
        return 0.0

    def InitialAuxValue(self, i, x):
        return 0.0


class PointwiseValues:
    def SigmaFn(self, i, s, x, t):
        self.count("SigmaFn")
        return (1.0 + 0.5 * s.u[0] ** 2) * s.q[0]

    def Sources(self, i, s, x, t):
        self.count("Sources")
        return s.phi[0] + 1.0

    def AuxG(self, i, s, x, t):
        self.count("AuxG")
        return s.phi[0] - s.u[0] ** 2


class PointwiseDerivatives:
    def dSigmaFn_du(self, i, s, x, t):
        self.count("dSigmaFn_du")
        return np.array([s.u[0] * s.q[0]])

    def dSigmaFn_dq(self, i, s, x, t):
        self.count("dSigmaFn_dq")
        return np.array([1.0 + 0.5 * s.u[0] ** 2])

    def dSigma_dPhi(self, i, s, x, t):
        self.count("dSigma_dPhi")
        return np.zeros(1)

    def dSources_dPhi(self, i, s, x, t):
        self.count("dSources_dPhi")
        return np.array([1.0])

    def AuxGPrime(self, i, out, s, x, t):
        self.count("AuxGPrime")
        out.u[0] = -2.0 * s.u[0]
        out.phi[0] = 1.0


def zeros_like(states):
    return {key: np.zeros_like(np.asarray(states[key])) for key in ("Variable", "Derivative", "Flux", "Aux", "Scalars")}


class BatchedValues:
    def SigmaFn_v(self, i, states, x, t):
        self.count("SigmaFn_v")
        u, q = np.asarray(states["Variable"])[:, 0], np.asarray(states["Derivative"])[:, 0]
        return (1.0 + 0.5 * u**2) * q

    def Sources_v(self, i, states, x, t):
        self.count("Sources_v")
        return np.asarray(states["Aux"])[:, 0] + 1.0

    def AuxG_v(self, i, states, x, t):
        self.count("AuxG_v")
        return np.asarray(states["Aux"])[:, 0] - np.asarray(states["Variable"])[:, 0] ** 2


class BatchedDerivatives:
    def dSigma(self, i, states, x, t):
        self.count("dSigma")
        u, q = np.asarray(states["Variable"])[:, 0], np.asarray(states["Derivative"])[:, 0]
        out = zeros_like(states)
        out["Variable"][:, 0] = u * q
        out["Derivative"][:, 0] = 1.0 + 0.5 * u**2
        return out

    def dSources(self, i, states, x, t):
        self.count("dSources")
        out = zeros_like(states)
        out["Aux"][:, 0] = 1.0
        # A slice with no entries may be left empty rather than built as zeros.
        out["Flux"] = np.zeros((0, 0))
        return out

    def AuxGPrime_v(self, i, states, x, t):
        self.count("AuxGPrime_v")
        out = zeros_like(states)
        out["Variable"][:, 0] = -2.0 * np.asarray(states["Variable"])[:, 0]
        out["Aux"][:, 0] = 1.0
        return out


class Pointwise(PointwiseValues, PointwiseDerivatives, Base):
    pass


class ValuesBatched(BatchedValues, PointwiseDerivatives, Base):
    pass


class DerivativesBatched(PointwiseValues, BatchedDerivatives, Base):
    pass


class AllBatched(BatchedValues, BatchedDerivatives, Base):
    pass


XS = list(np.linspace(0.05, 0.95, 19))


def run(system, tmp_path):
    runner = manta.Runner(system)
    runner.configure(
        {
            "PolynomialDegree": 3,
            "GridSize": 6,
            "LowerBoundary": 0.0,
            "UpperBoundary": 1.0,
            "delta_t": 0.25,
            "Relative_tolerance": 1e-8,
            "Absolute_tolerance": [1e-10],
            "MinStepSize": 1e-12,
            "OutputFilename": str(tmp_path / type(system).__name__),
            "WriteOutput": False,
        }
    )
    runner.run(1.0)
    return np.asarray(runner.getSolution(0, XS))


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    system = Pointwise()
    return run(system, tmp_path_factory.mktemp("pointwise")), system.calls


def test_the_pointwise_fixture_solves_a_nontrivial_problem(reference):
    u, calls = reference
    assert np.all(u > 0.0)
    for hook in ("SigmaFn", "Sources", "AuxG", "dSigmaFn_du", "dSigmaFn_dq", "dSources_dPhi", "AuxGPrime"):
        assert calls.get(hook, 0) > 0, hook


@pytest.mark.parametrize(
    "case, batched, pointwise",
    [
        (ValuesBatched, ("SigmaFn_v", "Sources_v", "AuxG_v"), ("SigmaFn", "Sources", "AuxG")),
        (
            DerivativesBatched,
            ("dSigma", "dSources", "AuxGPrime_v"),
            ("dSigmaFn_du", "dSigmaFn_dq", "dSigma_dPhi", "dSources_dPhi", "AuxGPrime"),
        ),
        (AllBatched, ("SigmaFn_v", "Sources_v", "AuxG_v", "dSigma", "dSources", "AuxGPrime_v"), ()),
    ],
)
def test_a_batched_hook_stands_in_for_its_pointwise_ones(reference, tmp_path, case, batched, pointwise):
    system = case()
    u = run(system, tmp_path)
    u_ref, _ = reference

    # The same arithmetic, in numpy over the batch rather than per point, so
    # the runs agree to round-off carried through IDA's steps.
    assert np.max(np.abs(u - u_ref)) < 1e-10 * np.max(np.abs(u_ref))
    for hook in batched:
        assert system.calls.get(hook, 0) > 0, f"{hook} was never called"
    for hook in pointwise:
        assert system.calls.get(hook, 0) == 0, f"{hook} was called beside its batched form"


def test_a_missing_hook_names_both_forms():
    class NoFlux(PointwiseDerivatives, Base):
        def Sources(self, i, s, x, t):
            return 1.0

        def AuxG(self, i, s, x, t):
            return s.phi[0]

    with pytest.raises(RuntimeError, match=r"SigmaFn \(or, over the whole batch, SigmaFn_v\)"):
        run_without_output(NoFlux())


def test_a_batched_derivative_of_the_wrong_shape_is_refused(tmp_path):
    class Transposed(PointwiseValues, BatchedDerivatives, Base):
        def dSigma(self, i, states, x, t):
            out = super().dSigma(i, states, x, t)
            out["Variable"] = np.asarray(out["Variable"]).T.copy()
            return out

    with pytest.raises(RuntimeError, match=r"dSigma returned \"Variable\" of shape"):
        run(Transposed(), tmp_path)


def run_without_output(system):
    runner = manta.Runner(system)
    runner.configure(
        {
            "PolynomialDegree": 2,
            "GridSize": 4,
            "LowerBoundary": 0.0,
            "UpperBoundary": 1.0,
            "delta_t": 0.1,
            "OutputFilename": "unused",
            "WriteOutput": False,
        }
    )
    runner.run(0.1)
