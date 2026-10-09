"""tauScaling = "Diffusive", from Python.

The formula and the Jacobian wiring are covered in C++
(Tests/UnitTests/LocalTauTests.cpp). What this file covers:

  * the effect the option exists for, on the toy problem that motivated it:
    -d/dx[2 x D u^n u'] = S with u^n small at a Dirichlet wall, where a constant
    tau puts tau * (u_h - u_b) into sigma_h at the wall and a Diffusive one does
    not. The closed form of the steady flux is known, so both are compared
    against it rather than against each other;
  * both TauKappa sources -- the cell's nodal d sigma_hat / d q extrapolated
    to its faces, and the face-point derivative call -- each reached by a
    pointwise Python case through the trampoline's per-point dSigmaFn_dq;
  * tauUpdate = "ContinuationStep" converging to the same steady state as
    "Residual" -- the two differ in how tau is updated on the way, not in the
    equations at the end -- and being refused for a time march;
  * the refusals reaching Python as ValueError. Diffusive with solveAdjoint is
    refused too, and tested in C++ (ConfigSourceTests.cpp): from here a case
    needs an adjoint problem before the configuration is looked at.
"""

import numpy as np
import pytest
from netCDF4 import Dataset
from scipy.special import erf

import manta

# See WallLayer.SigmaFn: the NaN from a negative iterate is the point.
pytestmark = pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")

H, W, D, UB, N = 0.1, 0.2, 0.01, 0.2, 2.5


def sigma_exact(x):
    return H * np.sqrt(np.pi * W) / 2 * erf(x / np.sqrt(W))


class WallLayer(manta.TransportSystem):
    """sigma_hat = 2 x D u^n q, S = H exp(-x^2/W), zero flux at x = 0, u(1) = UB."""

    variables = [manta.Field("u", "", "", lower=manta.Mixed(d=1.0))]

    # u^n in numpy floats, so a Newton iterate below zero gives NaN -- which the
    # solver rejects as a failed step -- rather than Python's complex power, or
    # an |u|^n that lets the iterate wander on into u < 0. That is what the JAX
    # version of this case does, and what lets it converge.
    def SigmaFn(self, i, s, x, t):
        return 2 * x * D * np.float64(s.u[0]) ** N * s.q[0]

    def dSigmaFn_du(self, i, s, x, t):
        return np.array([2 * x * D * N * np.float64(s.u[0]) ** (N - 1) * s.q[0]])

    def dSigmaFn_dq(self, i, s, x, t):
        return np.array([2 * x * D * np.float64(s.u[0]) ** N])

    def Sources(self, i, s, x, t):
        return H * np.exp(-x * x / W)

    def LowerBoundary(self, i, t):
        return 0.0

    def UpperBoundary(self, i, t):
        return UB

    def InitialValue(self, i, x):
        return UB

    def InitialDerivative(self, i, x):
        return 0.0


def config(stem, **overrides):
    c = {
        "Polynomial_degree": 4,
        "Grid_size": 5,
        "Lower_boundary": 0.0,
        "Upper_boundary": 1.0,
        "Relative_tolerance": 1e-6,
        "Absolute_tolerance": [1e-6],
        "initialTimestep": 1e-3,
        "MinStepSize": 1e-12,
        "SteadyStateSolver": "PseudoTransient",
        "SteadyStateTolerance": 1e-5,
        "delta_t": 1.0,
        "OutputFilename": stem,
    }
    c.update(overrides)
    return c


def steady_sigma(stem, **overrides):
    runner = manta.Runner(WallLayer())
    runner.configure(config(stem, **overrides))
    runner.run_ss()
    with Dataset(stem + ".nc") as nc:
        return np.array(nc.groups["u"].variables["sigma"][-1, :])


def wall_flux_error(stem, **overrides):
    return abs(steady_sigma(stem, **overrides)[-1] - sigma_exact(1.0))


@pytest.mark.parametrize("kappa", ["Nodal", "Face"])
def test_diffusive_tau_keeps_the_wall_flux_out_of_the_penalty(kappa):
    # Measured: 2.4e-2 at tau = 1 constant, against a flux of 3.96e-2 -- the
    # oscillation in sigma_h that the toy model was showing -- and, with the
    # same multiplier under Diffusive scaling, 1.9e-4 with kappa read at the
    # faces and 2.4e-4 with it extrapolated from the nodes.
    constant = wall_flux_error("wall_constant", tau=1.0)
    diffusive = wall_flux_error("wall_diffusive_" + kappa, tau=1.0, tauScaling="Diffusive",
                                TauKappa=kappa)
    assert constant > 1e-2, constant
    assert diffusive < 1e-3, diffusive


@pytest.mark.parametrize("kappa", ["Nodal", "Face"])
@pytest.mark.parametrize("update", ["ContinuationStep", "JacobianBuild"])
def test_a_tau_held_fixed_reaches_the_same_steady_state(update, kappa):
    diffusive = dict(tau=1.0, tauScaling="Diffusive", TauKappa=kappa, SteadyStateTolerance=1e-9)
    every = steady_sigma("wall_every_residual_" + kappa, tauUpdate="Residual", **diffusive)
    held = steady_sigma("wall_held_" + update + "_" + kappa, tauUpdate=update, **diffusive)
    assert np.max(np.abs(every - held)) < 1e-7 * np.max(np.abs(every))


def test_a_tau_frozen_per_step_is_refused_for_a_time_march():
    runner = manta.Runner(WallLayer())
    runner.configure(config("wall_march", tau=1.0, tauScaling="Diffusive",
                            tauUpdate="ContinuationStep", SteadyStateSolver="TimeMarch"))
    with pytest.raises(ValueError, match="ContinuationStep"):
        runner.run(1.0)


@pytest.mark.parametrize("overrides, needle", [
    ({"tauScaling": "Local"}, "tauScaling"),
    ({"tauScaling": "Diffusive", "tauFloor": 0.0}, "tauFloor"),
    ({"tauScaling": "Diffusive", "tauUpdate": "Newton"}, "tauUpdate"),
    ({"tauUpdate": "ContinuationStep"}, "Diffusive"),
    ({"tauScaling": "Diffusive", "TauKappa": "Trace"}, "TauKappa"),
    ({"TauKappa": "Face"}, "Diffusive"),
])
def test_bad_tau_scaling_configurations_are_refused(overrides, needle):
    runner = manta.Runner(WallLayer())
    with pytest.raises(ValueError, match=needle):
        runner.configure(config("refused", **overrides))
