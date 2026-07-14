from typing import NamedTuple


import MaNTA
from VectorizedTransportSystem import VectorizedTransportSystem
import jax.numpy as jnp
from functools import partial

import numpy as np
import jax
import equinox as eqx
from State import State, ScalarGPrime_Decorator, ScalarG_Decorator


class ScalarLDParams(NamedTuple):
    gamma: float
    gamma_d: float
    gamma_I: float
    u0: float
    beta: float
    kappa: float
    alpha: float

    @classmethod
    def make(cls, config) -> "ScalarLDParams":
        return cls(
            gamma=1.0,
            gamma_d=-0.1,
            gamma_I=-0.5,
            u0=0.1,
            beta=1.0,
            kappa=1.0,
            alpha=0.2,
        )


class ScalarLD(VectorizedTransportSystem):
    def __init__(self, config):
        super().__init__()
        self.nVars = 1
        self.nScalars = 3
        self.isUpperDirichlet = True
        self.isLowerDirichlet = True

        solver_config = {
            "OutputFilename": "out",
            "Polynomial_degree": 4,
            "Grid_size": 21,
            "tau": 0.1,
            "Lower_boundary": -1.0,
            "Upper_boundary": 1.0,
            "Relative_tolerance": 1e-3,
            "MinStepSize": 1e-10,
            "delta_t": 0.1,
        }

        self.xL = solver_config["Lower_boundary"]
        self.xR = solver_config["Upper_boundary"]
        self.nCells = solver_config["Grid_size"]
        self.k = solver_config["Polynomial_degree"]

        self.params = ScalarLDParams.make(config)
        self.points = MaNTA.getNodes(
            solver_config["Lower_boundary"],
            solver_config["Upper_boundary"],
            solver_config["Grid_size"],
            solver_config["Polynomial_degree"],
        )

        self.nPoints = len(self.points)
        self.M0 = 2 * self.params.u0 + 4 * self.params.beta / jnp.pi
        self.runner = MaNTA.Runner(self)

        self.runner.configure(solver_config)

        # This object will be passed to sigma and source functions

    def run(self, tFinal=None):
        if tFinal is not None:
            sFinal = self.runner.run(tFinal)
        else:
            sFinal = self.runner.run_ss()

    def sigma(self, index, state, x, t, params: ScalarLDParams):
        return params.kappa * state.Derivative[index]

    def source(self, index, state, x, t, params: ScalarLDParams):

        J = state.Scalars[1]
        return J * self.ScaledSource(x, params) + 0.5 * jnp.cos(jnp.pi * x)

    @ScalarG_Decorator
    @partial(jax.jit, static_argnames=("self",))
    def ScalarG(self, i, states, states_dot, integrator, t):
        E = states.Scalars[0]
        J = states.Scalars[1]
        I = states.Scalars[2]

        dEdt = states_dot.Scalars[0]
        dIdt = states_dot.Scalars[2]

        M = integrator(states.Variable[:, 0])

        def i0():
            return E - (self.M0 - M)

        def i1():
            return (
                J
                - self.params.gamma * E
                - self.params.gamma_d * dEdt
                - self.params.gamma_I * I
                + (states.Flux[0, 0] - states.Flux[-1, 0])
            )

        def i2():
            return dIdt - E

        return jax.lax.switch(i, [i0, i1, i2])

    @ScalarGPrime_Decorator
    @partial(jax.jit, static_argnames=("self",))
    def ScalarGPrime(self, states, states_dot, integrator, t):

        sArgs = (self.nVars, self.nAux, self.nScalars, len(self.points))

        # Scalar 0

        varshape = states.Variable.shape
        _zeros = jnp.zeros(varshape)
        _variable = -integrator.computeCellProducts(jnp.ones((self.nPoints,)))

        _scalar = jnp.array([1.0, 0.0, 0.0])

        derivs0 = State(_variable, _zeros, _zeros, None, _scalar)
        derivs_dt0 = State.make_zero(*sArgs)

        # Scalar 1

        _flux_L = integrator.phiL()
        _flux_R = -integrator.phiR()

        _flux = jnp.concatenate(
            [_flux_L, jnp.zeros(((self.nCells - 2) * (self.k + 1),)), _flux_R]
        )
        _scalar = jnp.array([-self.params.gamma, 1.0, -self.params.gamma_I])
        _scalar_dt = jnp.array([-self.params.gamma_d, 0.0, 0.0])

        derivs1 = State(_zeros, _zeros, _flux, None, _scalar)
        derivs_dt1 = State(_zeros, _zeros, _zeros, None, _scalar_dt)

        # Scalar 2

        _scalar = np.array([-1.0, 0.0, 0.0])
        _scalar_dt = jnp.array([0.0, 0.0, 1.0])

        derivs2 = State(_zeros, _zeros, _zeros, None, _scalar)
        derivs_dt2 = State(_zeros, _zeros, _zeros, None, _scalar_dt)

        out = [[derivs0, derivs1, derivs2], [derivs_dt0, derivs_dt1, derivs_dt2]]
        return out

    @partial(jax.jit, static_argnames=("self",))
    def ScaledSource(self, x, params):
        Ainv = (
            params.alpha * jnp.sqrt(jnp.pi) * jax.scipy.special.erf(1.0 / params.alpha)
        )
        return jnp.exp(-((x / params.alpha) ** 2)) / Ainv

    @partial(jax.jit, static_argnames=("self",))
    def dSources_dScalars(self, i, state, x, t):
        return jnp.array([0.0, self.ScaledSource(x, self.params), 0.0])

    @partial(jax.jit, static_argnames=("self",))
    def LowerBoundary(self, index, t):
        return self.params.u0

    @partial(jax.jit, static_argnames=("self",))
    def UpperBoundary(self, index, t):
        return self.params.u0

    def InitialValue(self, index, x):
        return self.params.u0 + self.params.beta * jnp.cos(jnp.pi * x / 2.0)

    @partial(jax.jit, static_argnames=("self",))
    def InitialScalarValue(self, s):

        def i0():
            return 0.0

        def i1():
            return -self.params.kappa * (
                self.InitialDerivative(0, self.xR) - self.InitialDerivative(0, self.xL)
            )

        def i2():
            return 0.0

        return jax.lax.switch(s, [i0, i1, i2])

    @ScalarG_Decorator
    @partial(jax.jit, static_argnames=("self",))
    def InitialScalarDerivative(self, i, state, state_dot, integrator):

        E = state.Scalars[0]
        mdot = integrator(state_dot.Variable[:, 0])

        def i0():
            return mdot

        def i2():
            return E

        return jax.lax.switch(i, [i0, i2])

    @partial(jax.jit, static_argnames=("self",))
    def isScalarDifferential(self, s):
        def i0():
            return True

        def i2():
            return False

        return jax.lax.switch(s, [i0, i2])


def runMaNTA():
    config = {
        "SourceCentre": 0.3,
        "D": 2.0,
        "a": 0.0,
    }
    transportSystem = ScalarLD(config)

    transportSystem.run()
    # transportSystem.setParams(LinearDiffusionParams(0.1, 0.1, 2.0, 1.0)


runMaNTA()
