import equinox as eqx
import jax
from jaxtyping import Array, ArrayLike, Float, Int
import jax.numpy as jnp
import numpy as np
from Integrator import Integrator

"""
Wrapper class for MaNTA State
"""


class State(eqx.Module):
    Variable: Float[ArrayLike, "..."]
    Derivative: Float[ArrayLike, "..."]
    Flux: Float[ArrayLike, "..."]
    Aux: Float[ArrayLike, "..."]
    Scalars: Float[ArrayLike, "..."]

    def __init__(self, Variable_, Derivative_, Flux_, Aux_, Scalars_):
        self.Variable = Variable_
        self.Derivative = Derivative_
        self.Flux = Flux_
        self.Aux = Aux_
        self.Scalars = Scalars_

    @classmethod
    def make_zero(cls, nVars, nAux, nScalars, nPoints):
        zero_ = jnp.zeros((nPoints, nVars))
        zero_aux = jnp.zeros((nPoints, nAux)) if nAux > 0 else None
        zero_scalars = jnp.zeros((nScalars,)) if nScalars > 0 else None
        return cls(
            Variable_=zero_,
            Derivative_=zero_,
            Flux_=zero_,
            Aux_=zero_aux,
            Scalars_=zero_scalars,
        )

    @classmethod
    def from_manta(cls, manta_state):

        return cls(
            Variable_=jnp.array(manta_state["Variable"]),
            Derivative_=jnp.array(manta_state["Derivative"]),
            Flux_=jnp.array(manta_state["Flux"]),
            Aux_=jnp.array(manta_state["Aux"]),
            Scalars_=jnp.array(manta_state["Scalars"]),
        )

    def to_manta(self):
        Scalars_out = []
        if self.Scalars is not None:
            if jnp.ndim(self.Scalars) == 2:
                Scalars_out = self.Scalars[0, :]
            else:
                Scalars_out = self.Scalars

        return {
            "Variable": np.asarray(self.Variable),
            "Derivative": np.asarray(self.Derivative),
            "Flux": np.asarray(self.Flux),
            "Aux": np.asarray(self.Aux),
            "Scalars": np.asarray(Scalars_out),
        }

    @staticmethod
    def vmap_axes():
        return State(0, 0, 0, 0, None)


"""
Decorator functions for converting inputs from the C++ side (dictionaries) to dataclasses for easier use in JAX
"""


def Physics_Decorator(func):
    def wrapper(self, states, positions, *args):
        states_, empty = eqx.partition(State.from_manta(states), lambda x: x.size > 0)
        positions_ = jnp.array(positions)

        result = func(self, states_, positions_, *args)

        for i in range(0, len(result)):
            for j in range(0, len(result[i])):
                if isinstance(result[i][j], State):
                    result[i][j] = eqx.combine(result[i][j], empty).to_manta()
        return result

    return wrapper


def MaNTA_Decorator(func):
    def wrapper(self, index, states, positions, *args):
        states_, empty = eqx.partition(State.from_manta(states), lambda x: x.size > 0)
        positions_ = jnp.array(positions)
        res = func(self, index, states_, positions_, *args)

        if isinstance(res, State):
            return eqx.combine(res, empty).to_manta()
        else:
            return res

    return wrapper


def ScalarG_Decorator(func):
    def wrapper(self, index, states, states_dt, weights, *args):
        states_, empty = eqx.partition(State.from_manta(states), lambda x: x.size > 0)
        states_dt_, empty = eqx.partition(
            State.from_manta(states_dt), lambda x: x.size > 0
        )

        integrator = Integrator(self.k, self.nCells, weights, None, None)
        res = func(self, index, states_, states_dt_, integrator, *args)

        if isinstance(res, State):
            return eqx.combine(res, empty).to_manta()
        else:
            return res

    return wrapper


def ScalarGPrime_Decorator(func):
    def wrapper(self, states, states_dt, weights, phis, phi_boundary, *args):
        states_, empty = eqx.partition(State.from_manta(states), lambda x: x.size > 0)
        states_dt_, empty = eqx.partition(
            State.from_manta(states_dt), lambda x: x.size > 0
        )

        integrator = Integrator(self.k, self.nCells, weights, phis, phi_boundary)
        result = func(self, states_, states_dt_, integrator, *args)

        for i in range(0, len(result)):
            for j in range(0, len(result[i])):
                if isinstance(result[i][j], State):
                    result[i][j] = eqx.combine(result[i][j], empty).to_manta()
        return result

    return wrapper
