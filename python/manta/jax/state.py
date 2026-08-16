import equinox as eqx
import jax
from jaxtyping import Array, ArrayLike, Float, Int
import jax.numpy as jnp
import numpy as np
from .integrator import Integrator

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
        # The pointwise hooks are handed a manta.State -- a named view of one
        # point -- while the batched ones still get a dict of (nPoints, nVars)
        # arrays.
        if hasattr(manta_state, "u"):
            return cls(
                Variable_=jnp.asarray(manta_state.u),
                Derivative_=jnp.asarray(manta_state.q),
                Flux_=jnp.asarray(manta_state.sigma),
                Aux_=jnp.asarray(manta_state.phi),
                Scalars_=jnp.asarray(manta_state.scalars),
            )

        # Scalars stay (nScalars,) in both forms. They used to be broadcast to
        # (nPoints, nScalars) here, because vmap_axes() was a bare 0 and so
        # mapped every field including that one; it now returns a State pytree
        # that maps None over Scalars, which is what they are -- global, not
        # per-point -- so the broadcast is no longer needed to make the shapes
        # line up under vmap.
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
            # A no-op on the (nScalars,) from_manta now produces, and still the
            # thing to do for a State that has been through a vmap that batched
            # the scalars anyway. atleast_2d handles the empty case too --
            # special-casing size == 0 here would leak a 2-D (1, 0) array out to
            # the pointwise State caster, which expects a 1-D Vector.
            Scalars_out = jnp.atleast_2d(self.Scalars)[0, :]
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


def ShiftedState_Decorator(func):
    """MaNTA_Decorator for a hook whose state is one argument later.

    `AuxGPrime(i, out, state, x, t)` is the one left: pointwise-only, and
    carrying an extra argument ahead of the state -- `out`, a buffer it fills
    instead of returning. `dAux_dp(i, pIndex, state, x)` had the same shape and
    was the other user, until dAux/dp moved onto the batched `dAux` and
    PyAdjointProblem began raising from the pointwise hook.

    MaNTA_Decorator's `(self, index, states, positions, *args)` therefore bound
    `states` to the extra argument and `positions` to the state, so the state
    reached jnp.array() -- a TypeError on a manta.State, "dtype object is not a
    valid JAX array type". Both had raised that since the C++ side adopted these
    signatures, and neither had ever been called in anger, because nAux is zero
    in every JAX fixture but one. They were two of the four faults behind the
    JAXAuxTest xfail; see Tests/README.md for the others.

    The extra argument's position is what this pins, not any one hook that has
    it: `test_the_shifted_decorator_converts_the_state_not_the_extra_argument`
    drives a stand-in, because AuxGPrime itself needs a State the Python side
    cannot build.

    The extra argument is passed through untouched, which for AuxGPrime is the
    point: `out` is a window onto solver memory the hook writes through, and a
    converted copy would be discarded when the hook returned.
    """

    def wrapper(self, index, extra, states, positions, *args):
        states_, _ = eqx.partition(State.from_manta(states), lambda x: x.size > 0)
        positions_ = jnp.array(positions)
        # No State ever comes back out -- AuxGPrime returns nothing, and neither
        # did dAux_dp -- so there is nothing to eqx.combine.
        return func(self, index, extra, states_, positions_, *args)

    return wrapper


def ScalarG_Decorator(func):
    def wrapper(self, index, states, states_dt, abscissae, weights, phi_boundary, t):
        del abscissae  # the node positions; a case has its own self.points
        states_, empty = eqx.partition(State.from_manta(states), lambda x: x.size > 0)
        states_dt_, empty = eqx.partition(
            State.from_manta(states_dt), lambda x: x.size > 0
        )

        integrator = Integrator(self.k, self.nCells, weights, phi_boundary)
        res = func(self, index, states_, states_dt_, integrator, t)

        if isinstance(res, State):
            return eqx.combine(res, empty).to_manta()
        else:
            return res

    return wrapper


def InitialScalarDerivative_Decorator(func):
    """The 4-argument shape: PyTransportSystem hands this one nodal data only.

    This is what ScalarG_Decorator used to be, and for a while all three hooks
    shared it. The C++ side then adopted the flat interface, which gave ScalarG
    and ScalarGPrime abscissae, phiBoundary and t; InitialScalarDerivative kept
    the short form, so it needs a decorator of its own rather than the same one.
    """

    def wrapper(self, index, states, states_dt, weights):
        states_, empty = eqx.partition(State.from_manta(states), lambda x: x.size > 0)
        states_dt_, empty = eqx.partition(
            State.from_manta(states_dt), lambda x: x.size > 0
        )

        # No phiBoundary in this signature, so integrator.phiL()/phiR() are not
        # available here -- only the quadrature.
        integrator = Integrator(self.k, self.nCells, weights, None)
        res = func(self, index, states_, states_dt_, integrator)

        if isinstance(res, State):
            return eqx.combine(res, empty).to_manta()
        else:
            return res

    return wrapper


def ScalarGPrime_Decorator(func):
    def wrapper(self, states, states_dt, abscissae, weights, phi_boundary, t):
        del abscissae  # as above
        states_, empty = eqx.partition(State.from_manta(states), lambda x: x.size > 0)
        states_dt_, empty = eqx.partition(
            State.from_manta(states_dt), lambda x: x.size > 0
        )

        integrator = Integrator(self.k, self.nCells, weights, phi_boundary)
        result = func(self, states_, states_dt_, integrator, t)

        for i in range(0, len(result)):
            for j in range(0, len(result[i])):
                if isinstance(result[i][j], State):
                    result[i][j] = eqx.combine(result[i][j], empty).to_manta()
        return result

    return wrapper
