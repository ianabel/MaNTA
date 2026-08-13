import jax
import jax.numpy as jnp
import equinox as eqx
import manta as MaNTA
from functools import partial

from .state import State, MaNTA_Decorator, Physics_Decorator

from jax.flatten_util import ravel_pytree


class JAXAdjointProblem(MaNTA.AdjointProblem):
    def __init__(
        self, transport_system: MaNTA.TransportSystem, g, spatialParameters=False
    ):
        MaNTA.AdjointProblem.__init__(self)
        self.params = transport_system.params
        self.g = g
        self.spatialParameters = spatialParameters
        self.ng = 1

        self.nVars = transport_system.nVars
        self.nAux = transport_system.nAux
        self.np = len(transport_system.params)
        self.np_boundary = 0

        self.sigma = transport_system.sigma
        self.source = transport_system.source

        self.daux_dp = jax.jit(jax.grad(transport_system.aux, argnums=4))

        self.UpperBoundarySensitivities = {}
        self.LowerBoundarySensitivities = {}
        self.param_axis = 0 if self.spatialParameters else None

        self.vmap_axes = (None, State.vmap_axes(), 0, None, self.param_axis)

    def setParams(self, params):
        self.params = params

    @MaNTA_Decorator
    @eqx.filter_jit
    def gFn(self, i, states, positions):

        out = jax.vmap(self.g, in_axes=(State.vmap_axes(), 0, self.param_axis))(
            states, positions, self.params
        )

        return out

    @MaNTA_Decorator
    @eqx.filter_jit
    def dgFndp(self, gIndex, states, positions):
        dgdp = jax.vmap(
            jax.grad(self.g, argnums=2), in_axes=(State.vmap_axes(), 0, self.param_axis)
        )(states, positions, self.params)
        g, _ = ravel_pytree(dgdp)
        g = jnp.reshape(g, (self.np - self.np_boundary, len(positions)))

        out = jnp.pad(
            g, pad_width=(0, self.np_boundary), mode="constant", constant_values=0
        )

        return out

    @MaNTA_Decorator
    @eqx.filter_jit
    def dg(self, i, states, positions):
        out = jax.vmap(
            jax.grad(self.g, argnums=0), in_axes=(State.vmap_axes(), 0, self.param_axis)
        )(states, positions, self.params)
        return out

    @Physics_Decorator
    def ComputePhysicsDerivatives(self, states, positions):
        index = jnp.arange(0, self.nVars)
        fluxes = []
        sources = []
        aux = []
        for i in index:
            fluxes.append(self.dSigma(i, states, positions))
            sources.append(self.dSources(i, states, positions))

        for i in range(0, self.nAux):
            aux.append(self.dAux(i, states, positions))
        return [fluxes, sources, aux]

    @eqx.filter_jit
    def dSigma(self, i, states, positions):
        out = jax.vmap(jax.grad(self.sigma, argnums=4), in_axes=(self.vmap_axes))(
            i, states, positions, 0.0, self.params
        )
        return out

    @eqx.filter_jit
    def dSources(self, i, states, positions):
        out = jax.vmap(jax.grad(self.source, argnums=4), in_axes=(self.vmap_axes))(
            i, states, positions, 0.0, self.params
        )
        return out

    @eqx.filter_jit
    def dAux(self, i, states, positions):
        out = jax.vmap(jax.grad(self.aux, argnums=4), in_axes=(self.vmap_axes))(
            i, states, positions, 0.0, self.params
        )
        return out

    @partial(jax.jit, static_argnums=(0,))
    def dgFn_dphi(self, i, state, x):
        return jax.grad(self.g, argnums=0)(state, x, self.params)["Aux"]

    def dAux_dp(self, index, pIndex, state, x):
        return self.daux_dp(index, state, x, 0.0, self.params)[pIndex]

    def computeUpperBoundarySensitivity(self, i, pIndex):
        if (i, pIndex) in self.UpperBoundarySensitivities:
            return True
        else:
            return False

    def computeLowerBoundarySensitivity(self, i, pIndex):
        if (i, pIndex) in self.LowerBoundarySensitivities:
            return True
        else:
            return False

    def getName(self, pIndex):
        if pIndex < len(self.params):
            return list(self.params._fields)[pIndex]
        else:
            return "BoundaryCondition" + str(pIndex)

    def addUpperBoundarySensitivity(self, i):
        self.UpperBoundarySensitivities[(i, self.np)] = True
        self.np += 1
        self.np_boundary += 1

    def addLowerBoundarySensitivity(self, i):
        self.LowerBoundarySensitivities[(i, self.np)] = True
        self.np += 1
        self.np_boundary += 1
