import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import manta as MaNTA
from functools import partial

from .integrator import scalar_integrator
from .state import State, MaNTA_Decorator, Physics_Decorator, ShiftedState_Decorator

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
        # dAux() differentiates this the way dSigma()/dSources() do the other
        # two. It was the one of the three never bound, so the batched aux
        # branch of ComputePhysicsDerivatives raised AttributeError -- reachable
        # only with nAux > 0, which no other JAX fixture has.
        self.aux = transport_system.aux

        self.daux_dp = jax.jit(jax.grad(transport_system.aux, argnums=4))

        # The scalar constraints, when the case writes them as a function of
        # the parameters; see manta.jax.scalars. None for a case that writes
        # ScalarG itself, which leaves dScalarG_dp nothing to differentiate.
        self.transport_system = transport_system
        self.nScalars = transport_system.nScalars
        self.scalar = getattr(transport_system, "scalar", None)

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

        # Rows only. A scalar pair pads *every* axis, which would add a spurious
        # point column as well as the boundary-parameter row -- and the extra
        # column is invisible downstream, since the non-spatial branch of
        # PyAdjointProblem::dGFndp reads dgdp(p, ind) with ind stopping at the
        # last real node.
        out = jnp.pad(
            g,
            pad_width=((0, self.np_boundary), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        return out

    @MaNTA_Decorator
    def dg(self, i, states, positions):
        out = self._dg_pointwise(states, positions)
        if self.nScalars > 0:
            # The pointwise dg/dmu, integrated: MaNTA wants dG/dmu, the
            # derivative of the objective itself, and G is weights . g over
            # these points.
            weights = jnp.asarray(self._objective_weights(len(positions)))
            out = eqx.tree_at(lambda s: s.Scalars, out, weights @ out.Scalars)
        return out

    @eqx.filter_jit
    def _dg_pointwise(self, states, positions):
        return jax.vmap(
            jax.grad(self.g, argnums=0), in_axes=(State.vmap_axes(), 0, self.param_axis)
        )(states, positions, self.params)

    def _objective_weights(self, nPoints):
        """The quadrature G is integrated with, over the physics nodes of the
        level the case was last handed a plan for."""
        plan = self.transport_system.evaluationPlan
        weights = np.asarray(
            MaNTA.getPhysicsWeights(plan.grid.cellBoundaries(), plan.k, plan.superconvergent)
        )
        if len(weights) != nPoints:
            raise RuntimeError(
                f"dg was handed {nPoints} points, but the evaluation plan's physics "
                f"nodes number {len(weights)}"
            )
        return weights

    def dScalarG_dp(self, s, states, states_dot, abscissae, weights, phi_boundary, t):
        """dG_s/dp, the gradient of the case's `scalar` in its parameters.

        A length-np vector, or (np, nPoints) with spatial parameters: ravelled
        exactly as dgFndp ravels, one row per internal parameter, with zero rows
        for the boundary parameters a scalar constraint cannot read.
        """
        if self.scalar is None:
            raise NotImplementedError(
                f"{type(self.transport_system).__name__} has global scalars but writes "
                "ScalarG itself, so their parameter derivative cannot be derived: "
                "define scalar(index, states, states_dot, integrator, t, params) on "
                "it instead, or override dScalarG_dp on its adjoint problem."
            )
        states_, _ = eqx.partition(State.from_manta(states), lambda x: x.size > 0)
        states_dot_, _ = eqx.partition(State.from_manta(states_dot), lambda x: x.size > 0)
        integrator = scalar_integrator(
            self.transport_system, jnp.asarray(weights), jnp.asarray(phi_boundary)
        )
        dp = jax.grad(self.scalar, argnums=5)(
            s, states_, states_dot_, integrator, t, self.params
        )
        flat, _ = ravel_pytree(dp)
        nInternal = self.np - self.np_boundary
        if self.spatialParameters:
            out = jnp.pad(
                jnp.reshape(flat, (nInternal, -1)), pad_width=((0, self.np_boundary), (0, 0))
            )
        else:
            out = jnp.pad(flat, pad_width=(0, self.np_boundary))
        return np.asarray(out, dtype=float)

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
