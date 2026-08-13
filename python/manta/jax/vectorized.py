import jax
import jax.numpy as jnp
import numpy as np
import manta as MaNTA
from .adjoint_problem import JAXAdjointProblem
from typing import NamedTuple, Any
from functools import partial
from .state import Physics_Decorator, State
from abc import abstractmethod
import equinox as eqx

"""
JAX-based transport system base class that overloads MaNTA TransportSystem.
Enables automatic differentiation of sigma and source terms using JAX.
"""


# Base class for JAX-based transport systems
class VectorizedTransportSystem(MaNTA.TransportSystem):
    def __init__(self, spec, spatialParameters=False):
        MaNTA.TransportSystem.__init__(self, spec)

        self.dAuxdvars = jax.jit(jax.grad(self.aux, argnums=1))

        self.dInitialValue = jax.jit(jax.grad(self.InitialValue, argnums=1))

        self.vmap_axes = (State.vmap_axes(), 0, 0 if spatialParameters else None)

    @abstractmethod
    def LowerBoundary(self, index, t):
        raise NotImplementedError("LowerBoundary not implemented in derived class")

    @abstractmethod
    def UpperBoundary(self, index, t):
        raise NotImplementedError("UpperBoundary not implemented in derived class")

    """
    Base methods for vectorizing 
    """

    @Physics_Decorator
    def ComputePhysics(self, states, positions, t):
        index = jnp.arange(0, self.nVars)
        fluxes = []
        sources = []
        aux = []
        for i in index:
            fluxes.append(self.SigmaFn_v(i, states, positions, t))
            sources.append(self.Sources_v(i, states, positions, t))

        for i in range(0, self.nAux):
            aux.append(self.AuxG_v(i, states, positions, t))

        return [fluxes, sources, aux]

    @Physics_Decorator
    def ComputePhysicsDerivatives(self, states, positions, t):
        index = jnp.arange(0, self.nVars)
        fluxes = []
        sources = []
        aux = []
        for i in index:
            fluxes.append(self.dSigma(i, states, positions, t))
            sources.append(self.dSources(i, states, positions, t))

        for i in range(0, self.nAux):
            aux.append(self.AuxGPrime_v(i, states, positions, t))

        return [fluxes, sources, aux]

    @partial(jax.jit, static_argnames=("self",))
    def SigmaFn_v(self, index, states, positions, t):
        return jax.vmap(
            lambda s, p, params: self.sigma(index, s, p, t, params),
            in_axes=(self.vmap_axes),
        )(states, positions, self.params)

    @partial(jax.jit, static_argnames=("self",))
    def Sources_v(self, index, states, positions, t):
        return jax.vmap(
            lambda s, p, params: self.source(index, s, p, t, params),
            in_axes=(self.vmap_axes),
        )(states, positions, self.params)

    @partial(jax.jit, static_argnames=("self",))
    def AuxG_v(self, index, states, positions, t):
        return jax.vmap(
            lambda s, p, params: self.aux(index, s, p, t, params),
            in_axes=(self.vmap_axes),
        )(states, positions, self.params)

    @partial(jax.jit, static_argnames=("self",))
    def dSigma(self, index, states, positions, t):
        return jax.vmap(
            lambda s, p, params: jax.grad(self.sigma, argnums=1)(
                index, s, p, t, params
            ),
            in_axes=(self.vmap_axes),
        )(states, positions, self.params)

    @partial(jax.jit, static_argnames=("self",))
    def dSources(self, index, states, positions, t):
        return jax.vmap(
            lambda s, p, params: jax.grad(self.source, argnums=1)(
                index, s, p, t, params
            ),
            in_axes=(self.vmap_axes),
        )(states, positions, self.params)

    @partial(jax.jit, static_argnames=("self",))
    def AuxGPrime_v(self, index, states, positions, t):
        return jax.vmap(
            lambda s, p, params: jax.grad(self.aux, argnums=1)(index, s, p, t, params),
            in_axes=(self.vmap_axes),
        )(states, positions, self.params)

    @abstractmethod
    def ScalarG(self, i, states, states_dot, abscissae, weights, phi_boundary, t):
        raise NotImplementedError("Scalar G function not implemented in derived class")

    @abstractmethod
    def ScalarGPrime(self, states, states_dot, abscissae, weights, phi_boundary, t):
        """d G_s / d state and d G_s / d state_dot, for every scalar.

        Returns a pair of lists, each of nScalars GlobalState dicts: the first is
        the derivative with respect to the state, the second with respect to its
        time derivative (identically zero for an algebraic scalar).

        `weights` is one quadrature weight per node, length nCells*(k+1), so an
        integral over the domain is `weights @ u`. `phi_boundary` is (k+1, 2), the
        basis functions evaluated at the two ends of the domain.

        The signature here previously declared an extra `phis` parameter before
        `phi_boundaries`. PyTransportSystem::ScalarGPrimeExtended passes five
        arguments, not six, so any subclass that followed this declaration would
        have failed with a TypeError on the first Jacobian evaluation. Nothing
        caught it because the abstract method only ever raised, and there was no
        Python test with a scalar -- see python/Tests/test_scalars.py.
        """
        raise NotImplementedError("ScalarGPrime not implemented in derived class")

    """
    Sigma and source, and auxiliary functions to be overloaded in derived classes

    Parameters
    ----------
    index : int
        Variable index
    state : dict
        Dictionary containing "Variable", "Derivative, "Flux", "Aux", and "Scalar" arrays
    x : float
        Spatial location
    t : float
        Time
    params : NamedTuple
        Transport system parameters, passed for JAX PyTree compatibility
    Returns
    -------
    float
        Computed sigma, source, or aux term
    """

    @abstractmethod
    def sigma(self, index, state, x, t, params):
        raise NotImplementedError("sigma function not implemented in derived class")

    @abstractmethod
    def source(self, index, state, x, t, params):
        raise NotImplementedError("source function not implemented in derived class")

    @abstractmethod
    def aux(self, index, state, x, t, params):
        raise NotImplementedError("aux function not implemented in derived class")

    @abstractmethod
    def InitialValue(self, index, x):
        raise NotImplementedError("InitialValue must be implemented in derived class")

    @abstractmethod
    def InitialScalarValue(self, s):
        raise NotImplementedError("InitialScalarValue not implemented in derived class")

    @abstractmethod
    @partial(jax.jit, static_argnames=("self",))
    def InitialDerivative(self, index, x):
        return self.dInitialValue(index, x)

    @abstractmethod
    def InitialAuxValue(self, index, x):
        raise NotImplementedError("InitialAuxValue not implemented in derived class")

    @abstractmethod
    def InitialScalarDerivative(self, i, states, states_dot, integrator):
        raise NotImplementedError(
            "InitialScalarDerivative not implemented in derived class"
        )

    @abstractmethod
    def isScalarDifferential(self, s) -> bool:
        raise NotImplementedError(
            "isScalarDifferential not implemented in derived class"
        )

    """
    Create the adjoint problem associated with this transport system
    
    Returns
    -------
    JAXAdjointProblem
        The adjoint problem object
    """

    @abstractmethod
    def createAdjointProblem(self):
        raise NotImplementedError(
            "createAdjointProblem not implemented in derived class"
        )
