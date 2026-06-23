import jax
import jax.numpy as jnp
import numpy as np
import MaNTA
from JAXAdjointProblem import JAXAdjointProblem
from typing import NamedTuple, Any
from functools import partial
from State import State, MaNTA_Decorator, MaNTA_Decorator2
from abc import abstractmethod
import equinox as eqx

"""
JAX-based transport system base class that overloads MaNTA TransportSystem.
Enables automatic differentiation of sigma and source terms using JAX.
"""


# Base class for JAX-based transport systems
class VectorizedTransportSystem(MaNTA.TransportSystem):
    def __init__(self, spatialParameters=False):
        MaNTA.TransportSystem.__init__(self)
        self.nAux = 0

        self.dAuxdvars = jax.jit(jax.grad(self.aux, argnums=1))

        self.dInitialValue = jax.jit(jax.grad(self.InitialValue, argnums=1))

        self.vmap_axes = (State.vmap_axes(), 0, 0 if spatialParameters else None)

    def LowerBoundary(self, index, t):
        pass

    def UpperBoundary(self, index, t):
        pass

    @MaNTA_Decorator2
    def ComputePhysics(self, states, positions, t):
        index = jnp.arange(0, self.nVars)
        fluxes = []
        sources = []
        for i in index:
            fluxes.append(self.SigmaFn_v(i, states, positions, t))
            sources.append(self.Sources_v(i, states, positions, t))

        return [fluxes, sources]
    
    @MaNTA_Decorator2
    def ComputePhysicsDerivatives(self, states, positions, t):
        index = jnp.arange(0, self.nVars)
        fluxes = []
        sources = []
        for i in index:
            fluxes.append(self.dSigma(i, states, positions, t))
            sources.append(self.dSources(i, states, positions, t))

        return [fluxes, sources]
 
    def SigmaFn_v(self, index, states, positions, t):
        return jax.vmap(
            lambda s, p, params: self.sigma(index, s, p, t, params),
            in_axes=(self.vmap_axes),
        )(states, positions, self.params)

    def Sources_v(self, index, states, positions, t):
        return jax.vmap(
            lambda s, p, params: self.source(index, s, p, t, params),
            in_axes=(self.vmap_axes),
        )(states, positions, self.params)

    def dSigma(self, index, states, positions, t):
        return jax.vmap(
            lambda s, p, params: jax.grad(self.sigma, argnums=1)(
                index, s, p, t, params
            ),
            in_axes=(self.vmap_axes),
        )(states, positions, self.params)

    def dSources(self, index, states, positions, t):
        return jax.vmap(
            lambda s, p, params: jax.grad(self.source, argnums=1)(
                index, s, p, t, params
            ),
            in_axes=(self.vmap_axes),
        )(states, positions, self.params)

    """
    Sigma and source, and auxilliary functions to be overloaded in derived classes

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
        Computed sigma or source term
    """

    @abstractmethod
    @partial(jax.jit, static_argnames=("self",))
    def sigma(self, index, state, x, t, params: NamedTuple):
        raise NotImplementedError("sigma function not implemented")

    @abstractmethod
    @partial(jax.jit, static_argnames=("self",))
    def source(self, index, state, x, t, params: NamedTuple):
        raise NotImplementedError("source function not implemented")

    @abstractmethod
    @partial(jax.jit, static_argnames=("self",))
    def aux(self, index, state, x, t, params: NamedTuple):
        pass

    @MaNTA_Decorator
    def AuxG(self, index, state, x, t):
        return self.aux(index, state, x, t, self.params)

    """
    Compute derivative of auxilliary functions

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
    Returns
    -------
    state : dict
        Dictionary containing "Variable", "Derivative, "Flux", "Aux", and "Scalar" arrays
    """

    @MaNTA_Decorator
    def dSigma_dPhi(self, index, state, x, t):
        return jax.grad(self.sigma, argnums=1)(index, state, x, t, self.params).Aux

    @MaNTA_Decorator
    def dSources_dPhi(self, index, state, x, t):
        return jax.grad(self.source, argnums=1)(index, state, x, t, self.params).Aux

    @MaNTA_Decorator
    def AuxGPrime(self, index, state, x, t):
        return self.dAuxdvars(index, state, x, t, self.params)

    @abstractmethod
    @partial(jax.jit, static_argnames=("self",))
    def InitialValue(self, index, x):
        raise NotImplementedError("InitialValue must be implemented in derived class")

    @abstractmethod
    @partial(jax.jit, static_argnames=("self",))
    def InitialDerivative(self, index, x):
        return self.dInitialValue(index, x)

    def InitialAuxValue(self, index, x):
        pass

    """
    Create the adjoint problem associated with this transport system
    
    Returns
    -------
    JAXAdjointProblem
        The adjoint problem object
    """

    @abstractmethod
    def createAdjointProblem(self):
        pass


# Need PyTree structure for class paramters to be able to compute adjoints


class NonlinearDiffusionParams(NamedTuple):
    SourceCentre: float
    D: float
    T_s: float
    a: float
    SourceWidth: float

    @classmethod
    def make(cls, config: MaNTA.TomlValue) -> "NonlinearDiffusionParams":
        return cls(
            SourceCentre=config["SourceCentre"],
            D=config["D"],
            T_s=50.0,
            a=config["a"],
            SourceWidth=0.02,
        )


class JAXNonlinearDiffusion(VectorizedTransportSystem):
    def __init__(self, config: MaNTA.TomlValue, grid: MaNTA.Grid):
        super().__init__()
        self.nVars = 1
        self.isUpperDirichlet = True
        self.isLowerDirichlet = False

        # This object will be passed to sigma and source functions
        self.params = NonlinearDiffusionParams.make(config)

    def g(self, state, x, params: NonlinearDiffusionParams):
        u = state.Variable[0]
        return 0.5 * u * u

    def sigma(self, index, state, x, t, params: NonlinearDiffusionParams):

        u = state.Variable[0]
        q = state.Derivative[0]
        return params.D * (u**params.a) * q

    def source(self, index, state, x, t, params: NonlinearDiffusionParams):
        y = x - params.SourceCentre
        return params.T_s * jnp.exp(-y * y / params.SourceWidth)

    def LowerBoundary(self, index, t):
        return 0.0

    def UpperBoundary(self, index, t):
        return 0.3

    def InitialValue(self, index, x):
        return 0.3

    def createAdjointProblem(self):
        adjointProblem = JAXAdjointProblem(self, self.g)
        adjointProblem.addUpperBoundarySensitivity(0)
        return adjointProblem


class JAXAuxTest(VectorizedTransportSystem):
    def __init__(self, config: MaNTA.TomlValue, grid: MaNTA.Grid):
        super().__init__()
        self.nVars = 1
        self.nAux = 1
        self.isUpperDirichlet = True
        self.isLowerDirichlet = False

        # This object will be passed to sigma and source functions
        self.params = NonlinearDiffusionParams.make(config)

    def g(self, state, x, params: NonlinearDiffusionParams):
        u = state.Variable[0]
        return 0.5 * u * u * params.D

    def sigma(self, index, state, x, t, params: NonlinearDiffusionParams):

        u = state.Variable[0]
        q = state.Derivative[0]
        return params.D * (u**params.a) * q

    def aux(self, index, state, x, t, params):
        a = state.Aux[0]
        u = state.Variable[0]
        return a - params.D * u * u

    def source(self, index, state, x, t, params: NonlinearDiffusionParams):
        y = x - params.SourceCentre
        u = state.Variable[0]
        a = state.Aux[0]
        return params.T_s * jnp.exp(-y * y / params.SourceWidth) + a - params.D * u * u

    def LowerBoundary(self, index, t):
        return 0.0

    def UpperBoundary(self, index, t):
        return 0.3

    def InitialValue(self, index, x):
        return 0.3

    def InitialAuxValue(self, index, x):
        u0 = self.InitialValue(index, x)
        return self.params.D * u0 * u0

    def createAdjointProblem(self):
        adjointProblem = JAXAdjointProblem(self, self.g)
        adjointProblem.addUpperBoundarySensitivity(0)
        return adjointProblem


def registerTransportSystems():

    MaNTA.registerPhysicsCase("JAXNonlinearDiffusion", JAXNonlinearDiffusion)
    MaNTA.registerPhysicsCase("JAXAuxTest", JAXAuxTest)
