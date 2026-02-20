from typing import NamedTuple

import MaNTA

import os
os.environ.pop("LD_LIBRARY_PATH", None)

# from JAXTransportSystem import JAXTransportSystem
# from JAXAdjointProblem import JAXAdjointProblem
import jax.numpy as jnp
import jax

from functools import partial

from yancc_wrapper import yancc_wrapper 

class StellaratorParams(NamedTuple):
    SourceCenter: float
    SourceHeight: float
    SourceWidth: float
    EdgeTemperature: float
    EdgeDensity: float
    n0: float

    @classmethod
    def from_config(cls, config: MaNTA.TomlValue):
        return cls(
            SourceCenter=config["SourceCenter"],
            SourceHeight=config["SourceHeight"],
            SourceWidth=config["SourceWidth"],
            EdgeTemperature=config["EdgeTemperature"],
            EdgeDensity=config["EdgeDensity"],
            n0=config["n0"]
        )

class StellaratorTransport(MaNTA.TransportSystem): 
    def __init__(self, config: MaNTA.TomlValue, grid: MaNTA.Grid):
        MaNTA.TransportSystem.__init__(self)
        self.nVars = 1
        self.params = StellaratorParams.from_config(config)
        self.yancc_wrapper = yancc_wrapper(self.Density, 1e19, 1e3)
        self.dSigmaFn_dVars = jax.jacfwd(self.yancc_wrapper.flux, argnums=0)

    def LowerBoundary(self, index, t):
        return 0.0

    def UpperBoundary(self, index, t):
        return self.params.EdgeTemperature

    #@partial(jax.jit, static_argnums=(0,1))
    def SigmaFn( self, index, state, x, t ):
        f = self.yancc_wrapper.flux(state, x)
        print(f)
        return -f[index]

    @partial(jax.jit, static_argnums=(0,1))
    def Sources( self, index, state, x, t ):
        return self.source(index, state, x, t, self.params)

    
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
    def sigma( self, index, state, x, t, params: NamedTuple ):
        pass

    def source( self, index, state, x, t, params: NamedTuple ):
        return params.SourceHeight * jnp.exp(-(x - params.SourceCenter)**2 / (2 * params.SourceWidth**2))

    #@partial(jax.jit, static_argnums=(0,1))
    def dSigmaFn_dq( self, index, state, x, t):
        return self.dSigmaFn_dVars(state,x)["Derivative"]
    
    #@partial(jax.jit, static_argnums=(0,1))
    def dSigmaFn_du( self, index, state, x, t):
        return self.dSigmaFn_dVars(state,x)["Variable"]
    
    def dSigma_dPhi( self, index, state, x, t):
        return self.dSigmaFn_dVars(state,x)["Aux"]
    
    @partial(jax.jit, static_argnums=(0,1))
    def dSources_du( self, index, state, x, t ):
        return jax.grad(self.Sources, argnums=1)(index, state, x, t)["Variable"]

    @partial(jax.jit, static_argnums=(0,1))
    def dSources_dq( self, index, state, x, t ):
        return jax.grad(self.Sources, argnums=1)(index, state, x, t)["Derivative"]

    @partial(jax.jit, static_argnums=(0,1))
    def dSources_dsigma( self, index, state, x, t ):
        return jax.grad(self.Sources, argnums=1)(index, state, x, t)["Flux"]
    
    @partial(jax.jit, static_argnums=(0,1))
    def dSources_dPhi( self, index, state, x, t ):
        return jax.grad(self.Sources, argnums=1)(index, state, x, t)["Aux"]
    
    @partial(jax.jit, static_argnums=(0,1))
    def InitialValue( self, index, x ):
        return self.params.EdgeTemperature * self.Density(x)
    
    @partial(jax.jit, static_argnums=(0,1))
    def InitialDerivative( self, index, x ):
        return jax.grad(self.InitialValue, argnums=1)(index, x)
    
    def InitialAuxValue(self, index, x):
        return 0.0

    def Density(self, x):
        return (self.params.n0 - self.params.EdgeDensity) * (1 - x) + self.params.EdgeDensity
    
    """
    Create the adjoint problem associated with this transport system
    
    Returns
    -------
    JAXAdjointProblem
        The adjoint problem object
    """
    def createAdjointProblem(self):
        pass

def registerTransportSystems():
    MaNTA.registerPhysicsCase("StellaratorTransport", StellaratorTransport)
