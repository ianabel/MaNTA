import MaNTA
import jax.numpy as jnp
import jax

from functools import partial

from yancc_wrapper import yancc_wrapper 

from typing import NamedTuple

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

# Magic tuple to make vmap work
vmap_axes = ({"Variable": 1, "Derivative": 1, "Flux": 1, "Aux": 1, "Scalars": None}, 0)
"""
class StellaratorTransport

Computes sources and neoclassical fluxes (returned from yancc) as required by MaNTA
"""
class StellaratorTransport(MaNTA.TransportSystem): 
    def __init__(self, config , grid: MaNTA.Grid = None):
        MaNTA.TransportSystem.__init__(self)
        self.nVars = 1

        ### Remember to set boundary conditions ####
        self.isUpperDirichlet  = True
        self.isLowerDirichlet  = False

        self.params = StellaratorParams.from_config(config)
        self.yancc_wrapper = yancc_wrapper(self.Density, 1e20, 1e3)
        self.dSigmaFn_dVars = jax.grad(self.SigmaFn, argnums=1)

    def LowerBoundary(self, index, t):
        return 0.0

    def UpperBoundary(self, index, t):
        return 1.5 * self.params.EdgeTemperature * self.Density(0.9)

    def SigmaFn( self, index, state, x, t ):
        return self.sigma(index, state, x, t, self.params)

    def Sources(self, index, state, x, t):
        return self.source(index, state, x, t, self.params)
    
    def SigmaFn_v( self, index, states, positions, t):
        x = jnp.array(positions)
        return jax.vmap(lambda s, p : self.sigma(index, s, p, t, self.params), in_axes=(vmap_axes))(states, x)

    def Sources_v( self, index, states, positions, t ):
        x = jnp.array(positions)
        return jax.vmap(lambda s, p : self.source(index, s, p, t, self.params), in_axes=(vmap_axes))(states, x)
    
    def dSigma(self, index, states, positions, t):
        x = jnp.array(positions)
        out =  jax.vmap(lambda s, p: jax.grad(self.sigma, argnums=1)(index, s, p, t, self.params), in_axes=(vmap_axes))(states, x)
        out["Scalars"] = []
        return out
    
    def dSources(self, index, states, positions, t):
        x = jnp.array(positions)
        out =  jax.vmap(lambda s, p: jax.grad(self.source, argnums=1)(index, s, p, t, self.params), in_axes=(vmap_axes))(states, x)
        out["Scalars"] = []
        return out

    
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
        return -self.yancc_wrapper.flux(state, x) 

    def source( self, index, state, x, t, params: NamedTuple ):
        return params.SourceHeight * jnp.exp(-(x - params.SourceCenter)**2 / (2 * params.SourceWidth**2))

    #@partial(jax.jit, static_argnums=(0,1))
    def dSigmaFn_dq( self, index, state, x, t):
        return self.dSigmaFn_dVars(index,state,x,t)["Derivative"]
    
    #@partial(jax.jit, static_argnums=(0,1))
    def dSigmaFn_du( self, index, state, x, t):
        return self.dSigmaFn_dVars(index,state,x,t)["Variable"]
    
    def dSigma_dPhi( self, index, state, x, t):
        return self.dSigmaFn_dVars(index,state,x,t)["Aux"]
    
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
        return 1.5 * self.params.EdgeTemperature * self.Density(x)
    
    @partial(jax.jit, static_argnums=(0,1))
    def InitialDerivative( self, index, x ):
        return jax.grad(self.InitialValue, argnums=1)(index, x)
    
    def InitialAuxValue(self, index, x):
        return 0.0

    def Density(self, x):
        return (self.params.n0 - self.params.EdgeDensity) * (1 - x*x) + self.params.EdgeDensity
    
    """
    Create the adjoint problem associated with this transport system
    
    Returns
    -------
    JAXAdjointProblem
        The adjoint problem object
    """
    def createAdjointProblem(self):
        pass
