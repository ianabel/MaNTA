import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import jax
import jax.numpy as jnp
import MaNTA
from JAXAdjointProblem import JAXAdjointProblem
from typing import NamedTuple, Any

"""
JAX-based transport system base class that overloads MaNTA TransportSystem.
Enables automatic differentiation of sigma and source terms using JAX.
"""

# Base class for JAX-based transport systems
class JAXTransportSystem(MaNTA.TransportSystem):
    def __init__(self):
        MaNTA.TransportSystem.__init__(self)
        self.nAux = 0
        self.dSigmadvar = jax.jit(jax.grad(self.SigmaFn, argnums=1))
        self.dSourcedvar = jax.jit(jax.grad(self.Sources, argnums=1))

        self.dAuxdvars = jax.jit(jax.grad(self.AuxG, argnums = 1))

        self.dInitialValue = jax.jit(jax.grad(self.InitialValue, argnums=1))

    def LowerBoundary(self, index, t):
        pass

    def UpperBoundary(self, index, t):
        pass


    def SigmaFn( self, index, state, x, t ):
        return self.sigma(index, state, x, t, self.params)

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
        pass

    def aux( self, index, state, x, t, params: NamedTuple):
        pass

    def dSigmaFn_dq( self, index, state, x, t):
        return self.dSigmadvar(index,state,x,t)["Derivative"]
    
    def dSigmaFn_du( self, index, state, x, t):
        return self.dSigmadvar(index,state,x,t)["Variable"]
    
    def dSigma_dPhi( self, index, state, x, t):
        return self.dSigmadvar(index,state,x,t)["Aux"]
        
    def dSources_du( self, index, state, x, t ):
        return self.dSourcedvar(index,state,x,t)["Variable"]

    def dSources_dq( self, index, state, x, t ):
        return self.dSourcedvar(index,state,x,t)["Derivative"]

    def dSources_dsigma( self, index, state, x, t ):
        return self.dSourcedvar(index,state,x,t)["Flux"]
    
    def dSources_dPhi( self, index, state, x, t ):
        return self.dSourcedvar(index,state,x,t)["Aux"]
    
    def AuxG( self, index, state, x, t):
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
    def AuxGPrime( self, index, state, x , t):
        return self.dAuxdvars(index, state, x, t)
      
    def InitialValue( self, index, x ):
        pass

    def InitialDerivative( self, index, x ):
        return self.dInitialValue(index,x)
    
    def InitialAuxValue(self, index, x):
        pass
    
    """
    Create the adjoint problem associated with this transport system
    
    Returns
    -------
    JAXAdjointProblem
        The adjoint problem object
    """
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
    def make(cls, config: MaNTA.TomlValue) -> 'NonlinearDiffusionParams':
        return cls(
             SourceCentre = config["SourceCentre"],
             D = config["D"],
             T_s = 50.0,
             a = config["a"],
             SourceWidth = 0.02
        )

class JAXNonlinearDiffusion(JAXTransportSystem):
    def __init__(self, config: MaNTA.TomlValue, grid: MaNTA.Grid):
        super().__init__()
        self.nVars = 1
        self.isUpperDirichlet  = True
        self.isLowerDirichlet  = False

        # This object will be passed to sigma and source functions
        self.params = NonlinearDiffusionParams.make(config)

    def g(self, state, x, params: NonlinearDiffusionParams):
        u = state["Variable"][0]
        return 0.5 * u * u
    
    def sigma( self, index, state, x, t, params: NonlinearDiffusionParams ):
        
        u = state["Variable"][0]
        q = state["Derivative"][0]
        return params.D*(u ** params.a) * q

    def source( self, index, state, x, t, params: NonlinearDiffusionParams ):
        y = x - params.SourceCentre
        return params.T_s*jnp.exp(-y*y/params.SourceWidth)


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
    
class JAXAuxTest(JAXTransportSystem):
    def __init__(self, config: MaNTA.TomlValue, grid: MaNTA.Grid):
        super().__init__()
        self.nVars = 1
        self.nAux = 1
        self.isUpperDirichlet  = True
        self.isLowerDirichlet  = False

        # This object will be passed to sigma and source functions
        self.params = NonlinearDiffusionParams.make(config)

    def g(self, state, x, params: NonlinearDiffusionParams):
        u = state["Variable"][0]
        return 0.5 * u * u
    
    def sigma( self, index, state, x, t, params: NonlinearDiffusionParams ):
        
        u = state["Variable"][0]
        q = state["Derivative"][0]
        return params.D*(u ** params.a) * q
    
    def aux( self, index ,state, x, t, params):
        a = state["Aux"][0]
        u = state["Variable"][0]
        return a - params.D*u*u

    def source( self, index, state, x, t, params: NonlinearDiffusionParams ):
        y = x - params.SourceCentre
        u = state["Variable"][0]
        a = state["Aux"][0]
        return params.T_s*jnp.exp(-y*y/params.SourceWidth) + a - params.D*u*u

    def LowerBoundary(self, index, t):
        return 0.0

    def UpperBoundary(self, index, t):
        return 0.3
    
    def InitialValue(self, index, x):
        return 0.3
    
    def InitialAuxValue(self, index, x):
        u0 = self.InitialValue(index, x)
        return self.params.D*u0*u0
    
    def createAdjointProblem(self):
        adjointProblem = JAXAdjointProblem(self, self.g)
        adjointProblem.addUpperBoundarySensitivity(0)
        return adjointProblem

def registerTransportSystems():

    MaNTA.registerPhysicsCase("JAXNonlinearDiffusion", JAXNonlinearDiffusion)
    MaNTA.registerPhysicsCase("JAXAuxTest", JAXAuxTest)

