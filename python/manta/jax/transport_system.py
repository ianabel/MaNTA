import jax
import numpy as np
import manta as MaNTA
from typing import NamedTuple
from abc import abstractmethod

from .adjoint_problem import JAXAdjointProblem
from .state import MaNTA_Decorator, ShiftedState_Decorator

"""
JAX-based transport system base class that overloads MaNTA TransportSystem.
Enables automatic differentiation of sigma and source terms using JAX.
"""

# Base class for JAX-based transport systems
class JAXTransportSystem(MaNTA.TransportSystem):
    # The spec comes from the concrete case and is forwarded: this base has no
    # way to know how many variables its subclass has.
    def __init__(self, spec):
        MaNTA.TransportSystem.__init__(self, spec)
        self.dSigmadvar = jax.jit(jax.grad(self.sigma, argnums=1))
        self.dSourcedvar = jax.jit(jax.grad(self.source, argnums=1))

        self.dAuxdvars = jax.jit(jax.grad(self.aux, argnums = 1))

        self.dInitialValue = jax.jit(jax.grad(self.InitialValue, argnums=1))

    @abstractmethod
    def LowerBoundary(self, index, t):
        raise NotImplementedError("LowerBoundary function not implemented")

    @abstractmethod
    def UpperBoundary(self, index, t):
        raise NotImplementedError("UpperBoundary function not implemented")

    @MaNTA_Decorator
    def SigmaFn( self, index, state, x, t ):
        return self.sigma(index, state, x, t, self.params)

    @MaNTA_Decorator
    def Sources( self, index, state, x, t ):
        return self.source(index, state, x, t, self.params)

    
    """
    Sigma and source, and auxiliary functions to be overloaded in derived classes

    Parameters
    ----------
    index : int
        Variable index
    state : dict
        Dictionary containing "Variable", "Derivative", "Flux", "Aux", and "Scalars" arrays
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
    def sigma( self, index, state, x, t, params ):
        raise NotImplementedError("sigma function not implemented")

    @abstractmethod
    def source( self, index, state, x, t, params ):
        raise NotImplementedError("source function not implemented")

    @abstractmethod
    def aux( self, index, state, x, t, params ):
        pass

    @MaNTA_Decorator
    def dSigmaFn_dq( self, index, state, x, t):

        return self.dSigmadvar(index,state,x,t, self.params).Derivative
    
    @MaNTA_Decorator
    def dSigmaFn_du( self, index, state, x, t):
        return self.dSigmadvar(index,state,x,t, self.params).Variable

    @MaNTA_Decorator
    def dSigma_dPhi( self, index, state, x, t):
        return self.dSigmadvar(index,state,x,t, self.params).Aux
    
    @MaNTA_Decorator
    def dSources_du( self, index, state, x, t ):
        return self.dSourcedvar(index,state,x,t, self.params).Variable

    @MaNTA_Decorator
    def dSources_dq( self, index, state, x, t ):
        return self.dSourcedvar(index,state,x,t, self.params).Derivative

    @MaNTA_Decorator
    def dSources_dsigma( self, index, state, x, t ):
        return self.dSourcedvar(index,state,x,t, self.params).Flux
    
    @MaNTA_Decorator
    def dSources_dPhi( self, index, state, x, t ):
        return self.dSourcedvar(index,state,x,t, self.params).Aux
    
    @MaNTA_Decorator
    def AuxG( self, index, state, x, t):
        return self.aux(index, state, x, t, self.params)
    
    """
    Compute the derivatives of the auxiliary constraint G_index.

    The one hook here that fills a buffer rather than returning a value: `out`
    is a non-owning window onto the solver's State, and it arrives zeroed, so
    only the nonzero entries need writing. That extra argument sits *ahead of*
    the state, which is why this takes ShiftedState_Decorator -- MaNTA_Decorator
    would convert `out` and hand the state to jnp.array().

    Parameters
    ----------
    index : int
        Auxiliary variable index
    out : manta.State
        Written in place; out.u[j], out.q[j] and out.phi[j] receive dG/du_j,
        dG/dq_j and dG/dphi_j
    state : State
        The solution at this point
    x : float
        Spatial location
    t : float
        Time
    """
    @ShiftedState_Decorator
    def AuxGPrime( self, index, out, state, x, t):
        d = self.dAuxdvars(index, state, x, t, self.params)
        for j, v in enumerate(np.asarray(d.Variable)):
            out.u[j] = float(v)
        for j, v in enumerate(np.asarray(d.Derivative)):
            out.q[j] = float(v)
        if d.Aux is not None:
            for j, v in enumerate(np.asarray(d.Aux)):
                out.phi[j] = float(v)
    
    @abstractmethod
    def InitialValue( self, index, x ):
        raise NotImplementedError("InitialValue function not implemented")

    def InitialDerivative( self, index, x ):
        return self.dInitialValue(index,x)
    
    @abstractmethod
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
