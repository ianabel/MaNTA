import jax
import os
import MaNTA

import jax.numpy as jnp
import equinox as eqx

from typing import Optional

from jaxtyping import ArrayLike

from jax.flatten_util import ravel_pytree

import abc

state_vmap_axes = {"Variable": 0, "Derivative": 0, "Flux": 0, "Aux": 0, "Scalars": None}

# class TransportMeta(type(MaNTA.TransportSystem), type(eqx.Module)): pass

class TransportModule(MaNTA.TransportSystem):
    params: eqx.Module
    nVars: int
    nAux: int
    isLowerDirichlet: bool
    isUpperDirichlet: bool

    @classmethod
    def from_other( cls, params, other ):
        return cls( params=params, nVars=other.nVars, nAux=other.nAux, isLowerDirichlet=other.isLowerDirichlet, isUpperDirichlet=other.isUpperDirichlet )
    
    @classmethod
    def from_config( cls, params, config):
        print(config["nVars"])
        print(f"Called from: {cls.__name__}")
        return cls( params=params,nVars=config["nVars"], nAux=config["nAux"], isLowerDirichlet=config["isLowerDirichlet"], isUpperDirichlet=config["isUpperDirichlet"] ) 

    @eqx.filter_jit
    def SigmaFn_v( self, index, states, positions, t):
        x = jnp.array(positions)
        return jax.vmap(lambda s, p : self.sigma(index, s, p, t, self.params), in_axes=(state_vmap_axes, 0))(states, x)

    @eqx.filter_jit
    def Sources_v( self, index, states, positions, t ):
        x = jnp.array(positions)
        return jax.vmap(lambda s, p : self.source(index, s, p, t, self.params), in_axes=(state_vmap_axes, 0))(states, x)
    
    @eqx.filter_jit
    def dSigma(self, index, states, positions, t):
        x = jnp.array(positions)
        out =  jax.vmap(lambda s, p: jax.grad(self.sigma, argnums=1)(index, s, p, t, self.params), in_axes=(state_vmap_axes, 0))(states, x)
        out["Scalars"] = []
        return out
    
    @eqx.filter_jit
    def dSources(self, index, states, positions, t):
        x = jnp.array(positions)
        out =  jax.vmap(lambda s, p: jax.grad(self.source, argnums=1)(index, s, p, t, self.params), in_axes=(state_vmap_axes, 0))(states, x)
        out["Scalars"] = []
        return out
    
    @abc.abstractmethod
    def sigma( self, index, state, x, t, params ):
        raise NotImplementedError
    
    @abc.abstractmethod
    def source( self, index, state, x, t, params ):
        raise NotImplementedError

    @abc.abstractmethod
    def LowerBoundary(self, index, t):
        raise NotImplementedError

    @abc.abstractmethod
    def UpperBoundary(self, index, t):
        raise NotImplementedError

    @abc.abstractmethod
    def InitialValue( self, index, x ):
        raise NotImplementedError
    
    @eqx.filter_jit
    def InitialDerivative( self, index, x ):
        return jax.grad(self.InitialValue, argnums=1)(index, x)
 

class AdjointModule(MaNTA.AdjointProblem):
    params: eqx.Module
    np: int 
    ng: int 
    np_boundary: int 
    spatialParameters: bool 
    g: callable 
    sigma: callable 
    source: callable 
    UpperBoundarySensitivities: dict 
    LowerBoundarySensitvities: dict 

    def __init__(
            self, 
            nVars, 
            np,
            g,
            sigma, 
            source, 
            params,
            spatialParameters: Optional[bool] = False, 
            computeBoundarySensitivies: Optional[bool] = False,
            np_boundary: Optional[int] = 0):
        MaNTA.AdjointProblem.__init__(self)
        self.g = g
        self.np = np
        self.ng = len(g)
        self.sigma =sigma
        self.source = source
        self.params = params
        self.spatialParameters = spatialParameters
        self.np_boundary = np_boundary
        self.UpperBoundarySensitivities = {}
        self.LowerBoundarySensitvities = {}
        if (computeBoundarySensitivies):
            for i in range(0, nVars):
                self.UpperBoundarySensitivities[(i,self.np)] = True
                self.np += 1
                self.np_boundary += 1


                self.LowerBoundarySensitivities[(i,self.np)] = True
                self.np += 1
                self.np_boundary += 1

    @classmethod 
    def from_transport_system(cls, transport_system: TransportModule, spatialParameters = False):
        return cls(
            nVars = transport_system.nVars,
            np = len(transport_system.params),
            g = [transport_system.g],
            sigma = transport_system.sigma,
            source = transport_system.source,
            params = transport_system.params, 
            spatialParameters = spatialParameters)

    def setParams(self, params):
        self.params = params

    def gFn(self, i, states, positions):
        x = jnp.array(positions)
        out = jax.vmap(self.g[i], in_axes=(state_vmap_axes, 0, None))(states, x, self.params)
       
        return out

    def dgFndp(self, gIndex, states, positions):
        x = jnp.array(positions)
        dgdp = jax.vmap(jax.grad(self.g[gIndex], argnums=2), in_axes=(state_vmap_axes, 0, None))(states, x, self.params)
        g, _ = ravel_pytree(dgdp)
        g = jnp.reshape(g, (self.np - self.np_boundary, len(positions)))

        out = jnp.pad(g, pad_width=(0, self.np_boundary), mode='constant', constant_values=0)
    
        return out

    def dg(self, i, states, positions):
        x = jnp.array(positions)

        out = jax.vmap(jax.grad(self.g[i], argnums=0), in_axes=(state_vmap_axes, 0, None))(states, x, self.params)  
        out["Scalars"] = []
        return out

    def dSigma(self, i, states, positions):
        x = jnp.array(positions)
        out = jax.vmap(jax.grad(self.sigma, argnums=4), in_axes=(None, state_vmap_axes, 0, None, None))(i, states, x, 0.0, self.params)  
        return out
    
    
    def dSources(self, i, states, positions):
        x = jnp.array(positions)
        out = jax.vmap(jax.grad(self.source, argnums=4), in_axes=(None, state_vmap_axes, 0, None, None))(i, states, x, 0.0, self.params)  
        return out

    def dgFn_dphi(self, i, state, x):
        return jax.grad(self.g, argnums=0)(state, x, self.params)["Aux"]
   
    def dAux_dp(self, index, pIndex, state, x):
        return self.daux_dp(index, state, x, 0.0, self.params )[pIndex]
    
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
            return "BoundaryCondition"+str(pIndex)
    
        
    
   
