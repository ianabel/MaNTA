import jax
import jax.numpy as jnp
import MaNTA

class JAXAdjointProblem(MaNTA.AdjointProblem):
    def __init__(self, transport_system: MaNTA.TransportSystem, g):
        MaNTA.AdjointProblem.__init__(self)
        self.params = transport_system.params
        self.g = g
        self.dg_dvar = jax.jit(jax.grad(self.g, argnums=0))
        self.dg_dp = jax.jit(jax.grad(self.g, argnums=2))
        self.np = len(transport_system.params)
        self.np_boundary = 0
        self.dsigma_dp = jax.jit(jax.grad(transport_system.sigma, argnums=4))
        self.dsource_dp = jax.jit(jax.grad(transport_system.source, argnums=4))
        self.daux_dp = jax.jit(jax.grad(transport_system.aux, argnums=4))

        self.UpperBoundarySensitivities = {}
        self.LowerBoundarySensitivities = {}

    def gFn(self, i, state, x):
        return self.g(state, x, self.params)
    
    def dgFndp(self, i, state, x):
        if ( i < self.np-1 ):
            return self.dg_dp(state, x, self.params)[i]
        else:
            return 0.0

    def dgFn_du(self, i, state, x):
        return self.dg_dvar(state, x, self.params)["Variable"]
    
    def dgFn_dq(self, i, state, x):
        return self.dg_dvar(state, x, self.params)["Derivative"]

    def dgFn_dsigma(self, i, state, x):
        return self.dg_dvar(state, x, self.params)["Flux"]
    
    def dgFn_dphi(self, i, state, x):
        return self.dg_dvar(state, x, self.params)["Aux"]

    def dSigmaFn_dp(self, index, pIndex, state, x):
        return self.dsigma_dp(index, state, x, 0.0, self.params)[pIndex]
    
    def dSources_dp(self, index, pIndex, state, x):
        return self.dsource_dp(index, state, x, 0.0, self.params)[pIndex]
    
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
        
    def addUpperBoundarySensitivity(self, i):
        self.UpperBoundarySensitivities[(i,self.np)] = True
        self.np += 1
        self.np_boundary += 1

    def addLowerBoundarySensitivity(self, i):
        self.LowerBoundarySensitivities[(i,self.np)] = True
        self.np += 1
        self.np_boundary += 1
    
   