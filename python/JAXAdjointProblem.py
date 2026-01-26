import jax
import jax.numpy as jnp
import MaNTA

class JAXAdjointProblem(MaNTA.AdjointProblem):
    def __init__(self, transport_system, g):
        MaNTA.AdjointProblem.__init__(self)
        self.transport_system = transport_system
        self.params = self.transport_system.params
        self.g = g
        self.dg_dvar = jax.jit(jax.grad(self.g, argnums=0))
        self.dg_dp = jax.jit(jax.grad(self.g, argnums=2))
        self.np = len(self.transport_system.params)+1
        self.np_boundary = 1
        print("Number of parameters in adjoint problem: ", self.np)
        self.dsigma_dp = jax.jit(jax.grad(self.transport_system.sigma, argnums=4))
        self.dsource_dp = jax.jit(jax.grad(self.transport_system.source, argnums=4))

    def gFn(self, i, state, x):
        return self.g(state, x, self.params)
    
    def dgFndp(self, i, state, x):
        if ( i < self.np -2 ):
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
    
    def computeUpperBoundarySensitivity(self, i, pIndex):
        if (pIndex >= self.np-1):
            return True
        else:
            return False
    def computeLowerBoundarySensitivity(self, i, pIndex):
        return False
    
    def getName(self, pIndex):
        if pIndex < self.np -1:
            return list(self.params._fields)[pIndex]
        else:
            return "BoundaryCondition"
    
   