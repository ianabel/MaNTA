import jax
import jax.numpy as jnp
import MaNTA

# Base class for JAX-based transport systems
class JAXTransportSystem(MaNTA.TransportSystem):
    def __init__(self):
        MaNTA.TransportSystem.__init__(self)
        self.dSigmadvar = jax.jit(jax.grad(self.SigmaFn, argnums=1))
        self.dSourcedvar = jax.jit(jax.grad(self.Sources, argnums=1))
        self.dInitialValue = jax.jit(jax.grad(self.InitialValue, argnums=1))

    def LowerBoundary(self, index, t):
        pass

    def UpperBoundary(self, index, t):
        pass

    def SigmaFn( self, index, state, x, t ):
        pass

    def Sources( self, index, state, x, t ):
        pass

    def dSigmaFn_dq( self, index, state, x, t):
        return self.dSigmadvar(index,state,x,t)["Derivative"]
    
    def dSigmaFn_du( self, index, state, x, t):
        return self.dSigmadvar(index,state,x,t)["Variable"]
        
    def dSources_du( self, index, state, x, t ):
        return self.dSourcedvar(index,state,x,t)["Variable"]

    def dSources_dq( self, index, state, x, t ):
        return self.dSourcedvar(index,state,x,t)["Derivative"]

    def dSources_dsigma( self, index, state, x, t ):
        return self.dSourcedvar(index,state,x,t)["Flux"]
    
    def InitialValue( self, index, x ):
        pass

    def InitialDerivative( self, index, x ):
        return self.dInitialValue(index,x)
    
class JAXLinearDiffusion(JAXTransportSystem):
    def __init__(self, config, grid):
        super().__init__()

        self.nVars = 1

        self.isUpperDirichlet  = True
        self.isLowerDirichlet  = True

        self.Centre = config["Centre"]
        self.InitialWidth = 0.1
        self.InitialHeight = 1.0
        self.kappa = config["kappa"]
        self.t0 = self.InitialWidth * self.InitialWidth / ( 4.0 * self.kappa )
    
    def SigmaFn( self, index, state, x, t ):
        tprime = state["Derivative"]
        return self.kappa * tprime[index]

    def Sources( self, index, state, x, t ):
        return 0.0
    
    def LowerBoundary(self, index, t):
        return 0.0

    def UpperBoundary(self, index, t):
        return 0.0
    
    def InitialValue( self, index, x ):
        alpha = 1 / self.InitialWidth
        y = (x - self.Centre)
        return self.InitialHeight * jnp.exp(-alpha * y * y)


class NonlinearDiffusion(JAXTransportSystem):
    def __init__(self,config,grid):
        super().__init__()
        self.nVars = 1
        self.isUpperDirichlet  = True
        self.isLowerDirichlet  = True

        self.Centre = config["Centre"]
        self.InitialWidth = 0.1
        self.InitialHeight = 1.0
        self.n = 2
        self.t0 = 1.1

    def SigmaFn( self, index, state, x, t ):

        u = state["Variable"][index]
        q = state["Derivative"][index]

        NonlinearKappa = self.n/2.0*u**self.n*(1-u**self.n)/(self.n+1.0)
        return NonlinearKappa * q

    def Sources( self, index, state, x, t ):
        return 0.0
    
    def LowerBoundary(self, index, t):
        return 0.0

    def UpperBoundary(self, index, t):
        return self.ExactSolution( 1, self.t0+t)
    
    def InitialValue(self, index, x):
        return self.ExactSolution(x,self.t0)
    
    def ExactSolution( self, x, t ):
        eta = x/t
  
        if (eta >= 1.0):
            return 0.0
        return jnp.pow(1.0-eta,1.0/self.n)


def registerTransportSystems():
    #MaNTA.testFunction(PythonLinearDiffusion)
    MaNTA.registerPhysicsCase("JAXLinearDiffusion", JAXLinearDiffusion)
    MaNTA.registerPhysicsCase("JAXNonlinearDiffusion", NonlinearDiffusion)

