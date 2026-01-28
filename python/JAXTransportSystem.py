import jax
import jax.numpy as jnp
import MaNTA
from JAXAdjointProblem import JAXAdjointProblem
from typing import NamedTuple, Any
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
        return self.sigma(index, state, x, t, self.params)

    def Sources( self, index, state, x, t ):
        return self.source(index, state, x, t, self.params)

    def sigma( self, index, state, x, t, params: NamedTuple):
        pass

    def source( self, index, state, x, t, params: NamedTuple):
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


# Need PyTree structure for class paramters to be able to compute adjoints
class LinearDiffusionParams(NamedTuple):
    Centre: float
    InitialWidth: float
    InitialHeight: float
    kappa: float

    @classmethod
    def make(cls, config: MaNTA.TomlValue) -> 'LinearDiffusionParams':
        InitialHeight = 1.0
        InitialWidth = 0.1
        return cls(
            Centre = config["Centre"],
            InitialWidth = InitialWidth,
            InitialHeight = InitialHeight,
            kappa = config["kappa"],
        )

class JAXLinearDiffusion(JAXTransportSystem):
    def __init__(self, config: MaNTA.TomlValue, grid: MaNTA.Grid):
        super().__init__()

        self.nVars = 1

        self.isUpperDirichlet  = True
        self.isLowerDirichlet  = True

        self.params = LinearDiffusionParams.make(config)

    def g(self, state, x, params):
        u = state["Variable"][0]
        return 0.5 * u * u

    def sigma( self, index, state, x, t, params ):
        tprime = state["Derivative"]
        return params.kappa * tprime[index]

    def source( self, index, state, x, t, params ):
        return 10.0
    
    def LowerBoundary(self, index, t):
        return 0.0

    def UpperBoundary(self, index, t):
        return 0.0
    
    def InitialValue( self, index, x ):
        alpha = 1 / 0.02
        y = (x - self.params.Centre)
        return self.params.InitialHeight * jnp.exp(-alpha * y * y)

def show_example(structured):
    flat, tree = structured.tree_flatten()
    unflattened = JAXLinearDiffusion.tree_unflatten(tree, flat)
    print(f"{structured=}\n  {flat=}\n  {tree=}\n  {unflattened=}")

class NonlinearDiffusionParams(NamedTuple):
    SourceCentre: float
    D: float
    T_s: float
    a: float
    SourceWidth: float
   

    @classmethod
    def make(cls, config: MaNTA.TomlValue) -> 'NonlinearDiffusionParams':
        
        SourceCentre = config["SourceCentre"]
        D = config["D"]
        T_s = 50.0
        a = config["a"]
        SourceWidth = 0.02
 
        return cls(
            T_s = T_s,
            a = a,
            SourceWidth = SourceWidth,
            SourceCentre = SourceCentre,
            D = D,
        )

class JAXNonlinearDiffusion(JAXTransportSystem):
    def __init__(self,config,grid):
        super().__init__()
        self.nVars = 1
        self.isUpperDirichlet  = True
        self.isLowerDirichlet  = False

        self.params = NonlinearDiffusionParams.make(config)

    def g(self, state, x, params):
        u = state["Variable"][0]
        return 0.5 * u * u

    def sigma( self, index, state, x, t, params ):
        
        u = state["Variable"][0]
        q = state["Derivative"][0]
        return params.D*(u ** params.a) * q

    def source( self, index, state, x, t, params ):
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


def registerTransportSystems():
    MaNTA.registerPhysicsCase("JAXLinearDiffusion", JAXLinearDiffusion)
    MaNTA.registerPhysicsCase("JAXNonlinearDiffusion", JAXNonlinearDiffusion)

