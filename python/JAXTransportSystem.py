import jax
import jax.numpy as jnp
import MaNTA
from JAXAdjointProblem import JAXAdjointProblem
from typing import NamedTuple, Any
# Base class for JAX-based transport systems

# def flatten_state(state: MaNTA.State):
#     children = (state["Variable"], state["Derivative"], state["Flux"], state["Aux"], state["Scalars"])
#     aux_data= tuple()
#     return (children, aux_data)

# def unflatten_state(aux_data: Any, children: Any) -> MaNTA.State:
#     return MaNTA.State({
#         "Variable": children[0],
#         "Derivative": children[1],
#         "Flux": children[2],
#         "Aux": children[3],
#         "Scalars": children[4],
#     })
# @jax.tree_util.register_pytree_node_class(MaNTA.State, flatten_state, unflatten_state)


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

    # User-defined functions to be overridden in derived classes
    def sigma( self, index, state, x, t, params: NamedTuple ):
        pass
    def source( self, index, state, x, t, params: NamedTuple ):
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

def registerTransportSystems():

    MaNTA.registerPhysicsCase("JAXNonlinearDiffusion", JAXNonlinearDiffusion)

