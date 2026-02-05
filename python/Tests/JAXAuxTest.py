from typing import NamedTuple
import sys
sys.path.insert(0, '../')  # To find MaNTA module

import MaNTA
from JAXTransportSystem import JAXTransportSystem
from JAXAdjointProblem import JAXAdjointProblem
import jax.numpy as jnp

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
    MaNTA.registerPhysicsCase("JAXAuxTest", JAXAuxTest)
