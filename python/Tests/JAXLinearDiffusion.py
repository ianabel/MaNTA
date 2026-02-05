from typing import NamedTuple
import sys
sys.path.insert(0, '../')  # To find MaNTA module

import MaNTA
from JAXTransportSystem import JAXTransportSystem
from JAXAdjointProblem import JAXAdjointProblem
import jax.numpy as jnp

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
    
    def createAdjointProblem(self):
        adjointProblem = JAXAdjointProblem(self, self.g)
        return adjointProblem

def registerTransportSystems():
    MaNTA.registerPhysicsCase("JAXLinearDiffusion", JAXLinearDiffusion)
