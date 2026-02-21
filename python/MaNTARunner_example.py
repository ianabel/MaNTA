from typing import NamedTuple

import MaNTA
from JAXTransportSystem import JAXTransportSystem
from JAXAdjointProblem import JAXAdjointProblem
import jax.numpy as jnp

class LinearDiffusionParams(NamedTuple):
    Centre: float
    InitialWidth: float
    InitialHeight: float
    kappa: float
    
config = {
    "OutputFilename": "out",
    "Polynomial_degree": 5,
    "Grid_size": 5,
    "Lower_boundary": -1.0,
    "Upper_boundary":  1.0,
    "Relative_tolerance" : 0.01,
    "tFinal": 1.0,
    "delta_t": 0.5,
    "solveAdjoint": True, 
}
    
class JAXLinearDiffusion(JAXTransportSystem):
    def __init__(self):
        super().__init__()

        self.nVars = 1

        self.isUpperDirichlet  = True
        self.isLowerDirichlet  = True

        self.params = LinearDiffusionParams(0.0, 0.1, 1.0, 2.0)

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

def runMaNTA():
    transportSystem = JAXLinearDiffusion()
    runner = MaNTA.Runner(transportSystem)
    runner.configure(config)
    runner.run()

runMaNTA()
