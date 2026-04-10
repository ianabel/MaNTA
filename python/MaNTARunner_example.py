from typing import NamedTuple

from functools import partial

import MaNTA
from VectorizedTransportSystem import VectorizedTransportSystem
from JAXAdjointProblem import JAXAdjointProblem
import jax.numpy as jnp
import jax

class NonlinearDiffusionParams(NamedTuple):
    D: float
    T_s: float
    a: float    
    SourceWidth: float
    SourceCentre: float
   
    @classmethod
    def make(cls, config) -> 'NonlinearDiffusionParams':
        return cls(
             SourceCentre = config["SourceCentre"],
             D = config["D"],
             T_s = 50.0,
             a = config["a"],
             SourceWidth = 0.02
        )

class JAXNonlinearDiffusion(VectorizedTransportSystem):
    def __init__(self, config):
        super().__init__()
        self.nVars = 1
        self.isUpperDirichlet  = True
        self.isLowerDirichlet  = False
        

        solver_config = {
            "OutputFilename": "out",
            "Polynomial_degree": 4,
            "Grid_size": 20,
            "tau": 1.0, 
            "Lower_boundary": 0.0,
            "Upper_boundary": 1.0,
            "Relative_tolerance": 0.01,
            "delta_t": 1.0,
            "restart": False,
            "solveAdjoint": True, 
        }
        print(config)
        self.params = NonlinearDiffusionParams.make(config)
        self.points = MaNTA.getNodes(solver_config["Lower_boundary"], solver_config["Upper_boundary"], solver_config["Grid_size"], solver_config["Polynomial_degree"])

        self.runner = MaNTA.Runner(self)

        self.runner.configure(solver_config)

        self.adjointProblem = JAXAdjointProblem(self, self.g)
        self.runner.setAdjointProblem(self.adjointProblem)
        # This object will be passed to sigma and source functions
    
    def run(self, tFinal = None, kappa = None):
        if (kappa is not None):
            self.params["D"] = kappa
        
        if (tFinal is not None):
            sFinal = self.runner.run(tFinal)
        else: 
            sFinal = self.runner.run_ss()

        return sFinal

    def runAdjointSolve(self, kappa = None):
        if (kappa is not None):
            self.params.D = kappa
        G, G_p = self.runner.runAdjointSolve()
        return G, G_p

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
        pass

def runMaNTA():
    config = {
        "SourceCentre" : 0.3,
        "D" : 2.0,
        "a" : 0.0,
    }
    print(config)
    transportSystem = JAXNonlinearDiffusion(config)

    transportSystem.run(tFinal = 5.0)
    G, G_p = transportSystem.runAdjointSolve()
    print(G)
    print(G_p)
    # transportSystem.setParams(LinearDiffusionParams(0.1, 0.1, 2.0, 1.0))
    # #runner.setTransportSystem(transportSystem)

   
    # u = runner.run(10.0)
    # G, G_p = runner.runAdjointSolve()
    # print(u)
    #print(transportSystem.params)
    
    


runMaNTA()
