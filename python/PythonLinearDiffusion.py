import MaNTA
import numpy as np
import sys

class PythonLinearDiffusion(MaNTA.TransportSystem):
    def __init__(self, config, grid):
        MaNTA.TransportSystem.__init__(self)
        self.nVars = 1
        # Really should sanitize input here, c.f.
        # if not ("chi0" in config) and ("kappa" in config) and ("gamma" in config):
        #     print("For the stuff transport model, you must specify chi0, kappa, and gamma")
        #     sys.exit(1)
        self.isUpperDirichlet  = True
        self.isLowerDirichlet  = True

        self.Centre = 0.0
        self.InitialWidth = 0.2
        self.InitialHeight = 1.0
        self.kappa = 1.0
        self.t0 = self.InitialWidth * self.InitialWidth / ( 4.0 * self.kappa )

# This problem uses VN lower boundary and dirichlet upper boundary
# assumed to be on [0,1] in normalised flux
    def LowerBoundary(self, index, t):
        return 0.0
    def UpperBoundary(self, index, t):
        return 0.0

    def SigmaFn( self, index, state, x, t ):
        tprim = state["Derivative"][0]
        return self.kappa * tprim

    def Sources( self, index, state, x, t ):
        return 0

    def dSigmaFn_dq( self, index, state, x, t):
        out = np.zeros(shape=(self.nVars,))
        out[0] = self.kappa
        return out
    
    def dSigmaFn_du( self, index, state, x, t):
        out = np.zeros(shape=(self.nVars,))
        return out
        
    def dSources_du( self, index, state, x, t ):
        out = np.zeros(shape=(self.nVars,))
        return out

    def dSources_dq( self, index, state, x, t ):
        out = np.zeros(shape=(self.nVars,))
        return out

    def dSources_dsigma( self, index, state, x, t ):
        out = np.zeros(shape=(self.nVars,))
        return out

    def InitialValue( self, index, x ):
        alpha = 1 / self.InitialWidth
        y = (x - self.Centre)
        return self.InitialHeight * np.exp(-alpha * y * y)

    def InitialDerivative( self, index, x ):
        y = (x - self.Centre)
        alpha = 1 / self.InitialWidth
        return -self.InitialHeight * (2.0 * y) * np.exp(-alpha * y * y) * alpha

    def ExactSolution( self, x, t ):
        EtaSquared = ( x - self.Centre )*( x - self.Centre )/( 4.0 * self.kappa * ( t + self.t0 ) )
        return self.InitialHeight * np.sqrt( self.t0/( t + self.t0 ) ) * np.exp( -EtaSquared )


def registerTransportSystems():
    MaNTA.registerPhysicsCase("PythonLinearDiffusion", PythonLinearDiffusion)

