import MaNTA
import numpy as np

class PythonLinearDiffusion(MaNTA.TransportSystem):
    def __init__(self, config):
        # Config should be a dict
        self.nVars = 1
        if not ("chi0" in config) and ("kappa" in config) and ("gamma" in config):
            print("For the stuff transport model, you must specify chi0, kappa, and gamma")
            sys.exit(1)


        self.Centre = 0.0
        self.InitialWidth = 0.2
        self.InitialHeight = 1.0
        self.kappa = config["kappa"]
        self.t0 = InitialWidth * InitialWidth / ( 4.0 * kappa );
        print("t0 = ",t0)

# This problem uses VN lower boundary and dirichlet upper boundary
# assumed to be on [0,1] in normalised flux
    def LowerBoundary(self, index, t):
        return self.ExactSolution( 0.0, t )
    def isLowerBoundaryDirichlet(self, index):
        return True
    def UpperBoundary(self, index, t):
        return self.ExactSolution( 1.0, t )
    def isUpperBoundaryDirichlet(self, index):
        return True

    def SigmaFn( self, index, uVals, qVals, x, t ):
        tprim = qVals[0]
        return self.kappa * tprim

    def Sources( self, index, u, q, sigma, x, t ):
        return 0

    def dSigmaFn_dq( self, index, out, u, q, x, t):
        out[0] = self.kappa
    def dSigmaFn_du( self, index, out, u, q, x, t):
        out[0] = 0.0
        
    def dSources_du( self, index, out, q, u, x, t ):
        out[0] = 0.0

    def dSources_dq( self, index, out, q, u, x, t ):
        out[0] = 0.0

    def dSources_dsigma( self, index, out, q, u, x, t ):
        out[0] = 0.0

    def InitialValue( self, index, x ):
        y = ( x - self.Centre )/self.InitialWidth
        return self.InitialHeight * np.exp( -y*y )

    def InitialDerivative( self, index, x ):
        y = ( x - self.Centre )/self.InitialWidth
        return self.InitialHeight * ( -2.0 * y ) * np.exp( -y*y ) * ( 1.0/self.InitialWidth )

    def ExactSolution( self, x, t ):
        EtaSquared = ( x - Centre )*( x - Centre )/( 4.0 * kappa * ( t + t0 ) );
        return self.InitialHeight * np.sqrt( self.t0/( t + self.t0 ) ) * np.exp( -EtaSquared );

def registerTransportSystems():
    MaNTA.registerPhysicsCase( "PythonLinearDiffusion", PythonLinearDiffusion.__init__ )

