import MaNTA
import numpy as np
import sys

# Solves
#
# d_t u + d_x ( (a/u^(3/2)) * du/dx ) = S(x)
#
# with S(x) = d * exp( -( x - c )^2 / b )
#
# on the domain x in [0,1] ; t in [0,T]
# u(x,0) = u1
# u(1,t) = u1 ; Dirichlet b.c.
# u'(0,t) = 0 ; Neumann b.c.
#
# We default to a = 6, b = 0.02, c = 0.3, d = 50.0, u1 = 0.3
#
# This has an exact steady state solution

class PythonToyModel(MaNTA.TransportSystem):
    def __init__(self, config, grid):
        MaNTA.TransportSystem.__init__(self)
        # Only one variable, no scalars
        self.nVars = 1
        self.nScalars = 0
        # Really should sanitize input here, c.f.
        # if not ("chi0" in config) and ("kappa" in config) and ("gamma" in config):
        #     print("For the stuff transport model, you must specify chi0, kappa, and gamma")
        #     sys.exit(1)
        self.isUpperDirichlet  = True
        self.isLowerDirichlet  = False

        self.a = 6.0
        self.b = 0.02
        self.c = 0.3
        seld.d = 50.0

        self.u1 = 0.3

# This problem uses VN lower boundary and dirichlet upper boundary
# assumed to be on [0,1]
    def LowerBoundary(self, index, t):
        return 0.0
    def UpperBoundary(self, index, t):
        return u1

    """
    Sigma_v and Sources_v, are vectorised calls to the flux and source functions

    Parameters
    ----------
    index : int
        Variable index
    states : array of dictionarie
        Element i is a dictionary containing "Variable", "Derivative, "Flux", "Aux", and "Scalar" arrays
        which describe the system state at point i
    positions : array of float
        Spatial locations where data is required, indexing corresponds to the states array
    t : float
        Time
    Returns
    -------
    float
        Computed sigma or source term
    """
    def SigmaFn_v( self, index, states, positions, t):
        nPoints = len(positions)
        SigmaVals = np.empty( nPoints )
        for i in nPoints:
            state = states[i]
            # [0] needed on the end as state['Variables'] is a one-element array because there's one variable
            u    = state['Variable'][0]
            dudx = state['Derivative'][0]
            flux = ( self.a / np.pow( u, 1.5 ) ) * dudx
            SigmaVals[i] = flux
        return SigmaVals

    # S(x) = d * exp( -( x - c )^2 / b )
    def Sources_v( self, index, states, positions, t ):
        nPoints = len(positions)
        SourceVals = np.empty( nPoints )
        for i in nPoints:
            x = positions[i]
            SourceVals[i] = self.d * np.exp( -( x - self.c ) ** 2 / self.b )
        return SourceVals

    def dSigma(self, index, states, positions, t):
    def dSources(self, index, states, positions, t):

    # We need initial u and du/dx at t=0
    def InitialValue( self, index, x ):
        return self.u1

    def InitialDerivative( self, index, x ):
        return 0.0

    # For testing
    def ExactSteadyStateSolution( self, x ):
        a = 6.0
        b = 0.02
        c = 0.3
        d = 50.0
        u1 = 0.3
        y = (x - c)/np.sqrt(b)
        G = (b*d/(4*a)) * ( np.exp( -(1-c)**2/b ) - np.exp( -y**2 ) ) + (d*np.sqrt( b*np.pi )/(4*a)) * ( (c-1)*scipy.special.erf( (c-1)/np.sqrt(b) ) + (1-x)*scipy.special.erf(c/np.sqrt(b)) - (x-c)*scipy.special.erf(y) )
        u2 = 1.0/np.sqrt(u1) - G
        return 1.0/(u2**2)

def registerTransportSystems():
    MaNTA.registerPhysicsCase("ToyModel1", PythonLinearDiffusion)

