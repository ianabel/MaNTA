import MaNTA

class StiffTransport(TransportSystem):
    def __init__(self, config):
        # Config should be a dict
        self.nVars = 1
        if not ("chi0" in config) and ("kappa" in config) and ("gamma" in config):
            print("For the stuff transport model, you must specify chi0, kappa, and gamma")
            sys.exit(1)

        self.chi0 = config["chi0"]
        self.kappa = config["kappa"]
        self.gamma = config["gamma"]
        selc.tprim_crit = config["CriticalGradient"]
        self.EdgeValue = config["EdgeTemperature"]

# This problem uses VN lower boundary and dirichlet upper boundary
# assumed to be on [0,1] in normalised flux
    def LowerBoundary(self, index, time):
        return 0.0
    def isLowerBoundaryDirichlet(self, index):
        return False
    def UpperBoundary(self, index, time):
        return self.EdgeValue
    def isUpperBoundaryDirichlet(self, index):
        return True;

    def SigmaFn( self, index, uVals, qVals, x, t ):
        tprim = qVals[0]
        if( tprim < self.tprim_crit ):
            return self.chi0 * tprim; # We return a flux, not a diffusivity
        else:
            return self.chi0 + self.kappa * pow( abs(tprim) - self.tprim_crit , gamma);

    def Sources( self, index, u, q, sigma, x, t ):
        return 0;

    def dSigmaFn_dq( self, index, out, u, q, x, t):
        if( q[0] < self.tprim_crit ):
            out[0] = 0.0
        else:
            out[0] = self.gamma * self.kappa * pow( abs(q[0]) - self.tprim_crit, gamma - 1 )
    def dSigmaFn_du( self, index, out, u, q, x, t):
        out[0] = 0.0
        
    def dSources_du( index, out, q, u, x, t ):
        out[0] = 0.0

    def dSources_dq( index, out, q, u, x, t ):
        out[0] = 0.0

    def dSources_dsigma( index, out, q, u, x, t ):
        out[0] = 0.0

    def InitialValue( index, x ):

    def InitialDerivative( index, x ):

