# A Python case in a package of its own, run by the `manta` command from an
# installed package. docs/out_of_tree.rst includes this file, and CI's
# out-of-tree job runs it, so the example in the documentation is one that is
# known to work.
import manta
import numpy as np


class MyDiffusion(manta.TransportSystem):
    variables = [manta.Field("temperature", "electron temperature", "eV")]

    def __init__(self, config, grid):
        super().__init__()
        # The whole configuration file arrives; this reads [MyDiffusion].
        self.kappa = config["MyDiffusion"]["Kappa"]

    def SigmaFn(self, i, state, x, t):
        return self.kappa * state.q[0]

    def Sources(self, i, state, x, t):
        return 1.0

    def dSigmaFn_dq(self, i, state, x, t):
        return np.full(self.nVars, self.kappa)

    def LowerBoundary(self, i, t):
        return 0.0

    def UpperBoundary(self, i, t):
        return 0.0

    def InitialValue(self, i, x):
        return 0.0

    def InitialDerivative(self, i, x):
        return 0.0


manta.registerPhysicsCase("MyDiffusion", MyDiffusion)
