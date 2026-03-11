from desc.objectives import ObjectiveFromUser

from jax import custom_jvp
from functools import partial

import MaNTA
from Stellarator import StellaratorTransport

class StellaratorObjective(ObjectiveFromUser):
    def __init__(self, config, eq):
        super().__init__(fun=self.RunObjective, thing=eq)

        # Create solver object
        st = StellaratorTransport(config["TransportSystem"])

        self.runner = MaNTA.Runner(st)
        self.runner.configure(config["Solver"])


    @partial(custom_jvp, nondiff_argnums=(0,1))
    def RunObjective(self, grid, field):
        
        ## set field within yancc wrapper
        self.st.yancc_wrapper.setField(field)
        ## what to do with grid??

        
        # run to steady state
        self.runner.run_ss()
        G, G_p = self.runner.runAdjointSolve()
        # compute adjoint
        # multiply field by G_p
        # postprocess G_p? 
        return G, G_p
    
    @RunObjective.devjvp
    def fun_jvp(self, grid, primals, tangents):
        field = primals
        field_dot = tangents

        G, G_p = self.RunObjective(grid, field)

        return G, G_p * field_dot


