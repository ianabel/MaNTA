from desc.objectives import ObjectiveFromUser
import jax.numpy as jnp
import jax
from jax import custom_jvp
from functools import partial

import MaNTA
from Stellarator import StellaratorTransport

class StellaratorObjective:
    def __init__(self, config, eq):
       

        # Create solver object
        st = StellaratorTransport(config)
        self.grid = st.yancc_wrapper.get_grid()

        self.StoredEnergyObjective = ObjectiveFromUser(fun=self.StoredEnergy, thing=eq)
        self.ProfileErrorObjective = ObjectiveFromUser(fun=self.ProfileError, thing=eq)


    @partial(custom_jvp, nondiff_argnums=(0,1))
    def StoredEnergy(self, grid, field):
        
        ## set field within yancc wrapper
        ## what to do with grid??

        
        # run to steady state
        self.st.run(field=field)
        G, _ = self.st.runAdjointSolve()
        # compute adjoint
        # multiply field by G_p
        # postprocess G_p? 
        return G[0]
    
    @StoredEnergy.devjvp
    def StoredEnergy_jvp(self, grid, primals, tangents):
        field, = primals
        field_dot, = tangents

        G, G_p = self.st.runAdjointSolve(grid, field)   

        field_dot_flatten = jax.flatten_util.ravel_pytree(field_dot)

        return G[0], jnp.dot(G_p[0, :], field_dot_flatten)

    @partial(custom_jvp, nondiff_argnums=(0,1))
    def ProfileError(self, grid, field):
        
        ## set field within yancc wrapper
        ## what to do with grid??

        
        # run to steady state
        self.st.run(field=field)
        G, _= self.st.runAdjointSolve()
        # compute adjoint
        # multiply field by G_p
        # postprocess G_p? 
        return G[1]
    
    @ProfileError.devjvp
    def ProfileError_jvp(self, grid, primals, tangents):
        field, = primals
        field_dot, = tangents

        G, G_p = self.st.runAdjointSolve(grid, field)   

        field_dot_flatten = jax.flatten_util.ravel_pytree(field_dot)

        return G[1], jnp.dot(G_p[1, :], field_dot_flatten)



