from desc.objectives import ObjectiveFromUser
import jax.numpy as jnp
import jax
from jax import custom_jvp
from functools import partial

import MaNTA
from Stellarator import StellaratorTransport

import desc
import yancc
# optimization is easiest for least squares objectives, so instead of maximizing
# stored energy we minimize 1/stored_energy^2 (the squaring happens later)

class StellaratorObjective:
    def __init__(self, config, eq):
       
        # Create solver object
        st = StellaratorTransport(config)
        self.grid = st.yancc_wrapper.get_grid()

        self.StoredEnergyObjective = ObjectiveFromUser(fun=self.StoredEnergy, thing=eq)
        self.ProfileErrorObjective = ObjectiveFromUser(fun=self.ProfileError, thing=eq)

    def data_to_yancc(self,data):
        yancc_dat = {
        "B_sup_t": data["B^theta"],
        "B_sup_z": data["B^zeta"],
        "B_sub_t": data["B_theta"],
        "B_sub_z": data["B_zeta"],
        "Bmag": data["|B|"],
        "dBdt": data["|B|_t"],
        "dBdz": data["|B|_z"],
        "sqrtg": data["sqrt(g)"],
        }

        yancc_dat = {
            key: self.grid.meshgrid_reshape(val, "rtz") for key, val in yancc_dat.items()
        }

        yancc_dat["Psi"] = self.grid.compress(
            data["Psi"] / self.grid.nodes[:, 0] ** 2, surface_label="rho"
        )
        yancc_dat["a_minor"] = jnp.full(self.grid.num_rho, data["a"])
        yancc_dat["R_major"] = jnp.full(self.grid.num_rho, data["R0"])
        yancc_dat["iota"] = self.grid.compress(data["iota"], surface_label="rho")
        yancc_dat["rho"] = self.grid.compress(self.grid.nodes[:, 0], surface_label="rho")

        yancc_fields = jax.vmap(lambda d: yancc.field.Field(**d, NFP=self.grid.NFP))(yancc_dat)
        yancc_fields = desc.backend.tree_unstack(yancc_fields)

        return yancc_fields

    @partial(custom_jvp, nondiff_argnums=(0,1))
    def StoredEnergy(self, grid, field):
        
        ## set field within yancc wrapper
        ## what to do with grid??
        yancc_fields = self.data_to_yancc(field)
        
        # run to steady state
        self.st.run(field=yancc_fields)
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

        return G[0], jnp.dot(G_p.flatten(), field_dot_flatten)



