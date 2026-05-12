import os 
import jax
os.environ.pop("LD_LIBRARY_PATH", None) # Required for Perlmutter to work properly

import yancc
from yancc.field import Field
from yancc.velocity_grids import MaxwellSpeedGrid, UniformPitchAngleGrid
from yancc.species import LocalMaxwellian
from yancc.solve import solve_dke

from scipy.constants import elementary_charge, mu_0, proton_mass
import jax.numpy as jnp
from jax.tree_util import tree_map
import equinox as eqx
from jaxtyping import Array, ArrayLike, Float, Int

from functools import partial
from typing import Optional

devices = jax.devices()


# Remove LD_LIBRARY_PATH to avoid conflicts with yancc's C++ extensions

import desc

import interpax

Lnorm = 1.0 # Normalization length in meters
Bnorm = 1.0 # Normalization magnetic field in Tesla

# Takes input MaNTA state, performs normalizations, returns fluxes
# Hold DESC equilibrium as well

class yancc_data(eqx.Module):
    """
    Create wrapper for yancc to interface with MaNTA, hold all field specific stuff
    Parameters
    ----------
    Density : f(Volume)
        the density isn't evolved yet, so it's just some prespecified function of the volume
    nNorm : float
        Normalization for density (m^-3)
    Tnorm : float
        Normalization for temperature (eV)

    """

    fields: eqx.Module
    fields_unstacked: list[eqx.Module] # list of field objects at each radial point 
    grid: eqx.Module
    pitchgrid: eqx.Module
    speedgrid: eqx.Module
    Vp: Float[ArrayLike, '...'] # dV/dr normalized by V[-1], function of volume only for now but can be more general in the future
    Vpp: Float[ArrayLike, '...']
    rho: Float[ArrayLike, '...']
    nNorm: float 
    Tnorm: float
    nx: int
    na: int 
    FluxNorm: float

    def __init__(
            self, 
            fields,
            grid, 
            Vp,
            Vpp,
            rho,  
            nNorm: Optional[float] = 1e20, 
            Tnorm: Optional[float] = 1e3, 
            nx: Optional[int] = 5, 
            na: Optional[int] = 65): 

        self.fields = fields
        self.grid = grid
        self.Vp = Vp
        self.Vpp = Vpp
        self.rho= rho
        self.nx = nx
        self.na = na

        self.nNorm = nNorm
        self.Tnorm = Tnorm

        Cs0 = jnp.sqrt(2 * Tnorm * elementary_charge / proton_mass)     # Normalization sound speed
        rho_star = (proton_mass * Cs0 / (elementary_charge * Bnorm)) / Lnorm  # Gyroradius

        tau_norm = rho_star ** 2 * Cs0 / Lnorm                          # Time normalization
        self.FluxNorm = nNorm * elementary_charge * Tnorm / tau_norm

        self.speedgrid = MaxwellSpeedGrid(nx)
        self.pitchgrid = UniformPitchAngleGrid(na)

        self.fields_unstacked = desc.backend.tree_unstack(fields)
        
        print("yancc_wrapper initialized successfully.")

    @classmethod
    def from_eq(cls, 
            rho: Float[ArrayLike, '...'],
            nNorm: Optional[float] = 1e20, 
            Tnorm: Optional[float] = 1e3, 
            nx: Optional[int] = 7, 
            na: Optional[int] = 65, 
            nt: Optional[int] = 17,
            nz: Optional[int] = 33,
            eq = None,
            grid = None):
        
        print("Initializing yancc wrapper")
        if (eq is None):
            print("No equilibrium passed, using ESTELL example")
            eq = desc.examples.get("W7-X")

        if (grid is None):
            grid = desc.grid.LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
       

        desc_data = eq.compute(["V(r)", "V_r(r)", "V_rr(r)"], grid=grid)
        V = grid.compress(desc_data['V(r)'])
        V_r = grid.compress(desc_data['V_r(r)'])/V[-1]
        V_rr = grid.compress(desc_data['V_rr(r)'])/V[-1]
        
        fields = []
        for r in rho:
            fields.append(Field.from_desc(eq, r, nt, nz))

        fields = tree_map(lambda *vals: jnp.stack(vals), *fields)
    
        return cls(fields=fields, grid = grid, Vp=V_r, Vpp=V_rr, nNorm=nNorm, Tnorm=Tnorm, nx=nx, na=na, rho=rho)

    # for constructing from data passed by DESC
    @classmethod
    def from_data(cls, data, grid, nNorm=1e20, Tnorm=1e3, nx=7, na=65):

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
            key: grid.meshgrid_reshape(val, "rtz") for key, val in yancc_dat.items()
        }

        yancc_dat["Psi"] = grid.compress(
            data["Psi"] / grid.nodes[:, 0] ** 2, surface_label="rho"
        )
        yancc_dat["a_minor"] = jnp.full(grid.num_rho, data["a"])
        yancc_dat["R_major"] = jnp.full(grid.num_rho, data["R0"])
        yancc_dat["iota"] = grid.compress(data["iota"], surface_label="rho")
        yancc_dat["rho"] = grid.compress(grid.nodes[:, 0], surface_label="rho")
        
        V = grid.compress(data['V(r)'])
        V_r = grid.compress(data['V_r(r)'])/V[-1]
        V_rr = grid.compress(data['V_rr(r)'])/V[-1]

        fields = jax.vmap(lambda d: yancc.field.Field(**d, NFP=grid.NFP))(yancc_dat)
        return cls(fields=fields, grid=grid,rho=yancc_dat["rho"], Vp=V_r, Vpp=V_rr, nNorm=nNorm, Tnorm=Tnorm, nx=nx, na=na)

    @classmethod
    def from_fields(cls, fields, grid, V_r, V_rr, nNorm=1e20, Tnorm=1e3, nx=7, na=65):
        return cls(fields=fields, grid=grid, rho=fields.rho, Vp = V_r, Vpp = V_rr, nNorm=nNorm, Tnorm=Tnorm, nx=nx, na=na)

    @classmethod 
    def from_other(cls, fields_, grid_, other):

        return cls(fields=fields_, grid=grid_, Vp = other.Vp, Vpp = other.Vpp, rho=other.rho, nNorm=other.nNorm, Tnorm=other.Tnorm, nx=other.nx, na=other.na)

    def get_fields(self):
        return self.fields, self.Vp, self.Vpp

# to avoid any surprises with jitting, we pass all the data as arguments rather than storing anything in the wrapper object
"""
Compute fluxes using yancc given the MaNTA state
Parameters
----------
state : dict
    Dictionary containing "Variable", "Derivative, "Flux", "Aux", and "Scalar"
Returns
-------
dict
    Fluxes computed by yancc, normalized to be dimensionless
"""
# @eqx.filter_jit
def flux(state, x, field, Vprim, n, nprime, yancc_params: yancc_data):
    # For now we only evolve the ion energy
    # print("tracing flux")
    p_i = 2. / 3. * state.Variable[0]
    p_i_prime = 2. / 3. * state.Derivative[0]

    dndrho = nprime * Vprim
    Erho = 0.0
    Ti = p_i / n
    dTidrho = (p_i_prime*Vprim - Ti*dndrho) / n
    species = [
    LocalMaxwellian(
        # can just give mass and charge in units of proton mass and elementary charge
        yancc.species.Species(1,1), 
        temperature=Ti * yancc_params.Tnorm, 
        density=n * yancc_params.nNorm, 
        dTdrho=dTidrho * yancc_params.Tnorm, 
        dndrho=dndrho * yancc_params.nNorm),
    ]
    _, _, fluxes, _  = solve_dke(field, yancc_params.pitchgrid, yancc_params.speedgrid, species, Erho, verbose = False)
    #assert stats['res'] < 1e-5
    # print(fluxes)
    fout = fluxes['<heat_flux>'][0] * Vprim / (yancc_params.FluxNorm)

    return fout
    






