import os 
import jax
os.environ.pop("LD_LIBRARY_PATH", None) # Required for Perlmutter to work properly

if os.environ["JAX_COMPILATION_CACHE_DIR"] is not None:
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

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
    Vprim: Float[ArrayLike, '...'] # dV/dr normalized by V[-1], function of volume only for now but can be more general in the future
    rho: Float[ArrayLike, '...']
    Density: callable = eqx.field(static=True) # function of volume only for now, can be more general in the future
    nNorm: float = eqx.field(static=True)
    Tnorm: float = eqx.field(static=True)
    nx: int = eqx.field(static=True)
    na: int = eqx.field(static=True)
    FluxNorm: float

    def __init__(
            self, 
            Density, 
            fields,
            grid, 
            Vprim,
            rho,  
            nNorm: Optional[float] = 1e20, 
            Tnorm: Optional[float] = 1e3, 
            nx: Optional[int] = 5, 
            na: Optional[int] = 43): 

        self.fields = fields
        self.grid = grid
        self.Vprim = Vprim
        self.rho=rho
        self.nx = nx
        self.na = na

        self.nNorm = nNorm
        self.Tnorm = Tnorm

        Cs0 = jnp.sqrt(2 * Tnorm * elementary_charge / proton_mass)     # Normalization sound speed
        rho_star = (proton_mass * Cs0 / (elementary_charge * Bnorm)) / Lnorm  # Gyroradius

        tau_norm = rho_star ** 2 * Cs0 / Lnorm                          # Time normalization
        self.FluxNorm = nNorm * elementary_charge * Tnorm / tau_norm
        self.Density = Density # Constant density for now

        self.speedgrid = MaxwellSpeedGrid(nx)
        self.pitchgrid = UniformPitchAngleGrid(na)

        self.fields_unstacked = desc.backend.tree_unstack(fields)
        
        print("yancc_wrapper initialized successfully.")

    @classmethod
    def from_eq(cls, 
            Volume: Float[ArrayLike, '...'] = eqx.field(static=True), 
            Density:  callable = eqx.field(static=True), 
            nNorm: Optional[float] = 1e20, 
            Tnorm: Optional[float] = 1e3, 
            nx: Optional[int] = 5, 
            na: Optional[int] = 43, 
            nt: Optional[int] = 17,
            nz: Optional[int] = 33,
            eq = None,
            grid = None,
            rho: Optional[Float[ArrayLike, '...']] = None):
        
        print("Initializing yancc wrapper")
        if (eq is None):
            eq = desc.examples.get("W7-X")

        if (grid is None):
            rho = jnp.linspace(0,1,len(Volume))
            grid = desc.grid.LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
       

        desc_data = eq.compute(["V(r)", "V_r(r)"], grid=grid)
        V = grid.compress(desc_data['V(r)'])
        Vn = V/V[-1] # normalize
        dVdr = grid.compress(desc_data['V_r(r)'])
        dVndr = dVdr/V[-1] # normalize
        Vn = interpax.CubicSpline(rho, Vn)
        dVndr = interpax.CubicSpline(rho, dVndr)
        rho = Vn
        fields = []
        r = []
        i = 0
        rho_from_normalized_volume = lambda Vnorm : desc.backend.root_scalar(lambda x: Vn(x) - Vnorm, jnp.sqrt(Vnorm))
        for pos in Volume:
            r.append(rho_from_normalized_volume(pos))
            fields.append(Field.from_desc(eq, r[i], nt, nz))
            i+=1

        fields = tree_map(lambda *vals: jnp.stack(vals), *fields)
        r = jnp.array(r)
        Vprim = jnp.array(dVndr(r))

        return cls(Density=Density, fields=fields, grid = grid, Vprim=Vprim, nNorm=nNorm, Tnorm=Tnorm, nx=nx, na=na, rho=r)

    @classmethod
    def from_fields(cls, fields, grid, Density, nNorm=1e20, Tnorm=1e3, nx=5, na=43):

        V = grid.compress(fields['V(r)'])
        dVdr = grid.compress(fields['V_r(r)'])
        dVndr = dVdr/V[-1] # normalize

        return cls(Density=Density, fields=fields, grid=grid, Vprim = dVndr, nNorm=nNorm, Tnorm=Tnorm, nx=nx, na=na)

    @classmethod 
    def from_other(cls, fields, grid, other):
        return cls(Density=other.Density, fields=fields, grid=grid, Vprim = other.Vprim, rho=other.rho, nNorm=other.nNorm, Tnorm=other.Tnorm, nx=other.nx, na=other.na)

    def get_fields(self):
        return self.fields, self.Vprim


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

@eqx.filter_jit
def flux(state, x, field, Vprim, yancc_params: yancc_data):
    # For now we only evolve the ion energy
    p_i = 2. / 3. * state["Variable"][0]
    p_i_prime = 2. / 3. * state["Derivative"][0]

    dndrho = jax.grad(yancc_params.Density, argnums=0)(x)*Vprim
    Erho = 0.0
    Ti = p_i / yancc_params.Density(x)
    dTidrho = (p_i_prime*Vprim - Ti*dndrho) / yancc_params.Density(x)
    species = [
    LocalMaxwellian(
        # can just give mass and charge in units of proton mass and elementary charge
        yancc.species.Species(1,1), 
        temperature=Ti * yancc_params.Tnorm, 
        density=yancc_params.Density(x) * yancc_params.nNorm, 
        dTdrho=dTidrho * yancc_params.Tnorm, 
        dndrho=dndrho * yancc_params.nNorm),
    ]

    _, _, fluxes, _  = solve_dke(field, yancc_params.pitchgrid, yancc_params.speedgrid, species, Erho, verbose = False)
    #assert stats['res'] < 1e-5
    fout = fluxes['<heat_flux>'][0] * Vprim / (yancc_params.FluxNorm)
    return fout
    






