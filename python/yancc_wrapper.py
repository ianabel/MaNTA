import os 
os.environ.pop("LD_LIBRARY_PATH", None) # Required for Perlmutter to work properly

import yancc
from yancc.field import Field
from yancc.velocity_grids import MaxwellSpeedGrid, UniformPitchAngleGrid
from yancc.species import LocalMaxwellian
from yancc.solve import solve_dke

from scipy.constants import elementary_charge, proton_mass

import jax

import jax.numpy as jnp

from functools import partial

# Remove LD_LIBRARY_PATH to avoid conflicts with yancc's C++ extensions

import desc
import interpax

Lnorm = 1.0 # Normalization length in meters
Bnorm = 1.0 # Normalization magnetic field in Tesla

# Takes input MaNTA state, performs normalizations, returns fluxes
# Hold DESC equilibrium as well

class yancc_wrapper():
    def __init__(self, Density, nNorm, Tnorm, nx = 5, na = 65, nt = 17, nz = 33):
        print("Initializing yancc wrapper with parameters:")
        print(f"  nx={nx}, na={na}, nt={nt}, nz={nz}")
        self.nx = nx
        self.na = na
        self.nt = nt
        self.nz = nz

        self.nNorm = nNorm
        self.Tnorm = Tnorm

        Cs0 = jnp.sqrt(2 * Tnorm * elementary_charge / proton_mass)     # Normalization sound speed
        rho_star = (proton_mass * Cs0 / (elementary_charge * Bnorm)) / Lnorm  # Gyroradius

        tau_norm = rho_star ** 2 * Cs0 / Lnorm                          # Time normalization
        self.FluxNorm = nNorm * elementary_charge * Tnorm / tau_norm
        self.Density = Density # Constant density for now
        
        self.eq = desc.examples.get("W7-X")
        r = jnp.linspace(0,1,20)
        desc_grid = desc.grid.LinearGrid(rho=r, M=self.eq.M_grid, N=self.eq.N_grid, NFP=self.eq.NFP)
        desc_data = self.eq.compute(["V(r)", "V_r(r)"], grid=desc_grid)
        V = desc_grid.compress(desc_data['V(r)'])
        Vn = V/V[-1] # normalize
        dVdr = desc_grid.compress(desc_data['V_r(r)'])
        dVndr = dVdr/V[-1] # normalize
        self.Vn = interpax.CubicSpline(r, Vn)
        self.dVndr = interpax.CubicSpline(r, dVndr)

        self.speedgrid = MaxwellSpeedGrid(nx)
        self.pitchgrid = UniformPitchAngleGrid(na)
        print("yancc wrapper initialized successfully.")

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
    def flux(self, state, x):
    
        p_i = 2. / 3. * state["Variable"][0]
        p_i_prime = 2. / 3. * state["Derivative"][0]

        rho = self.rho_from_normalized_volume(x)
        Vprim = self.dVndr(rho)

        dndrho = jax.grad(self.Density, argnums=0)(x)*Vprim
        Erho = 0.0
        Ti = p_i / self.Density(x)
        dTidrho = (p_i_prime*Vprim - Ti*dndrho) / self.Density(x)
        species = [
        LocalMaxwellian(
            # can just give mass and charge in units of proton mass and elementary charge
            yancc.species.Species(1,1), 
            temperature=Ti * self.Tnorm, 
            density=self.Density(x) * self.nNorm, 
            dTdrho=dTidrho * self.Tnorm, 
            dndrho=dndrho * self.nNorm),
        ]
         
        field = Field.from_desc(self.eq, rho, self.nt, self.nz)


        _, _, fluxes, stats  = solve_dke(field, self.pitchgrid, self.speedgrid, species, Erho, print_every=10)
        assert stats['res'] < 1e-5
        fout = fluxes['<heat_flux>'][0] * Vprim / (self.FluxNorm)
        return self.Density(x) * fout

    @partial(jax.jit, static_argnums=(0,))
    def rho_from_normalized_volume(self, Vnorm):
        return desc.backend.root_scalar(lambda x: self.Vn(x) - Vnorm, jnp.sqrt(Vnorm))
    






