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
from jax.tree_util import tree_map

from functools import partial

# Remove LD_LIBRARY_PATH to avoid conflicts with yancc's C++ extensions

import desc
import interpax

Lnorm = 1.0 # Normalization length in meters
Bnorm = 1.0 # Normalization magnetic field in Tesla

# Takes input MaNTA state, performs normalizations, returns fluxes
# Hold DESC equilibrium as well

class yancc_wrapper():
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
    def __init__(self, Density, nNorm = 1e20, Tnorm = 1e3, nx = 5, na = 33, nt = 17, nz = 33):
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
        self.fields = []
        self.index = []
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
    def compute_field(self,x):
         rho = self.rho_from_normalized_volume(x)
         return Field.from_desc(self.eq, rho, self.nt, self.nz), rho, self.dVndr(rho)


    def compute_fields(self, x):
        if not self.fields:
            print("computing field")
            rho = []

            self.xvals = x
            i = 0
            for pos in x:
                rho.append(self.rho_from_normalized_volume(pos))
                self.fields.append(Field.from_desc(self.eq,rho[i],self.nt,self.nz))
                self.index.append(i)
                i+=1
            self.fields = tree_map(lambda *vals: jnp.stack(vals), *self.fields)
            self.rho = jnp.array(rho)
            self.Vprim = jnp.array(self.dVndr(self.rho))
        return self.fields, self.rho, self.Vprim

    

    def flux(self, state, x, field, rho, Vprim):
        
        # For now we only evolve the ion energy
        p_i = 2. / 3. * state["Variable"][0]
        p_i_prime = 2. / 3. * state["Derivative"][0]

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

        _, _, fluxes, stats  = solve_dke(field, self.pitchgrid, self.speedgrid, species, Erho, verbose = 0)
        #assert stats['res'] < 1e-5
        fout = fluxes['<heat_flux>'][0] * Vprim / (self.FluxNorm)
        return fout



    @partial(jax.jit, static_argnums=(0,))
    def rho_from_normalized_volume(self, Vnorm):
        return desc.backend.root_scalar(lambda x: self.Vn(x) - Vnorm, jnp.sqrt(Vnorm))
    






