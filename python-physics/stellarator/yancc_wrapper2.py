import os
import jax

os.environ.pop("LD_LIBRARY_PATH", None)  # Required for Perlmutter to work properly

import yancc
from yancc.field import Field
from yancc.velocity_grids import MaxwellSpeedGrid, UniformPitchAngleGrid
from yancc.species import LocalMaxwellian, Electron
from yancc.solve import solve_dke

import jax.numpy as jnp
from jax.tree_util import tree_map
import equinox as eqx
from jaxtyping import ArrayLike, Float

from typing import Optional
from stellarator_state import StellaratorParams, StellaratorState


import desc


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
    fields_unstacked: list[eqx.Module]  # list of field objects at each radial point
    grid: eqx.Module
    pitchgrid: eqx.Module
    speedgrid: eqx.Module
    Vp: Float[
        ArrayLike, "..."
    ]  # dV/dr normalized by V[-1], function of volume only for now but can be more general in the future
    Vpp: Float[ArrayLike, "..."]
    rho: Float[ArrayLike, "..."]
    nx: int
    na: int

    def __init__(
        self,
        fields,
        grid,
        Vp,
        Vpp,
        rho,
        nx: int = 5,
        na: int = 65,
    ):

        self.fields = fields
        self.grid = grid
        self.Vp = Vp
        self.Vpp = Vpp
        self.rho = rho

        self.speedgrid = MaxwellSpeedGrid(nx)
        self.pitchgrid = UniformPitchAngleGrid(na)

        self.na = na
        self.nx = nx

        self.fields_unstacked = desc.backend.tree_unstack(fields)

        print(
            f"yancc_wrapper initialized successfully with resolution na={self.na}, nx={self.nx}."
        )

    @classmethod
    def from_eq(
        cls,
        rho: Float[ArrayLike, "..."],
        nx: Optional[int] = 5,
        na: Optional[int] = 43,
        nt: Optional[int] = 17,
        nz: Optional[int] = 33,
        eq=None,
        grid=None,
    ):

        print("Initializing yancc wrapper")
        if eq is None:
            print("No equilibrium passed, using W7-X example")
            eq = desc.examples.get("W7-X")

        if grid is None:
            grid = desc.grid.LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)

        desc_data = eq.compute(["V(r)", "V_r(r)", "V_rr(r)"], grid=grid)
        V = grid.compress(desc_data["V(r)"])
        V_r = grid.compress(desc_data["V_r(r)"]) / V[-1]
        V_rr = grid.compress(desc_data["V_rr(r)"]) / V[-1]

        fields = []
        for r in rho:
            fields.append(Field.from_desc(eq, r, nt, nz))

        fields = tree_map(lambda *vals: jnp.stack(vals), *fields)

        return cls(
            fields=fields,
            grid=grid,
            Vp=V_r,
            Vpp=V_rr,
            nx=nx,
            na=na,
            rho=rho,
        )

    # for constructing from data passed by DESC
    @classmethod
    def from_data(cls, data, grid, nx=5, na=43):

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

        V = grid.compress(data["V(r)"])
        V_r = grid.compress(data["V_r(r)"]) / V[-1]
        V_rr = grid.compress(data["V_rr(r)"]) / V[-1]

        fields = jax.vmap(lambda d: yancc.field.Field(**d, NFP=grid.NFP))(yancc_dat)
        return cls(
            fields=fields,
            grid=grid,
            rho=yancc_dat["rho"],
            Vp=V_r,
            Vpp=V_rr,
            nx=nx,
            na=na,
        )

    @classmethod
    def from_fields(cls, fields, grid, V_r, V_rr, scale=1.0, nx=5, na=43):
        return cls(
            fields=fields,
            grid=grid,
            rho=fields.rho,
            Vp=V_r,
            Vpp=V_rr,
            nx=nx,
            na=na,
        )

    @classmethod
    def from_other(cls, fields_, grid_, other):
        return cls(
            fields=fields_,
            grid=grid_,
            Vp=other.Vp,
            Vpp=other.Vpp,
            rho=other.rho,
            nx=other.nx,
            na=other.na,
        )

    def get_fields(self):
        return self.fields, self.Vp, self.Vpp


"""
Computes physics information for stellarator model

Parameters
----------
index : int
    Variable index
state : eqx.Module
    Object containing state information
x : float
    Spatial location
t : float
    Time
field: yancc.Field
    Magnetic field object
vp: float
    V'
vpp: float
    V''
params : NamedTuple
    Transport system parameters, passed for JAX PyTree compatibility
Returns
-------
float
    Computed flux and aux terms
"""


# we keep this function pure and not dependent on class data
def compute_dke_sol(
    _state,
    x,
    t,
    field,
    vp,
    vpp,
    pitchgrid: UniformPitchAngleGrid,
    speedgrid: MaxwellSpeedGrid,
    params: StellaratorParams,
    evolveDensity=False,
):

    state = StellaratorState.from_state(_state, x, vp, vpp, params)

    def constant_density(
        state: StellaratorState,
        x,
        t,
        field,
        vp,
        vpp,
        pitchgrid,
        speedgrid,
        params: StellaratorParams,
    ):
        Erho = jnp.array(0.0)

        species = [
            LocalMaxwellian(
                params.constants.IonSpecies.yancc_species,
                temperature=state.Ti * params.constants.T0eV,
                density=state.n * params.constants.n0,
                dTdrho=state.dTidrho * params.constants.T0eV,
                dndrho=state.dndrho * params.constants.n0,
            ),
        ]

        sol, info = solve_dke(
            field,
            pitchgrid,
            speedgrid,
            species,
            Erho=Erho,
            # m=50,
            rtol=1e-3,
            throw=False,
            verbose=0,
            multigrid_options={"smooth_solver": "banded", "max_grids": 2},
        )
        flux = (
            -sol.get("<heat_flux>")[0]
            * vp
            / (params.constants.HeatEquationNormalization())
        )

        return [flux], []

    def ambipolar(
        state: StellaratorState,
        x,
        t,
        field,
        vp,
        vpp,
        pitchgrid,
        speedgrid,
        params: StellaratorParams,
    ):
        species = [
            LocalMaxwellian(
                params.constants.IonSpecies.yancc_species,
                temperature=state.Ti * params.constants.T0eV,
                density=state.n * params.constants.n0,
                dTdrho=state.dTidrho * params.constants.T0eV,
                dndrho=state.dndrho * params.constants.n0,
            ),
            LocalMaxwellian(
                Electron,
                temperature=state.Te * params.constants.T0eV,
                density=state.n * params.constants.n0,
                dTdrho=state.dTedrho * params.constants.T0eV,
                dndrho=state.dndrho * params.constants.n0,
            ),
        ]

        sol, info = solve_dke(
            field,
            pitchgrid,
            speedgrid,
            species,
            Erho=state.Er * params.constants.T0eV,
            # m=50,
            rtol=1e-3,
            throw=False,
            verbose=0,
            multigrid_options={"smooth_solver": "banded", "max_grids": 2},
        )

        particle_flux = (
            -sol.get("<particle_flux>")[1]
            * vp
            / (params.constants.DensityEquationNormalization())
        )

        heat_flux = (
            -sol.get("<heat_flux>")
            * vp
            / (params.constants.HeatEquationNormalization())
        )

        aux_g_out = vp * sol.get("J_rho") / (params.constants.CurrentNormalization())

        return [particle_flux, heat_flux[0], heat_flux[1]], [aux_g_out]

    if evolveDensity:
        return ambipolar(
            state,
            x,
            t,
            field,
            vp,
            vpp,
            pitchgrid,
            speedgrid,
            params,
        )
    else:
        return constant_density(
            state,
            x,
            t,
            field,
            vp,
            vpp,
            pitchgrid,
            speedgrid,
            params,
        )


def dke_field_jac(
    states,
    positions,
    t,
    field,
    vp,
    vpp,
    pitchgrid,
    speedgrid,
    params,
    evolveDensity=False,
):
    def _dke_sol(tree_in):
        _field, _vp, _vpp = tree_in
        return compute_dke_sol(
            states,
            positions,
            t,
            _field,
            _vp,
            _vpp,
            pitchgrid,
            speedgrid,
            params,
            evolveDensity,
        )

    return eqx.filter_jacrev(_dke_sol)((field, vp, vpp))


def test_flux(
    _state,
    x,
    t,
    field,
    vp,
    vpp,
    pitchgrid: UniformPitchAngleGrid,
    speedgrid: MaxwellSpeedGrid,
    params: StellaratorParams,
    evolveDensity=False,
):

    state = StellaratorState.from_state(_state, x, vp, vpp, params)

    def ambipolar(state, x, t, field, vp, vpp, *args):

        particle_flux = -vp * 0.01 * state.dndrho
        ion_heat_flux = -vp * 0.01 * state.dTidrho

        electron_heat_flux = -vp * 0.01 * state.dTedrho

        aux_g_out = vp * 100

        return [particle_flux, ion_heat_flux, electron_heat_flux], [aux_g_out]

    def constant_density(state, x, t, field, vp, vpp, *args):
        return [vp * 0.01 * state.dTidrho], []

    if evolveDensity:
        return ambipolar(state, x, t, field, vp, vpp, pitchgrid, speedgrid, params)
    else:
        return constant_density(
            state, x, t, field, vp, vpp, pitchgrid, speedgrid, params
        )


def test_flux_field_jac(
    states,
    positions,
    t,
    field,
    vp,
    vpp,
    pitchgrid,
    speedgrid,
    params,
    evolveDensity=False,
):
    def _dke_sol(tree_in):
        _field, _vp, _vpp = tree_in
        return test_flux(
            states,
            positions,
            t,
            _field,
            _vp,
            _vpp,
            pitchgrid,
            speedgrid,
            params,
            evolveDensity,
        )

    return eqx.filter_jacrev(_dke_sol)((field, vp, vpp))
