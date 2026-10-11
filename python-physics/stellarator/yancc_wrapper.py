import os
import jax

os.environ.pop("LD_LIBRARY_PATH", None)  # Required for Perlmutter to work properly

import yancc
from yancc.field import Field
from yancc.velocity_grids import MaxwellSpeedGrid, UniformPitchAngleGrid

import jax.numpy as jnp
from jax.tree_util import tree_map
import equinox as eqx
from jaxtyping import ArrayLike, Float

from typing import Optional


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
