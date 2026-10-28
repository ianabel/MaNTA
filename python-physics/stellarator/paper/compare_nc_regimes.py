import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import netCDF4
from yancc import Field, UniformPitchAngleGrid, solve_mdke
from yancc.species import Electron, _nustar, _Estar, LocalMaxwellian
import desc.io
from desc.grid import LinearGrid
import manta
import jax
from jax.sharding import PartitionSpec, NamedSharding
import equinox as eqx

P = PartitionSpec
devices = jax.devices()
print(devices)
mesh = jax.make_mesh(
    (jax.device_count(),), ("axis",), axis_types=(jax.sharding.AxisType.Auto,)
)
data_sharding = NamedSharding(
    mesh,
    P(
        "axis",
    ),
)


def load_and_rescale(name):
    eq = desc.examples.get(name)

    return desc.compat.rescale(
        eq, L=("a", 1.7), B=("<B>", 5.86), scale_pressure=False, copy=True, verbose=1
    )


# Magnetic field on a flux surface (here from a BOOZ_XFORM file).

eqs_high = desc.io.load("eq4_all_equilibria.h5")
eqs_low = desc.io.load("eq5_all_equilibria.h5")

# The MaNTA run behind each optimisation: desc-optimize-stored-energy.py's
# first run, <fname>_acc.nc, on the family's initial equilibrium, which is
# therefore the one V' is taken from; and the Normalization it was run with
# (n0 in m^-3, T0 in eV)
manta_runs = {
    "A": ("stellarator_opteq4_acc.nc", eqs_high[0], {"n0": 1e19, "T0": 1000.0}),
    "B": ("stellarator_opteq5_acc.nc", eqs_low[0], {"n0": 1e19, "T0": 1000.0}),
}


def manta_local_state(nc_file, eq, rho, normalization):
    """Electron density [m^-3], temperature [eV] and E_rho [V] at rho, last
    time slice.

    The transported variables are V' n and (3/2) V' n T_e, with
    V' = V_r(rho) / V(rho_last) and rho_last the last MaNTA node, as the
    drivers build it (objective_from_user_fun), so T_e needs no V' and n
    takes it from `eq`. Er is in T0/e per unit rho: under ErModel = Diffusive
    the variable V' Er (an "Er" group), otherwise the auxiliary Er itself (a
    top-level variable). It reaches yancc as Er * T0, as in dke_sol.py.
    """
    with netCDF4.Dataset(nc_file) as d:
        x = d["x"][:]
        Vn = np.interp(rho, x, d["Density/u"][-1, :])
        Vue = np.interp(rho, x, d["ElectronEnergy/u"][-1, :])
        if "Er" in d.groups:
            Er, Er_has_Vp = np.interp(rho, x, d["Er/u"][-1, :]), True
        else:
            Er, Er_has_Vp = np.interp(rho, x, d["Er"][-1, :]), False
        nodes = manta.getNodes(
            np.asarray(d["Grid/CellBoundaries"][:]), int(d["Grid/PolyOrder"][...])
        )

    def at(rho_, key):
        grid = LinearGrid(rho=np.array([rho_]), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
        return float(grid.compress(eq.compute(key, grid=grid)[key])[0])

    Vp = at(rho, "V_r(r)") / at(nodes[-1], "V(r)")
    n = Vn / Vp * normalization["n0"]
    Te = 2.0 / 3.0 * Vue / Vn * normalization["T0"]
    Erho = (Er / Vp if Er_has_Vp else Er) * normalization["T0"]
    return n, Te, Erho


# example_equilibria = ["precise_QH", "W7-X"]
example_equilibria = []

# eq_qh = load_and_rescale("precise_QH")

# eq_aries = desc.examples.get("ARIES-CS")


eq_w7x = load_and_rescale("W7-X")
#
example_eqs = []
for name in example_equilibria:
    example_eqs.append(load_and_rescale(name))

eq0 = eqs_low[0]
eq_low = eqs_low[-1]
eq_high = eqs_high[-1]

species = [
    LocalMaxwellian(
        Electron,
        temperature=15.0e3,  # eV
        density=1.0e20,  # 1/m^3
        dTdrho=-2.0e3,
        dndrho=-0.4e20,
    )
]


def compute_Dij(nu, field, species, Er=-1e3):
    # Uniform finite-difference grid in pitch angle.
    pitchgrid = UniformPitchAngleGrid(95)
    # Monoenergetic drives.
    erhohat = _Estar(species[0], field, Er, 1.0)  # E_rho / v   [V*s/m]

    sol, _ = solve_mdke(
        field,
        pitchgrid,
        erhohat,
        nu,
        verbose=0,
        multigrid_options={"smooth_solver": "banded", "max_grids": 3},
    )

    Dij = sol.get("Dij")
    return Dij[0, 0], Dij[2, 0]


nuhats = jax.device_put(jnp.logspace(-7, 0, 20), data_sharding)

fun = eqx.filter_jit(
    eqx.filter_vmap(compute_Dij, in_axes=(0, None, None, None), out_axes=(0, 0))
)

rhos = [0.5, 0.9]
names = ["initial", "A", "B", "W7-X"] + example_equilibria
eqs = [eq0, eq_high, eq_low, eq_w7x] + example_eqs
colors = ["r", "b", "g", "c", "m"]

plt.rcParams.update({"font.family": "serif", "font.size": 10})

fig = plt.figure(figsize=(5, 2.5 * len(rhos)), layout="constrained")
subfigs = fig.subfigures(len(rhos), 1, wspace=0, hspace=0.1)

for rho, i in zip(rhos, range(0, len(rhos))):
    print(f"Solving at rho={rho}")
    axs = subfigs[i].subplots(1, 2, sharex=True)
    for e, n, c in zip(eqs, names, colors):
        print(f"Solving mdke for {n}")
        field = Field.from_desc(e, rho, 25, 47)
        local = None
        if n in manta_runs:
            # the electrons MaNTA reached at this rho: their Er drives the
            # mdke, and their nu* places the line. Neither reads the
            # gradients
            nc_file, run_eq, normalization = manta_runs[n]
            ne, Te, Erho = manta_local_state(nc_file, run_eq, rho, normalization)
            print(
                f"{n}: n_e = {ne:.3e} m^-3, T_e = {Te:.1f} eV, "
                f"E_rho = {Erho:.3e} V at rho={rho}"
            )
            local = LocalMaxwellian(
                Electron, temperature=Te, density=ne, dTdrho=0.0, dndrho=0.0
            )
            d11, d31 = fun(nuhats, field, [local], Erho)
        else:
            d11, d31 = fun(nuhats, field, species, -1e3)
        axs[0].loglog(nuhats, d11, label=n, color=c)
        axs[1].loglog(nuhats, jnp.abs(d31), label=n, color=c)
        if local is not None:
            nustar = _nustar(local, field, 1.0)
            axs[0].axvline(nustar, color=c, linestyle="--")
            axs[1].axvline(nustar, color=c, linestyle="--")

    subfigs[i].suptitle(rf"$\rho={rho}$", x=0.5, fontsize=10)
    axs[0].set_ylabel(r"$\hat{D}_{11}$")
    axs[1].set_ylabel(r"$|\hat{D}_{31}|$")
    for a in axs:    
        a.legend() 
        a.set_box_aspect(1)

    if i == len(rhos) - 1:
        axs[0].set_xlabel(r"$\nu^*$")
        axs[1].set_xlabel(r"$\nu^*$")

print("done")

# for i in range(0, len(rhos)):
#     ax[i, 0].set_title(rf"$\rho=${rhos[i]}", loc="right")


# for a in ax_flat:
#     a.legend()
#     a.set_box_aspect(1)
# for a in ax_flat[-2:]:
#     a.set_xlabel(r"$\nu^*$")
# for a in ax_flat[0::2]:
#     a.set_ylabel(r"$\hat{D}_{11}$")
# for a in ax_flat[1::2]:
#     a.set_ylabel(r"$|\hat{D}_{31}|$")

# fig.set_figheight(2.5 * len(rhos))
# fig.set_figwidth(5.0)
fig.savefig("figs/nc_mdke_compare.pdf", dpi=500)
plt.show()
