import matplotlib.pyplot as plt
import jax.numpy as jnp
from yancc import Field, UniformPitchAngleGrid, solve_mdke
from yancc.species import Electron, _nustar, _Estar, LocalMaxwellian
import desc.io
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
# Magnetic field on a flux surface (here from a BOOZ_XFORM file).
eq = desc.io.load("../eq_amb5_all_equilibria.h5")
eq0 = desc.examples.get("precise_QH")
# eq = desc.examples.get("W7-X")

eq2 = desc.compat.rescale(
    eq0, L=("a", 1.7), B=("<B>", 5.86), scale_pressure=False, copy=True, verbose=1
)
eq_aries = desc.examples.get("ARIES-CS")
eq0 = eq[0]
eq1 = eq[-1]
eq1.solve(x_scale="ess")

species = [
    LocalMaxwellian(
        Electron,
        temperature=5.0e3,  # eV
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
    eqx.filter_vmap(compute_Dij, in_axes=(0, None, None), out_axes=(0, 0))
)

rhos = [0.5, 0.9]
names = ["initial", "optimized", "QH", "ARIES-CS"]
eqs = [eq0, eq1, eq2, eq_aries]
colors = ["r", "b", "g", "c"]

plt.rcParams.update({"font.family": "serif", "font.size": 10})
fig, ax = plt.subplots(len(rhos), 2, sharex=True)
for rho, i in zip(rhos, range(0, len(rhos))):
    print(f"Solving at rho={rho}")
    for e, n, c in zip(eqs, names, colors):
        field = Field.from_desc(e, rho, 25, 47)
        d11, d31 = fun(nuhats, field, species)
        ax[i, 0].loglog(nuhats, d11, label=n, color=c)
        ax[i, 1].loglog(nuhats, jnp.abs(d31), label=n, color=c)
        ax[i, 0].axvline(_nustar(species[0], field, 1.0), color=c, linestyle="--")
        ax[i, 1].axvline(_nustar(species[0], field, 1.0), color=c, linestyle="--")
print("done")


for a in ax.flatten():
    a.legend()
    a.set_box_aspect(1)
for a in ax[-1, :]:
    a.set_xlabel(r"$\nu^*$")
for a in ax[:, 0]:
    a.set_ylabel(r"$D_{11}$")
for a in ax[:, 1]:
    a.set_ylabel(r"$|D_{31}|$")

fig.tight_layout()
fig.savefig("nc_mdke_compare.png", dpi=500)
plt.show()
