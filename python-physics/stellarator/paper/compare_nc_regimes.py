import matplotlib.pyplot as plt
import jax.numpy as jnp
from numpy import expand_dims
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


def load_and_rescale(name):
    eq = desc.examples.get(name)

    return desc.compat.rescale(
        eq, L=("a", 1.7), B=("<B>", 5.86), scale_pressure=False, copy=True, verbose=1
    )


# Magnetic field on a flux surface (here from a BOOZ_XFORM file).

eqs_high = desc.io.load("../eq_amb5_all_equilibria.h5")
eqs_low = desc.io.load("eq_all_equilibria.h5")

example_equilibria = ["precise_QH", "W7-X"]

# eq_qh = load_and_rescale("precise_QH")

# eq_aries = desc.examples.get("ARIES-CS")


# eq_w7x = load_and_rescale("W7-X")
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
    eqx.filter_vmap(compute_Dij, in_axes=(0, None, None), out_axes=(0, 0))
)

rhos = [0.5]
names = ["initial", "low col", "high col"] + example_equilibria
eqs = [eq0, eq_low, eq_high] + example_eqs
colors = ["r", "b", "g", "c", "m"]

plt.rcParams.update({"font.family": "serif", "font.size": 10})
fig, ax = plt.subplots(len(rhos), 2, sharex=True)
ax_flat = ax.flatten()
for rho, i in zip(rhos, range(0, len(rhos))):
    print(f"Solving at rho={rho}")
    for e, n, c in zip(eqs, names, colors):
        print(f"Solving mdke for {n}")
        field = Field.from_desc(e, rho, 25, 47)
        d11, d31 = fun(nuhats, field, species)
        ax[2 * i].loglog(nuhats, d11, label=n, color=c)
        ax[2 * i + 1].loglog(nuhats, jnp.abs(d31), label=n, color=c)
        ax[2 * i].axvline(_nustar(species[0], field, 1.0), color=c, linestyle="--")
        ax[2 * i + 1].axvline(_nustar(species[0], field, 1.0), color=c, linestyle="--")
print("done")

# for i in range(0, len(rhos)):
#     ax[i, 0].set_title(rf"$\rho=${rhos[i]}", loc="right")


for a in ax_flat:
    a.legend()
    a.set_box_aspect(1)
for a in ax_flat[-2:]:
    a.set_xlabel(r"$\nu^*$")
for a in ax_flat[0::2]:
    a.set_ylabel(r"$\hat{D}_{11}$")
for a in ax_flat[1::2]:
    a.set_ylabel(r"$|\hat{D}_{31}|$")

fig.tight_layout()
fig.savefig("figs/nc_mdke_compare.pdf", dpi=500)
plt.show()
