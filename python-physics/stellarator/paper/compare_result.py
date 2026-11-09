import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import desc.io

plt.rcParams.update({"font.family": "serif", "font.size": 8})

import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load(eq, fname):

    data = Dataset(fname)
    x = np.array(data.variables["x"])
    nv = np.array(data.groups["Density"].variables["u"])
    uiv = np.array(data.groups["IonEnergy"].variables["u"])
    uev = np.array(data.groups["ElectronEnergy"].variables["u"])

    Er = np.array(data.variables["Er"])
    data.close()

    desc_grid = desc.grid.LinearGrid(rho=x, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    desc_data = eq.compute(["V(r)", "V_r(r)"], grid=desc_grid)
    V = desc_grid.compress(desc_data["V(r)"])
    Vp = desc_grid.compress(desc_data["V_r(r)"]) / V[-1]
    n = nv[-1, :] / Vp
    p_i = 2.0 / 3.0 * uiv[-1, :] / Vp
    p_e = 2.0 / 3.0 * uev[-1, :] / Vp
    Ti = p_i / n
    Te = p_e / n
    return x, n, Ti, Te, Er[-1, :]


_eqA = desc.io.load("eq6_all_equilibria.h5")
_eqB = desc.io.load("eq7_all_equilibria.h5")

fs = ["stellarator_opteq6.nc", "stellarator_opteq7.nc"]
eqs = [_eqA[-1], _eqB[-1]]
linspecs = ["--", ""]
fig, ax = plt.subplots(1, 3, figsize=(5.0, 2.0), constrained_layout=True)
names = ["A", "B"]
colors = ["r", "b"]

for i in range(0, len(fs)):
    x, n, Ti, Te, Er = load(eqs[i], fs[i])
    ax[0].plot(x, n * 0.1, colors[i], label=names[i], linewidth=1.5)
    ax[0].set_xlabel(r"$\rho$")

    ax[0].set_ylabel(r"n $(10^{20} m^{-3})$")
    ax[0].set_box_aspect(1)

    ax[1].plot(x, Ti, colors[i] + linspecs[0], label=fr"$T_i$, {names[i]}", linewidth=1.5)
    ax[1].plot(x, Te, colors[i] + linspecs[1], label=fr"$T_e$, {names[i]}", linewidth=1.5)
    ax[1].set_xlabel(r"$\rho$")
    ax[1].set_ylabel("T (keV)")
    ax[1].set_box_aspect(1)

    ax[2].plot(x, Er, colors[i], label = names[i], linewidth=1.5)
    ax[2].set_xlabel(r"$\rho$")
    ax[2].set_ylabel("Er (kV/m)")
    ax[2].set_box_aspect(1)
ax[0].legend()
ax[1].legend()
ax[2].legend()
ax[2].set_ylim(bottom=-100)
fig.suptitle("Profiles from optimized stellarator")
fig.savefig("figs/optimized_profiles.pdf", dpi=500)
plt.show()
