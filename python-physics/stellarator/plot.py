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
    x = np.array(data.variables["x"][:])
    nv = np.array(data.groups["Density"].variables["u"][:])
    uiv = np.array(data.groups["IonEnergy"].variables["u"][:])
    uev = np.array(data.groups["ElectronEnergy"].variables["u"][:])

    Er = np.array(data.variables["Er"][:])
    JB = np.array(data.variables["BootstrapCurrent"][:])
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
    return x, n, Ti, Te, Er[-1, :], JB[-1, :]


eq = desc.io.load("./paper/eq_all_equilibria.h5")[-1]

rho, n, Ti, Te, Er, JB = load(eq, "./paper/stellarator_opteq8.nc")

fig, ax = plt.subplots(1, 4, figsize=(10.0, 3.5), constrained_layout=True)

ax[0].plot(rho, n * 0.1, "r", label="n", linewidth=1.5)
ax[0].set_xlabel(r"$\rho$")
ax[0].set_ylabel(r"n $(10^{20} m^{-3})$")
ax[0].legend()
ax[0].set_box_aspect(1)

ax[1].plot(rho, Ti, "r", label=rf"$T_i$, A", linewidth=1.5)
ax[1].plot(rho, Te, "b", label=rf"$T_e$, A", linewidth=1.5)
ax[1].set_xlabel(r"$\rho$")
ax[1].set_ylabel("T (keV)")
ax[1].legend()
ax[1].set_box_aspect(1)

ax[2].plot(rho, Er, "r", label=rf"$E_r$, A", linewidth=1.5)
ax[2].set_xlabel(r"$\rho$")
ax[2].set_ylabel(r"$E_r$ (V/m)")
ax[2].legend()
ax[2].set_box_aspect(1)

ax[3].plot(rho, JB, "r", label=rf"$J||B$, A", linewidth=1.5)
ax[3].set_xlabel(r"$\rho$")
ax[3].set_ylabel(r"$J||B$ (A/m$^2$)")
ax[3].legend()
ax[3].set_box_aspect(1)

plt.show()

