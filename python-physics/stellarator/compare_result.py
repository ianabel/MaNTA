import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import desc.io

plt.rcParams.update({"font.family": "serif", "font.size": 10})

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


_eqs = desc.io.load("eq_amb5_all_equilibria.h5")

fs = ["stellarator_opt0.nc", "stellarator_opt.nc"]
eqs = [_eqs[0], _eqs[-1]]
linspecs = ["--", ""]
fig, ax = plt.subplots(1, 3)

for i in range(0, len(fs)):
    x, n, Ti, Te, Er = load(eqs[i], fs[i])
    ax[0].plot(x, n, "r" + linspecs[i], label="Final", linewidth=1.5)
    ax[0].set_xlabel(r"$\rho$")

    ax[0].set_ylabel(r"n $(10^{20} m^{-3})$")
    ax[0].set_box_aspect(1)

    ax[1].plot(x, Ti, "r" + linspecs[i], label=r"$T_i$", linewidth=1.5)
    ax[1].plot(x, Te, "b" + linspecs[i], label=r"$T_e$", linewidth=1.5)
    ax[1].set_xlabel(r"$\rho$")
    ax[1].set_ylabel("T (keV)")
    ax[1].legend()
    ax[1].set_box_aspect(1)

    ax[2].plot(x, Er, "r" + linspecs[i], linewidth=1.5)
    ax[2].set_xlabel(r"$\rho$")
    ax[2].set_ylabel("Er (kV/m)")
    ax[2].set_box_aspect(1)
fig.suptitle("Profiles from optimized stellarator")
fig.set_figwidth(5.5)
fig.set_figheight(3.0)
fig.tight_layout()
fig.savefig("figs/optimized_profiles.png", dpi=300, bbox_inches="tight")
plt.show()
