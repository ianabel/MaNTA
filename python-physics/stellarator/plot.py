"""Plot the final profiles of a stellarator run.

Handles both shapes of run:

* multi-channel (evolveDensity = True): density, ion and electron energy are
  transported, plus Er (an aux variable, or an "Er" group holding V' Er under
  ErModel = "Diffusive") and the bootstrap current;
* single-channel (evolveDensity = False): only the ion energy is transported.
  The density is the prescribed profile the case was run with, which is not in
  the output, so its parameters are set below (DENSITY_PROFILE). There is no
  Te, Er or bootstrap current to plot.

Edit the settings below and run `python plot.py`.
"""

import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import desc.io
import matplotlib.pyplot as plt
import numpy as np
from desc.equilibrium import EquilibriaFamily
from netCDF4 import Dataset

plt.rcParams.update({"font.family": "serif", "font.size": 8})

# ---- settings ---------------------------------------------------------------

# The runs to plot; several are overlaid, one colour each.
FILES = [
    "./paper/stellarator_opteq_itg_flux.nc",
]
# Legend labels, one per file; None uses the file names.
LABELS = None

# The equilibrium the runs were made on, and which member of the family.
EQ_FILE = "./paper/eq_itg_flux_all_equilibria.h5"
EQ_INDEX = -1

# The prescribed density of a single-channel run, as in its config:
# (EdgeDensity, n0, rhoUpper, alpha). Ignored for multi-channel runs. These
# are paper/desc-optimize-itg-flux.py's.
DENSITY_PROFILE = (0.4, 1.0, 1.0, 4.0)

# -----------------------------------------------------------------------------


def initial_profile(x, edge_value, peak_value, xr, exponent):
    """StellaratorState.initial_profile, repeated so plotting does not need
    the manta build on the path."""
    return (peak_value - edge_value) * (1 - x**exponent / xr**exponent) + edge_value


def load(eq, fname, density_profile):
    """The final profiles of the run in `fname` on its own grid, as a dict.
    Te and Er are present only for a multi-channel run, JB only
    for one that wrote the bootstrap current."""

    data = Dataset(fname)
    x = np.array(data.variables["x"][:])
    single_channel = "Density" not in data.groups

    desc_grid = desc.grid.LinearGrid(rho=x, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    desc_data = eq.compute(["V(r)", "V_r(r)"], grid=desc_grid)
    V = desc_grid.compress(desc_data["V(r)"])
    Vp = desc_grid.compress(desc_data["V_r(r)"]) / V[-1]

    uiv = np.array(data.groups["IonEnergy"].variables["u"][-1, :])
    if single_channel:
        n = initial_profile(x, *density_profile)
    else:
        n = np.array(data.groups["Density"].variables["u"][-1, :]) / Vp

    out = {"rho": x, "n": n, "Ti": 2.0 / 3.0 * uiv / Vp / n}

    if not single_channel:
        uev = np.array(data.groups["ElectronEnergy"].variables["u"][-1, :])
        out["Te"] = 2.0 / 3.0 * uev / Vp / n
        if "Er" in data.groups:
            # ErModel = "Diffusive": the transported variable is V' Er
            out["Er"] = np.array(data.groups["Er"].variables["u"][-1, :]) / Vp
        else:
            out["Er"] = np.array(data.variables["Er"][-1, :])
        if "BootstrapCurrent" in data.variables:  # older runs do not carry it
            out["JB"] = np.array(data.variables["BootstrapCurrent"][-1, :])

    data.close()
    return out


def main():
    labels = LABELS or [os.path.splitext(os.path.basename(f))[0] for f in FILES]
    if len(labels) != len(FILES):
        raise ValueError("LABELS needs one label per file")

    eq = desc.io.load(EQ_FILE)
    if isinstance(eq, EquilibriaFamily):
        eq = eq[EQ_INDEX]

    runs = [load(eq, f, DENSITY_PROFILE) for f in FILES]

    # one panel per quantity any run carries
    panels = [("n", r"n $(10^{20} m^{-3})$"), ("T", "T (keV)")]
    if any("Er" in r for r in runs):
        panels.append(("Er", r"$E_r$ (V/m)"))
    if any("JB" in r for r in runs):
        panels.append(("JB", r"$J||B$ (A/m$^2$)"))

    fig, ax = plt.subplots(
        1, len(panels), figsize=(2.6 * len(panels), 3.2), constrained_layout=True,
        squeeze=False,
    )
    ax = ax[0]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    multi = len(runs) > 1

    # One run: each quantity labelled in its panel, as before. Several: one
    # colour per run, named in a single legend under the panels, with Te
    # dashed beside Ti.
    for k, (run, label) in enumerate(zip(runs, labels)):
        c = colors[k % len(colors)] if multi else None
        for a, (key, _) in zip(ax, panels):
            if key == "n":
                a.plot(run["rho"], run["n"] * 0.1, color=c or "r", label="n", lw=1.5)
            elif key == "T":
                a.plot(run["rho"], run["Ti"], color=c or "r", label=r"$T_i$", lw=1.5)
                if "Te" in run:
                    a.plot(run["rho"], run["Te"], "--" if multi else "-",
                           color=c or "b", label=r"$T_e$", lw=1.5)
            elif key in run:
                name = r"$E_r$" if key == "Er" else r"$J||B$"
                a.plot(run["rho"], run[key], color=c or "r", label=name, lw=1.5)

    for a, (key, ylabel) in zip(ax, panels):
        a.set_xlabel(r"$\rho$")
        a.set_ylabel(ylabel)
        a.set_box_aspect(1)
        if not multi:
            a.legend()
    if multi:
        handles = [plt.Line2D([], [], color=colors[k % len(colors)], lw=1.5)
                   for k in range(len(runs))]
        if any("Te" in r for r in runs):
            handles += [plt.Line2D([], [], color="k", lw=1.5),
                        plt.Line2D([], [], color="k", ls="--", lw=1.5)]
            labels = labels + [r"$T_i$", r"$T_e$"]
        fig.legend(handles, labels, loc="outside lower center",
                   ncol=min(len(labels), 4), frameon=False)

    plt.show()


if __name__ == "__main__":
    main()
